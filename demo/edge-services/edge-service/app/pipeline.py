"""Basic capture pipeline: prefilter + mock model inference."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Any

from PIL import Image

from .config import settings
from .oak_controller import Frame
from .utils import configure_logging, now_iso
from .validation import validate_image_bytes


@dataclass(frozen=True)
class ProcessedFrame:
    image_bytes: bytes
    filename: str
    metadata: dict[str, Any]


def _deterministic_score(payload: bytes) -> float:
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:2], "big")
    return value / 65535.0


def _normalize_pixels_rgb(pixels: list[tuple[int, int, int]]) -> list[float]:
    mean_r, mean_g, mean_b = settings.imagenet_mean
    std_r, std_g, std_b = settings.imagenet_std
    normalized = []
    for r, g, b in pixels:
        normalized.append(((r / 255.0) - mean_r) / std_r)
        normalized.append(((g / 255.0) - mean_g) / std_g)
        normalized.append(((b / 255.0) - mean_b) / std_b)
    return normalized


class Pipeline:
    def __init__(self) -> None:
        self._width = settings.downsample_width
        self._height = settings.downsample_height
        self._logger = configure_logging("pipeline")
        self._logger.info(
            "pipeline init downsample=%sx%s normalize=%s blob_path=%s",
            self._width,
            self._height,
            settings.normalize_inputs,
            settings.blob_path or "none",
        )

    def process(self, frame: Frame, filename: str, image_index: int) -> ProcessedFrame:
        info = validate_image_bytes(frame.data, settings.max_image_bytes)

        if frame.inference:
            metadata = {
                "captured_at": now_iso(),
                "image_index": image_index,
                "filename": filename,
                "source_ext": info.ext,
            }
            metadata.update(frame.inference)
            return ProcessedFrame(image_bytes=frame.data, filename=filename, metadata=metadata)

        image = Image.open(io.BytesIO(frame.data))
        downsampled = image.resize((self._width, self._height))
        gray = downsampled.convert("L")
        pixels = list(gray.getdata())
        avg_luma = sum(pixels) / max(len(pixels), 1)
        prefilter_score = avg_luma / 255.0
        prefilter_passed = prefilter_score >= settings.prefilter_threshold

        metadata: dict[str, Any] = {
            "captured_at": now_iso(),
            "image_index": image_index,
            "filename": filename,
            "prefilter_score": round(prefilter_score, 4),
            "prefilter_passed": prefilter_passed,
            "source_ext": info.ext,
            "blob_path": settings.blob_path,
            "blob_loaded": bool(settings.blob_path),
            "normalize_inputs": settings.normalize_inputs,
            "inference_source": "mock",
        }

        if prefilter_passed:
            rgb_pixels = list(downsampled.convert("RGB").getdata())
            model_input = rgb_pixels
            if settings.normalize_inputs:
                model_input = _normalize_pixels_rgb(rgb_pixels)
                metadata["normalization"] = "imagenet"

            payload = f"{settings.blob_path or 'no_blob'}:{model_input[:10]}".encode("utf-8")
            model_score = _deterministic_score(payload)
            model_passed = model_score >= settings.model_threshold
            metadata.update(
                {
                    "model_score": round(model_score, 4),
                    "model_passed": model_passed,
                    "tag": "model_passed" if model_passed else "model_failed",
                }
            )
        else:
            metadata.update({"model_score": None, "model_passed": None, "tag": "prefilter_failed"})

        return ProcessedFrame(image_bytes=frame.data, filename=filename, metadata=metadata)

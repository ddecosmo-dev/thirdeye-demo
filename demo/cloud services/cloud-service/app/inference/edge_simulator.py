"""Standalone edge-inference simulator for cloud-hosted validation.

This module mirrors the current edge-device logging shape:

{
  "run_id": "run_...",
  "created_at": "...",
  "entries": [
    {
      "frame_index": 1,
      "timestamp": "...",
      "prefilter_passed": true,
      "prefilter_reason": "Passed",
      "mean_intensity": 123.45,
      "blur_variance": 678.9,
      "scenic_score": 0.1234
    }
  ]
}

The simulator is intentionally separate from the live cloud pipeline so we can
exercise the edge behavior in isolation before wiring it into upload handling.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# Prefilter cutoffs mirror the edge-device script.
PREFILTER_MIN_INTENSITY = 20.0
PREFILTER_MAX_INTENSITY = 235.0
PREFILTER_BLUR_THRESHOLD = 100.0


@dataclass(frozen=True)
class EdgeSimulationConfig:
    """Configuration for a single simulated edge run."""

    blur_thresh: float = PREFILTER_BLUR_THRESHOLD
    min_intensity: float = PREFILTER_MIN_INTENSITY
    max_intensity: float = PREFILTER_MAX_INTENSITY


def _load_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("cv2 is required for edge simulation") from exc
    return cv2


def _load_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for edge simulation") from exc
    return torch


def _load_aesthetic_scorer():
    from .model_aesthetic import AestheticScorer

    return AestheticScorer


def run_prefilter(
    frame: np.ndarray,
    blur_thresh: float = PREFILTER_BLUR_THRESHOLD,
    min_intensity: float = PREFILTER_MIN_INTENSITY,
    max_intensity: float = PREFILTER_MAX_INTENSITY,
) -> tuple[bool, str, float, float]:
    """Run the same fast prefilter used on-device."""

    cv2 = _load_cv2()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_intensity = float(np.mean(gray))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if mean_intensity < min_intensity:
        return False, "Too Dark", mean_intensity, laplacian_var
    if mean_intensity > max_intensity:
        return False, "Overexposed", mean_intensity, laplacian_var
    if laplacian_var < blur_thresh:
        return False, "Too Blurry", mean_intensity, laplacian_var

    return True, "Passed", mean_intensity, laplacian_var


def _iter_image_paths(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() != ".zip":
            raise ValueError(f"Unsupported input file: {source}")
        return []

    image_paths = [
        path
        for path in sorted(source.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    return image_paths


def _safe_extract_zip(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue

            relative_path = Path(info.filename)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Unsafe path in zip: {info.filename}")

            target_path = destination / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source_handle, open(target_path, "wb") as target_handle:
                target_handle.write(source_handle.read())


def _load_bgr_image(image_path: Path) -> np.ndarray:
    cv2 = _load_cv2()
    data = image_path.read_bytes()
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise ValueError(f"Unable to decode image: {image_path}")

    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def _score_frame(scorer: Any, frame: np.ndarray) -> tuple[float, np.ndarray]:
    if scorer.processor is None:
        raise RuntimeError("AestheticScorer must be loaded before scoring frames")

    cv2 = _load_cv2()
    torch = _load_torch()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = scorer.processor(images=rgb_frame, return_tensors="pt")
    device = torch.device(getattr(scorer, "_device", "cpu"))
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        return scorer.score(inputs)


def _build_entries(
    image_paths: Iterable[Path],
    scorer: Any,
    config: EdgeSimulationConfig,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

    for frame_index, image_path in enumerate(image_paths, start=1):
        frame = _load_bgr_image(image_path)
        prefilter_passed, prefilter_reason, mean_intensity, blur_variance = run_prefilter(
            frame,
            blur_thresh=config.blur_thresh,
            min_intensity=config.min_intensity,
            max_intensity=config.max_intensity,
        )
        scenic_score, _ = _score_frame(scorer, frame)

        entries.append(
            {
                "frame_index": frame_index,
                "timestamp": dt.datetime.now().isoformat(),
                "prefilter_passed": prefilter_passed,
                "prefilter_reason": prefilter_reason,
                "mean_intensity": round(mean_intensity, 2),
                "blur_variance": round(blur_variance, 2),
                "scenic_score": round(float(scenic_score), 4),
            }
        )

        logger.info(
            "Frame %s prefilter=%s mean=%.1f blur=%.1f scenic=%.4f image=%s",
            frame_index,
            prefilter_reason,
            mean_intensity,
            blur_variance,
            scenic_score,
            image_path.name,
        )

    return entries


def simulate_edge_inference_from_paths(
    image_paths: Iterable[Path],
    output_root: str | Path | None = None,
    run_id: str | None = None,
    results_filename: str = "edge_results.json",
    scorer: Any | None = None,
    config: EdgeSimulationConfig | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Simulate edge inference on an existing set of image paths."""

    output_root_path = Path(output_root) if output_root is not None else Path(tempfile.gettempdir()) / "thirdeye_edge_simulations"
    run_id = run_id or f"run_{dt.datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    run_dir = output_root_path / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if scorer is None:
        AestheticScorer = _load_aesthetic_scorer()
        scorer = AestheticScorer()
        scorer.load("cpu")

    config = config or EdgeSimulationConfig()
    entries = _build_entries(image_paths, scorer, config)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "created_at": dt.datetime.now().isoformat(),
        "entries": entries,
    }

    results_path = run_dir / results_filename
    results_path.write_text(json.dumps(payload, indent=2))
    return results_path, payload


def simulate_edge_inference(
    input_path: str | Path,
    output_root: str | Path | None = None,
    run_id: str | None = None,
    results_filename: str = "edge_results.json",
    scorer: Any | None = None,
    config: EdgeSimulationConfig | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Simulate the current edge inference flow over a folder or zip archive.

    Returns the path to the written JSON file and the in-memory payload.
    """

    source = Path(input_path)
    if not source.exists():
        raise FileNotFoundError(source)

    if source.is_file():
        extracted_dir = Path(tempfile.mkdtemp(prefix="thirdeye_edge_sim_"))
        _safe_extract_zip(source, extracted_dir)
        image_paths = _iter_image_paths(extracted_dir)
    else:
        image_paths = _iter_image_paths(source)

    return simulate_edge_inference_from_paths(
        image_paths=image_paths,
        output_root=output_root,
        run_id=run_id,
        results_filename=results_filename,
        scorer=scorer,
        config=config,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the edge-inference simulator over a zip or folder of images.")
    parser.add_argument("input_path", type=Path, help="Path to a folder of images or a .zip archive.")
    parser.add_argument("--output-root", type=Path, default=None, help="Directory where the run folder and JSON should be written.")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run identifier to use for the output folder.")
    parser.add_argument("--blur-thresh", type=float, default=PREFILTER_BLUR_THRESHOLD, help="Laplacian variance threshold used by the prefilter.")
    parser.add_argument("--min-intensity", type=float, default=PREFILTER_MIN_INTENSITY, help="Lower bound for mean intensity.")
    parser.add_argument("--max-intensity", type=float, default=PREFILTER_MAX_INTENSITY, help="Upper bound for mean intensity.")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    parser = _build_arg_parser()
    args = parser.parse_args()

    config = EdgeSimulationConfig(
        blur_thresh=args.blur_thresh,
        min_intensity=args.min_intensity,
        max_intensity=args.max_intensity,
    )
    results_path, payload = simulate_edge_inference(
        args.input_path,
        output_root=args.output_root,
        run_id=args.run_id,
        config=config,
    )
    print(f"Wrote edge simulation results to {results_path}")
    print(f"Frames processed: {len(payload['entries'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""Lightweight image validation utilities for edge safety checks.

Uses `imghdr` when available; falls back to `Pillow` (`PIL`) for more
robust detection on environments where `imghdr` is missing or cannot
classify raw bytes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import imghdr
except Exception:  # pragma: no cover - defensive for minimal or constrained runtimes
    imghdr = None

try:
    from PIL import Image
    from io import BytesIO
except Exception:
    Image = None


@dataclass(frozen=True)
class ImageInfo:
    ext: str
    kind: str


ALLOWED_IMAGE_TYPES: dict[str, str] = {
    "jpeg": ".jpg",
    "png": ".png",
}


def _detect_with_pillow(data: bytes) -> Optional[str]:
    if Image is None:
        return None
    try:
        with Image.open(BytesIO(data)) as im:
            fmt = (im.format or "").lower()
            return fmt
    except Exception:
        return None


def validate_image_bytes(data: bytes, max_bytes: int) -> ImageInfo:
    if len(data) > max_bytes:
        raise ValueError("Image exceeds max byte limit")

    kind: Optional[str] = None
    if imghdr is not None:
        try:
            kind = imghdr.what(None, data)
        except Exception:
            kind = None

    if not kind:
        kind = _detect_with_pillow(data)

    if not kind or kind not in ALLOWED_IMAGE_TYPES:
        raise ValueError("Unsupported image type")

    return ImageInfo(ext=ALLOWED_IMAGE_TYPES[kind], kind=kind)

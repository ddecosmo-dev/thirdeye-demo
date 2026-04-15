"""Lightweight image validation utilities for edge safety checks."""

from __future__ import annotations

import imghdr
from dataclasses import dataclass


@dataclass(frozen=True)
class ImageInfo:
    ext: str
    kind: str


ALLOWED_IMAGE_TYPES: dict[str, str] = {
    "jpeg": ".jpg",
    "png": ".png",
}


def validate_image_bytes(data: bytes, max_bytes: int) -> ImageInfo:
    if len(data) > max_bytes:
        raise ValueError("Image exceeds max byte limit")

    kind = imghdr.what(None, data)
    if not kind or kind not in ALLOWED_IMAGE_TYPES:
        raise ValueError("Unsupported image type")

    return ImageInfo(ext=ALLOWED_IMAGE_TYPES[kind], kind=kind)

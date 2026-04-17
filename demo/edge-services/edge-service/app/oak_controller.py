"""Frame source abstraction with a mock mode for local testing."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterable

from .config import settings


@dataclass(frozen=True)
class Frame:
    data: bytes
    filename_hint: str
    ext: str | None = None


class FrameSource:
    def start(self) -> None:
        return None

    def next_frame(self) -> Frame | None:
        raise NotImplementedError

    def stop(self) -> None:
        return None


class MockFrameSource(FrameSource):
    def __init__(self, image_dir: str, fps_limit: int) -> None:
        self._image_dir = image_dir
        self._fps_limit = max(1, fps_limit)
        self._files = self._load_files(image_dir)
        self._index = 0
        self._last_emit = 0.0

    def _load_files(self, image_dir: str) -> list[str]:
        entries = []
        for name in sorted(os.listdir(image_dir)):
            if name.lower().endswith((".jpg", ".jpeg", ".png")):
                entries.append(os.path.join(image_dir, name))
        return entries

    def next_frame(self) -> Frame | None:
        if not self._files:
            return None

        min_interval = 1.0 / float(self._fps_limit)
        now = time.time()
        elapsed = now - self._last_emit
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

        path = self._files[self._index]
        self._index = (self._index + 1) % len(self._files)
        self._last_emit = time.time()

        with open(path, "rb") as handle:
            data = handle.read()
        return Frame(data=data, filename_hint=os.path.basename(path))


class OakFrameSource(FrameSource):
    def next_frame(self) -> Frame | None:
        # TODO: Implement DepthAI pipeline capture.
        raise NotImplementedError("Oak camera integration not implemented")


def build_frame_source() -> FrameSource:
    if settings.mock_image_dir:
        return MockFrameSource(settings.mock_image_dir, settings.capture_fps_limit)
    # TODO: Switch to OakFrameSource when DepthAI is available.
    return OakFrameSource()

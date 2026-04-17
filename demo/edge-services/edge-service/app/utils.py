"""Shared helpers for ids, timestamps, and disk checks."""

from __future__ import annotations

import os
import re
import shutil
import time
from datetime import datetime


LABEL_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def utc_timestamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def safe_label(label: str | None) -> str:
    if not label:
        return "run"
    cleaned = LABEL_RE.sub("-", label.strip())
    cleaned = cleaned.strip("-")
    return cleaned or "run"


def create_run_id(label: str | None) -> str:
    # TODO: Add collision handling if two runs start in same second.
    return f"{utc_timestamp()}_{safe_label(label)}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def available_disk_bytes(path: str) -> int:
    usage = shutil.disk_usage(path)
    return usage.free

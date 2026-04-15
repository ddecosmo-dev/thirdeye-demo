"""Shared helpers for run ids, timestamps, and hashing."""

from __future__ import annotations

import hashlib
import os
import re
import time
from datetime import datetime


LABEL_RE = re.compile(r"[^a-zA-Z0-9_-]+")


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


def sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

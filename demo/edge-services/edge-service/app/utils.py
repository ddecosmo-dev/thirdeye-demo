"""Shared helpers for ids, timestamps, and logging."""

from __future__ import annotations

import logging
import os
import re
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
    return f"{utc_timestamp()}_{safe_label(label)}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def configure_logging(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

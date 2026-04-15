"""Filesystem-based run storage with JSON metadata per cycle."""

from __future__ import annotations

import json
import os
from typing import Any

from .config import settings
from .utils import ensure_dir, now_iso


RUNS_DIRNAME = "runs"


def runs_root() -> str:
    root = os.path.join(settings.data_dir, RUNS_DIRNAME)
    ensure_dir(root)
    return root


def run_dir(run_id: str) -> str:
    path = os.path.join(runs_root(), run_id)
    ensure_dir(path)
    return path


def images_dir(run_id: str) -> str:
    path = os.path.join(run_dir(run_id), "images")
    ensure_dir(path)
    return path


def raw_dir(run_id: str) -> str:
    path = os.path.join(run_dir(run_id), "raw")
    ensure_dir(path)
    return path


def metadata_path(run_id: str) -> str:
    return os.path.join(run_dir(run_id), "metadata.json")


def read_metadata(run_id: str) -> dict[str, Any] | None:
    path = metadata_path(run_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_metadata(run_id: str, data: dict[str, Any]) -> None:
    path = metadata_path(run_id)
    data["updated_at"] = now_iso()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def create_metadata(run_id: str, label: str, status: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    base = {
        "run_id": run_id,
        "label": label,
        "status": status,
        "created_at": now_iso(),
    }
    if extra:
        base.update(extra)
    # TODO: Introduce a schema version field for metadata evolution.
    write_metadata(run_id, base)
    return base

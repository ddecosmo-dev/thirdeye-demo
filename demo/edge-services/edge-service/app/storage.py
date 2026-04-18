"""Filesystem-based run storage for processor temp runs."""

from __future__ import annotations

import json
import os
from typing import Any

from .config import settings
from .utils import ensure_dir, now_iso


def runs_root() -> str:
    root = os.path.join(settings.data_dir, settings.runs_dirname)
    ensure_dir(root)
    return root


def run_dir(run_id: str) -> str:
    path = os.path.join(runs_root(), run_id)
    ensure_dir(path)
    return path


def temp_dir(run_id: str) -> str:
    path = os.path.join(run_dir(run_id), settings.temp_dirname)
    ensure_dir(path)
    return path


def temp_images_dir(run_id: str) -> str:
    path = os.path.join(temp_dir(run_id), "images")
    ensure_dir(path)
    return path


def run_metadata_path(run_id: str) -> str:
    return os.path.join(run_dir(run_id), "run.json")


def temp_metadata_path(run_id: str) -> str:
    return os.path.join(temp_dir(run_id), "metadata.jsonl")


def read_run_metadata(run_id: str) -> dict[str, Any] | None:
    path = run_metadata_path(run_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_run_metadata(run_id: str, data: dict[str, Any]) -> None:
    path = run_metadata_path(run_id)
    data["updated_at"] = now_iso()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def update_run_metadata(run_id: str, patch: dict[str, Any]) -> None:
    current = read_run_metadata(run_id) or {}
    current.update(patch)
    write_run_metadata(run_id, current)


def append_image_metadata(run_id: str, payload: dict[str, Any]) -> None:
    path = temp_metadata_path(run_id)
    payload["logged_at"] = now_iso()
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def create_run_metadata(run_id: str, label: str, status: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    base = {
        "run_id": run_id,
        "label": label,
        "status": status,
        "created_at": now_iso(),
    }
    if extra:
        base.update(extra)
    write_run_metadata(run_id, base)
    return base

"""Filesystem-based run storage with JSON metadata per cycle."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

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


def results_path(run_id: str) -> str:
    """Path to results.json for a run."""
    return os.path.join(run_dir(run_id), "results.json")


def read_results(run_id: str) -> dict[str, Any] | None:
    """Read inference results for a run."""
    path = results_path(run_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_results(run_id: str, results: dict[str, Any]) -> None:
    """Write inference results and update metadata."""
    path = results_path(run_id)
    results["saved_at"] = now_iso()
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    
    # Update metadata to mark status as completed
    meta = read_metadata(run_id) or {}
    meta["status"] = "completed"
    meta["results_available"] = True
    write_metadata(run_id, meta)


def embeddings_path(run_id: str) -> str:
    """Path to embeddings.npz for a run (binary format)."""
    return os.path.join(run_dir(run_id), "embeddings.npz")


def read_embeddings(run_id: str) -> np.ndarray | None:
    """Load embeddings from npz format. Returns (N, 384) array or None if not found."""
    path = embeddings_path(run_id)
    if not os.path.exists(path):
        return None
    try:
        data = np.load(path)
        return data["embeddings"]
    except Exception as e:
        import logging
        logging.error(f"Failed to load embeddings: {e}")
        return None


def write_embeddings(run_id: str, embeddings: np.ndarray) -> None:
    """Save embeddings in efficient npz format."""
    path = embeddings_path(run_id)
    np.savez_compressed(path, embeddings=embeddings.astype(np.float32))


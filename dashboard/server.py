"""FastAPI backend for the ThirdEye pipeline dashboard."""

from __future__ import annotations

import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import shutil
import tempfile
import threading
import traceback
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

import pipeline as pl

# ─── App Setup ───────────────────────────────────────────────────────────────
app = FastAPI(title="ThirdEye Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STAGE_LABELS = {
    "extracting": "Extracting zip",
    "loading_models": "Loading ML models",
    "scoring": "Scoring images (tech + aes + obj)",
    "clustering": "Clustering embeddings (HDBSCAN)",
    "tsne": "Computing t-SNE projection",
    "selecting": "Selecting champions",
    "done": "Done",
    "error": "Error",
    "idle": "Idle",
}

_TOTAL_STAGES = 6


# ─── Pipeline State ───────────────────────────────────────────────────────────
@dataclass
class PipelineState:
    status: str = "idle"
    stage: str = "idle"
    stage_index: int = 0
    images_done: int = 0
    images_total: int = 0
    error: str | None = None
    images_dir: Path | None = None
    image_filenames: list[str] = field(default_factory=list)
    scores_df: pd.DataFrame | None = None
    embeddings: np.ndarray | None = None
    epsilon: float = 0.12
    min_cluster_size: int = 2
    ignore_object: bool = False
    edge_results: dict | None = None  # filename -> edge entry from results.json


_state = PipelineState()
_state_lock = threading.Lock()
_cached_models: dict | None = None


def _update_state(**kwargs: Any) -> None:
    with _state_lock:
        for k, v in kwargs.items():
            setattr(_state, k, v)


# ─── Zip Extraction (adapted from cloud-service/app/ingest.py) ────────────────
def _extract_zip_safely(zip_path: Path, dest_dir: Path) -> tuple[list[Path], dict | None]:
    """
    Extract a zip file to dest_dir, enforcing path safety.
    Returns (sorted image paths, edge_results dict keyed by image_filename or None).
    Raises ValueError on unsafe content.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".heic", ".heif"}

    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/")
            parts = name.split("/")
            if any(p == ".." for p in parts) or name.startswith("/"):
                raise ValueError(f"Unsafe path in zip: {info.filename}")

        image_paths: list[Path] = []
        seen_names: dict[str, int] = {}
        edge_results: dict | None = None

        for info in zf.infolist():
            if info.is_dir():
                continue
            basename = Path(info.filename).name
            if basename.startswith("._") or basename.startswith("__MACOSX"):
                continue

            if basename.lower() == "results.json":
                try:
                    with zf.open(info) as src:
                        payload = json.loads(src.read().decode("utf-8"))
                    entries = payload.get("entries", []) if isinstance(payload, dict) else []
                    edge_results = {
                        e["image_filename"]: e
                        for e in entries
                        if isinstance(e, dict) and "image_filename" in e
                    }
                except (json.JSONDecodeError, UnicodeDecodeError):
                    edge_results = None
                continue

            ext = Path(basename).suffix.lower()
            if ext not in image_extensions:
                continue

            # Deduplicate filenames
            if basename in seen_names:
                seen_names[basename] += 1
                stem = Path(basename).stem
                basename = f"{stem}_{seen_names[basename]}{ext}"
            else:
                seen_names[basename] = 0

            dest_path = dest_dir / basename
            with zf.open(info) as src, open(dest_path, "wb") as dst:
                dst.write(src.read())
            image_paths.append(dest_path)

    return sorted(image_paths), edge_results


# ─── Background Pipeline Thread ───────────────────────────────────────────────
def _run_pipeline_thread(zip_path: str, epsilon: float, min_cluster_size: int = 2) -> None:
    global _cached_models

    # Clean up any previous temp directory
    with _state_lock:
        old_dir = _state.images_dir
    if old_dir and old_dir.exists():
        shutil.rmtree(old_dir, ignore_errors=True)

    try:
        # Stage 1: Extract zip
        _update_state(stage="extracting", stage_index=1, images_done=0, images_total=0)
        tmp_root = Path(tempfile.gettempdir()) / f"thirdeye_{uuid.uuid4().hex}"
        images_dir = tmp_root / "images"
        image_paths, edge_results = _extract_zip_safely(Path(zip_path), images_dir)
        Path(zip_path).unlink(missing_ok=True)  # remove temp upload file
        filenames = [p.name for p in image_paths]

        if not image_paths:
            raise ValueError("No image files found in the zip archive.")

        _update_state(
            images_dir=images_dir,
            image_filenames=filenames,
            images_total=len(image_paths),
            edge_results=edge_results,
        )

        # Stage 2: Load models
        _update_state(stage="loading_models", stage_index=2)

        device = _get_device()
        if _cached_models is None:
            _cached_models = pl.load_models(device)
        models = _cached_models

        # Stage 3: Score images
        _update_state(stage="scoring", stage_index=3)

        def progress_cb(n: int) -> None:
            _update_state(images_done=n)

        df, embeddings = pl.score_images(image_paths, models, device, progress_cb)
        df = pl.aggregate_scores(df)

        # Stage 4: Cluster
        _update_state(stage="clustering", stage_index=4)
        df = pl.run_clustering(embeddings, df, epsilon, min_cluster_size)

        # Stage 5: t-SNE
        _update_state(stage="tsne", stage_index=5)
        tsne_coords = pl.run_tsne(embeddings)
        df["tsne_x"] = tsne_coords[:, 0]
        df["tsne_y"] = tsne_coords[:, 1]

        # Stage 6: Select champions
        _update_state(stage="selecting", stage_index=6)
        df = pl.select_champions(df)
        df = pl.assign_rejection_reasons(df)

        _update_state(
            status="done",
            stage="done",
            stage_index=6,
            scores_df=df,
            embeddings=embeddings,
            epsilon=epsilon,
            min_cluster_size=min_cluster_size,
        )

    except Exception as exc:
        _update_state(
            status="error",
            stage="error",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )


def _get_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─── Request/Response Models ──────────────────────────────────────────────────
class RunRequest(BaseModel):
    epsilon: float = Field(default=0.12, ge=0.01, le=0.99)


class ReclusterRequest(BaseModel):
    epsilon: float = Field(ge=0.01, le=0.99)
    min_cluster_size: int = Field(default=2, ge=2, le=50)
    ignore_object: bool = False


# ─── Serialization Helper ─────────────────────────────────────────────────────
def _edge_entry_to_dict(entry: dict) -> dict:
    return {
        "frame_index": int(entry["frame_index"]) if "frame_index" in entry else None,
        "timestamp": entry.get("timestamp"),
        "scenic_score": float(entry["scenic_score"]) if "scenic_score" in entry else None,
        "blur_variance": float(entry["blur_variance"]) if "blur_variance" in entry else None,
        "mean_intensity": float(entry["mean_intensity"]) if "mean_intensity" in entry else None,
        "prefilter_passed": bool(entry["prefilter_passed"]) if "prefilter_passed" in entry else None,
        "prefilter_reason": entry.get("prefilter_reason"),
    }


def _df_to_image_list(df: pd.DataFrame, edge_results: dict | None = None) -> list[dict]:
    records = []
    for _, row in df.iterrows():
        filename = str(row["filename"])
        edge = None
        if edge_results is not None:
            entry = edge_results.get(filename)
            if entry is not None:
                edge = _edge_entry_to_dict(entry)
        records.append(
            {
                "idx": int(row["idx"]),
                "filename": filename,
                "tech_score": float(row.get("technical_score", 0)),
                "aes_score": float(row.get("aesthetic_score", 0)),
                "obj_score": float(row.get("object_aesthetic_score", 0)),
                "tech_norm": float(row.get("tech_norm", 0)),
                "aes_norm": float(row.get("aes_norm", 0)),
                "obj_norm": float(row.get("obj_norm", 0)),
                "aggregated_score": float(row.get("aggregated_score", 0)),
                "tech_penalized": bool(row.get("tech_penalized", False)),
                "cluster_id": int(row.get("cluster_id", -1)),
                "tsne_x": float(row.get("tsne_x", 0)),
                "tsne_y": float(row.get("tsne_y", 0)),
                "is_champion": bool(row.get("is_final_selection", 0) == 1),
                "rejection_reason": row.get("rejection_reason"),
                "edge": edge,
            }
        )
    return records


# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/")
async def serve_index() -> HTMLResponse:
    html_path = Path(__file__).parent / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.get("/health")
async def health() -> dict:
    with _state_lock:
        pipeline_status = _state.status
    return {"status": "ok", "pipeline_status": pipeline_status}


@app.post("/run", status_code=202)
async def run_pipeline(
    file: UploadFile = File(...),
    epsilon: float = Form(default=0.12),
    min_cluster_size: int = Form(default=2),
) -> dict:
    with _state_lock:
        if _state.status == "running":
            raise HTTPException(status_code=409, detail="Pipeline already running")

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are supported")

    if not (0.01 <= epsilon <= 0.99):
        raise HTTPException(status_code=400, detail="epsilon must be between 0.01 and 0.99")

    if not (2 <= min_cluster_size <= 50):
        raise HTTPException(status_code=400, detail="min_cluster_size must be between 2 and 50")

    # Save upload to a temp file so the background thread can read it
    tmp_upload = Path(tempfile.gettempdir()) / f"thirdeye_upload_{uuid.uuid4().hex}.zip"
    with open(tmp_upload, "wb") as f:
        f.write(await file.read())

    _update_state(
        status="running",
        stage="extracting",
        stage_index=0,
        images_done=0,
        images_total=0,
        error=None,
        scores_df=None,
        embeddings=None,
        edge_results=None,
        epsilon=epsilon,
        min_cluster_size=min_cluster_size,
    )

    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(str(tmp_upload), epsilon, min_cluster_size),
        daemon=True,
    )
    thread.start()
    return {"status": "started"}


@app.get("/status")
async def get_status() -> dict:
    with _state_lock:
        s = _state
        pct = 0.0
        if s.status == "running" and s.stage == "scoring" and s.images_total > 0:
            pct = (s.images_done / s.images_total) * 100
        elif s.status == "done":
            pct = 100.0
        return {
            "status": s.status,
            "stage": s.stage,
            "stage_label": _STAGE_LABELS.get(s.stage, s.stage),
            "stage_index": s.stage_index,
            "total_stages": _TOTAL_STAGES,
            "images_done": s.images_done,
            "images_total": s.images_total,
            "pct": round(pct, 1),
            "error": s.error,
        }


@app.get("/results")
async def get_results() -> dict:
    with _state_lock:
        if _state.status != "done" or _state.scores_df is None:
            raise HTTPException(status_code=404, detail="Pipeline has not completed yet")
        df = _state.scores_df.copy()
        epsilon = _state.epsilon
        min_cluster_size = _state.min_cluster_size
        ignore_object = _state.ignore_object
        edge_results = _state.edge_results

    images = _df_to_image_list(df, edge_results)
    champions = [img["idx"] for img in images if img["is_champion"]]
    cluster_ids = [r for r in df["cluster_id"].unique() if r != -1]
    noise_count = int((df["cluster_id"] == -1).sum())

    return {
        "epsilon": epsilon,
        "min_cluster_size": min_cluster_size,
        "ignore_object": ignore_object,
        "cluster_count": len(cluster_ids),
        "noise_count": noise_count,
        "image_count": len(df),
        "images": images,
        "champions": champions,
    }


@app.post("/recluster")
async def recluster(req: ReclusterRequest) -> dict:
    with _state_lock:
        if _state.status != "done" or _state.scores_df is None or _state.embeddings is None:
            raise HTTPException(status_code=404, detail="No completed pipeline run to recluster")
        df = _state.scores_df.copy()
        embeddings = _state.embeddings.copy()

    df = pl.aggregate_scores(df, ignore_object=req.ignore_object)
    df = pl.run_clustering(embeddings, df, req.epsilon, req.min_cluster_size)
    df = pl.select_champions(df)
    df = pl.assign_rejection_reasons(df)

    # Persist updated fields back to state
    with _state_lock:
        _state.scores_df[["aggregated_score", "tech_penalized", "cluster_id", "is_final_selection", "rejection_reason"]] = df[
            ["aggregated_score", "tech_penalized", "cluster_id", "is_final_selection", "rejection_reason"]
        ]
        _state.epsilon = req.epsilon
        _state.min_cluster_size = req.min_cluster_size
        _state.ignore_object = req.ignore_object

    cluster_ids = [r for r in df["cluster_id"].unique() if r != -1]
    noise_count = int((df["cluster_id"] == -1).sum())
    champions = df.index[df["is_final_selection"] == 1].tolist()

    # Return re-computed fields including aggregated_score (changes when ignore_object toggles)
    updates = []
    for _, row in df.iterrows():
        updates.append(
            {
                "idx": int(row["idx"]),
                "aggregated_score": float(row["aggregated_score"]),
                "cluster_id": int(row["cluster_id"]),
                "is_champion": bool(row["is_final_selection"] == 1),
                "rejection_reason": row.get("rejection_reason"),
            }
        )

    return {
        "epsilon": req.epsilon,
        "cluster_count": len(cluster_ids),
        "noise_count": noise_count,
        "champions": champions,
        "updates": updates,
    }


@app.get("/images/{filename}")
async def serve_image(filename: str) -> FileResponse:
    with _state_lock:
        images_dir = _state.images_dir

    if images_dir is None:
        raise HTTPException(status_code=404, detail="No images available")

    # Path traversal protection
    safe_name = Path(filename).name
    full_path = images_dir / safe_name
    if not full_path.resolve().is_relative_to(images_dir.resolve()):
        raise HTTPException(status_code=403, detail="Access denied")
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(
        path=str(full_path),
        headers={"Cache-Control": "public, max-age=3600"},
    )

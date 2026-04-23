"""Cloud API entrypoint for cycle control and image ingest."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import settings
from .cycles import notify_edge
from .ingest import ingest_zip
from .inference import InferenceRunner, dataframe_to_results_json
from .models import (
    CycleActionResponse,
    IngestResponse,
    InferenceRequest,
    InferenceResponse,
    StartCycleRequest,
    StartCycleResponse,
    TSNERequest,
    TSNEResponse,
)
from .storage import create_metadata, read_metadata, run_dir, runs_root, write_metadata, write_results, read_results, write_embeddings, read_embeddings
from .utils import create_run_id, now_iso, safe_label, sha256_file

# Configure logging to output all messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
for name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
    logging.getLogger(name).setLevel(logging.INFO)

logger = logging.getLogger(__name__)

app = FastAPI(title="ThirdEye Cloud Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models cache and progress tracking
_MODELS_CACHE = None
_DEVICE = "cpu"  # Will be set at startup
_RUN_PROGRESS = {}  # {run_id: {"stage": str, "images_done": int, "images_total": int}}

@app.on_event("startup")
async def startup_event():
    """Pre-load all models at startup."""
    global _MODELS_CACHE, _DEVICE
    
    # Detect GPU
    try:
        import torch
        if torch.cuda.is_available():
            _DEVICE = "cuda"
            logger.warning(f"🎮 GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            _DEVICE = "cpu"
            logger.warning("⚠️  No GPU found, using CPU (inference will be slow)")
    except ImportError:
        _DEVICE = "cpu"
    
    logger.warning("=" * 70)
    logger.warning(f"LOADING ALL MODELS AT STARTUP (Device: {_DEVICE.upper()})")
    logger.warning("=" * 70)
    try:
        from .inference.runner import load_models
        _MODELS_CACHE = load_models(_DEVICE)
        logger.warning("=" * 70)
        logger.warning("ALL MODELS LOADED SUCCESSFULLY AT STARTUP")
        logger.warning("=" * 70)
    except Exception as e:
        logger.error(f"Failed to load models at startup: {e}", exc_info=True)
        raise

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# TODO: Add auth (shared secret/JWT) before exposing publicly.
@app.post("/cycle/start", response_model=StartCycleResponse)
async def start_cycle(payload: StartCycleRequest) -> StartCycleResponse:
    label = safe_label(payload.label)
    run_id = create_run_id(label)

    # TODO: Add server-side validation of config schema once finalized.
    create_metadata(
        run_id,
        label=label,
        status="running",
        extra={
            "requested_duration_seconds": payload.duration_seconds,
            "config": payload.config or {},
            "started_at": now_iso(),
        },
    )

    # TODO: Include run_id in the edge start payload once edge supports it.
    await notify_edge("start", payload.model_dump())

    return StartCycleResponse(run_id=run_id, label=label, status="running")


# TODO: Support idempotent stop requests with a "stopping" state.
@app.post("/cycle/stop", response_model=CycleActionResponse)
async def stop_cycle(run_id: str) -> CycleActionResponse:
    meta = read_metadata(run_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Run not found")
    meta["status"] = "stopped"
    meta["ended_at"] = now_iso()
    write_metadata(run_id, meta)

    # TODO: Retry notify_edge or enqueue if edge is offline.
    await notify_edge("stop", {"run_id": run_id})

    return CycleActionResponse(run_id=run_id, status="stopped")


# TODO: Track abort reason history for debugging.
@app.post("/cycle/abort", response_model=CycleActionResponse)
async def abort_cycle(run_id: str, reason: str | None = None) -> CycleActionResponse:
    meta = read_metadata(run_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Run not found")
    meta["status"] = "aborted"
    meta["ended_at"] = now_iso()
    if reason:
        meta["abort_reason"] = reason
    write_metadata(run_id, meta)

    # TODO: Add audit log event for abort requests.
    await notify_edge("abort", {"run_id": run_id, "reason": reason})

    return CycleActionResponse(run_id=run_id, status="aborted")


# TODO: Add checksum validation against client-provided value.
@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    run_id: str | None = Form(default=None),
    metadata_json: str | None = Form(default=None),
    file: UploadFile = File(...),
) -> IngestResponse:
    if run_id is None:
        run_id = create_run_id("ingest")
        # TODO: Replace "ingest" label with client-provided cycle label.
        create_metadata(run_id, label="ingest", status="uploaded")
    else:
        run_dir(run_id)

    # TODO: Add per-file image validation (format, resolution, size).
    result = ingest_zip(run_id, file, metadata_json)
    checksum = sha256_file(result["raw_path"])

    meta = read_metadata(run_id) or {}
    meta.setdefault("ingest", {})
    meta["ingest"]["checksum_sha256"] = checksum
    meta["status"] = meta.get("status", "uploaded")
    write_metadata(run_id, meta)

    return IngestResponse(
        run_id=run_id,
        status=meta["status"],
        file_count=result["file_count"],
        total_uncompressed_bytes=result["total_uncompressed_bytes"],
        checksum_sha256=checksum,
    )


# TODO: Add pagination and filters (status, date range, label).
@app.get("/runs")
async def list_runs() -> dict[str, Any]:
    root = runs_root()
    runs = []
    for name in sorted(os.listdir(root)):
        meta = read_metadata(name)
        if meta:
            runs.append(meta)
    return {"runs": runs}


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    meta = read_metadata(run_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Run not found")
    return meta


@app.get("/runs/{run_id}/images/{filename}")
async def get_run_image(run_id: str, filename: str):
    """Serve an image from a run's images directory."""
    run_path = os.path.join(runs_root(), run_id)
    if not os.path.exists(run_path):
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    images_directory = os.path.join(run_path, "images")
    if not os.path.exists(images_directory):
        raise HTTPException(status_code=404, detail=f"No images directory for run {run_id}")

    image_path = os.path.join(images_directory, filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"Image {filename} not found for run {run_id}")

    content_type = "image/jpeg"
    if filename.lower().endswith(".png"):
        content_type = "image/png"
    elif filename.lower().endswith(".gif"):
        content_type = "image/gif"
    elif filename.lower().endswith(".webp"):
        content_type = "image/webp"

    return FileResponse(image_path, media_type=content_type)


# ─── Background Inference Task ────────────────────────────────────────────────
async def _run_inference_background(
    run_id: str,
    image_paths: list[Path],
    epsilon: float,
    min_cluster_size: int,
    ignore_object: bool,
    device: str,
) -> None:
    """Run inference in background and update _RUN_PROGRESS."""
    try:
        # Define progress callback
        def progress_callback(done: int, total: int):
            global _RUN_PROGRESS
            _RUN_PROGRESS[run_id] = {
                "stage": "scoring",
                "images_done": done,
                "images_total": total,
            }
            # Only log every 10 images or on the last one
            if done % 10 == 0 or done == total:
                pct = int((done / max(total, 1)) * 100)
                logger.info(f"Progress: {done}/{total} images scored ({pct}%) for run {run_id}")
        
        def _run_inference_sync() -> dict:
            runner = InferenceRunner(device=device, pre_loaded_models=_MODELS_CACHE)
            return runner.run(
                image_paths=image_paths,
                epsilon=epsilon,
                min_cluster_size=min_cluster_size,
                ignore_object=ignore_object,
                progress_cb=progress_callback,
            )
        
        # Initialize progress
        _RUN_PROGRESS[run_id] = {"stage": "scoring", "images_done": 0, "images_total": len(image_paths)}
        
        # Run inference in thread
        results = await asyncio.to_thread(_run_inference_sync)
        
        # Convert results to JSON-serializable format
        _RUN_PROGRESS[run_id] = {"stage": "converting", "images_done": len(image_paths), "images_total": len(image_paths)}
        df = results["df"]
        embeddings = results["embeddings"]
        results_list = dataframe_to_results_json(df, embeddings)
        
        # Save embeddings separately for runtime TSNE computation
        _RUN_PROGRESS[run_id] = {"stage": "saving", "images_done": len(image_paths), "images_total": len(image_paths)}
        write_embeddings(run_id, embeddings)
        logger.info(f"Saved {len(embeddings)} embeddings (shape: {embeddings.shape})")
        
        # Count champions
        champion_count = int(df["is_final_selection"].sum())
        
        # Save results
        results_data = {
            "run_id": run_id,
            "image_count": len(image_paths),
            "champion_count": champion_count,
            "inference_params": {
                "epsilon": epsilon,
                "min_cluster_size": min_cluster_size,
                "ignore_object": ignore_object,
                "device": device,
            },
            "results": results_list,
        }
        
        write_results(run_id, results_data)
        
        logger.info(f"✅ Inference complete for run {run_id}: {champion_count} champions selected")
        
        # Mark as complete
        _RUN_PROGRESS[run_id] = {"stage": "completed", "images_done": len(image_paths), "images_total": len(image_paths)}
        
        # Update metadata
        meta = read_metadata(run_id) or {}
        meta["status"] = "completed"
        write_metadata(run_id, meta)
        
    except Exception as e:
        logger.error(f"❌ Background inference failed for run {run_id}: {e}", exc_info=True)
        _RUN_PROGRESS[run_id] = {"stage": "error", "images_done": 0, "images_total": 0}
        meta = read_metadata(run_id) or {}
        meta["status"] = "error"
        meta["error"] = str(e)
        write_metadata(run_id, meta)


# ─── Inference Endpoint ───────────────────────────────────────────────────────
@app.post("/infer", status_code=202)
async def run_inference(payload: InferenceRequest) -> dict[str, Any]:
    """
    Trigger ML inference on images in a run. Returns immediately (202 Accepted).
    Actual inference runs in background. Poll /progress/{run_id} to track progress.
    """
    run_id = payload.run_id
    logger.info(f"🚀 Inference request received for run_id: {run_id}")
    
    meta = read_metadata(run_id)
    if not meta:
        logger.error(f"Metadata not found for run {run_id}")
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    images_directory = os.path.join(run_dir(run_id), "images")
    logger.info(f"Looking for images in: {images_directory}")
    
    if not os.path.exists(images_directory):
        logger.error(f"Images directory does not exist: {images_directory}")
        raise HTTPException(status_code=400, detail=f"No images directory for run {run_id} at {images_directory}")
    
    # Collect image paths
    image_paths = []
    all_files = os.listdir(images_directory)
    logger.info(f"Found {len(all_files)} files in images directory")
    
    for filename in sorted(all_files):
        if filename.lower() in [".gitkeep", "metadata.json", "config.json"]:
            continue
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic")):
            image_paths.append(Path(images_directory) / filename)
    
    logger.info(f"Filtered to {len(image_paths)} image files")
    
    if not image_paths:
        logger.error(f"No valid image files found in {images_directory}")
        raise HTTPException(status_code=400, detail=f"No images found in run {run_id}. Found files: {all_files[:10]}")
    
    try:
        # Update status to inferencing
        meta["status"] = "inferencing"
        write_metadata(run_id, meta)
        
        logger.info(f"Starting inference on {len(image_paths)} images for run {run_id}")
        
        # Use the device determined at startup - models are already loaded on this device
        device = _DEVICE
        logger.warning(f"🎮 Using startup device: {device}")
        
        # Start background inference task - returns immediately
        asyncio.create_task(_run_inference_background(
            run_id=run_id,
            image_paths=image_paths,
            epsilon=payload.epsilon,
            min_cluster_size=payload.min_cluster_size,
            ignore_object=payload.ignore_object,
            device=device,
        ))
        
        # Return immediately with 202 Accepted
        return {
            "run_id": run_id,
            "status": "accepted",
            "message": f"Inference started for {len(image_paths)} images. Poll /progress/{run_id} to track progress.",
            "image_count": len(image_paths),
        }
    
    except Exception as e:
        logger.error(f"Failed to start inference for run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start inference: {str(e)}")


@app.get("/progress/{run_id}")
async def get_progress(run_id: str) -> dict[str, Any]:
    """
    Lightweight endpoint for real-time inference progress.
    Returns: {stage, images_done, images_total, percent_complete}
    """
    global _RUN_PROGRESS
    
    if run_id not in _RUN_PROGRESS:
        # If not tracking yet, return 0% with unknown totals
        logger.debug(f"📡 Progress endpoint called for {run_id} (not tracking yet)")
        return {
            "stage": "pending",
            "images_done": 0,
            "images_total": 0,
            "percent_complete": 0,
        }
    
    progress = _RUN_PROGRESS[run_id]
    images_done = progress.get("images_done", 0)
    images_total = progress.get("images_total", 1)  # Avoid division by zero
    percent = round((images_done / max(images_total, 1)) * 100) if images_total > 0 else 0
    
    logger.debug(f"📡 Progress endpoint: {run_id} -> {images_done}/{images_total} ({percent}%)")
    
    return {
        "stage": progress.get("stage", "unknown"),
        "images_done": images_done,
        "images_total": images_total,
        "percent_complete": percent,
    }


@app.get("/runs/{run_id}/results")
async def get_results(run_id: str) -> dict[str, Any]:
    """Retrieve inference results for a run."""
    results = read_results(run_id)
    if not results:
        raise HTTPException(status_code=404, detail=f"No results found for run {run_id}")
    return results


@app.post("/tsne", response_model=TSNEResponse)
async def compute_tsne(payload: TSNERequest) -> TSNEResponse:
    """
    Compute t-SNE visualization from saved embeddings.
    
    This allows re-projecting with different perplexity parameters
    to demonstrate how visualization changes without re-running inference.
    """
    run_id = payload.run_id
    
    # Load saved embeddings
    embeddings = read_embeddings(run_id)
    if embeddings is None:
        raise HTTPException(status_code=404, detail=f"No embeddings found for run {run_id}")
    
    # Load results to get image metadata
    results_data = read_results(run_id)
    if not results_data:
        raise HTTPException(status_code=404, detail=f"No results found for run {run_id}")
    
    results_list = results_data.get("results", [])
    if len(results_list) != len(embeddings):
        logger.warning(f"Mismatch: {len(results_list)} results vs {len(embeddings)} embeddings")
    
    try:
        # Import here to avoid circular dependency and allow optional GPU
        from .inference.runner import run_tsne
        
        logger.info(f"Computing t-SNE for run {run_id} with perplexity={payload.perplexity}")
        tsne_coords = run_tsne(embeddings, perplexity=payload.perplexity, seed=payload.seed)
        
        # Build response with coordinates
        tsne_response = []
        for idx, (coord, result) in enumerate(zip(tsne_coords, results_list)):
            tsne_response.append({
                "index": result["index"],
                "filename": result["filename"],
                "x": float(coord[0]),
                "y": float(coord[1]),
            })
        
        logger.info(f"t-SNE complete for run {run_id}")
        
        return TSNEResponse(
            run_id=run_id,
            status="completed",
            image_count=len(embeddings),
            perplexity=payload.perplexity,
            tsne_coordinates=tsne_response,
        )
    
    except Exception as e:
        logger.error(f"t-SNE computation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"t-SNE computation failed: {str(e)}")

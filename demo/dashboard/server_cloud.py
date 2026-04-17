"""FastAPI backend for the ThirdEye pipeline dashboard (cloud-connected version).

This dashboard now delegates model inference to the cloud service.
The dashboard handles zip uploads, status polling, and result display.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
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
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from cloud_client import CloudServiceClient

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App Setup ───────────────────────────────────────────────────────────────
app = FastAPI(title="ThirdEye Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cloud service client configuration
CLOUD_SERVICE_URL = os.getenv("CLOUD_SERVICE_URL", "http://localhost:8001")
cloud_client = CloudServiceClient(base_url=CLOUD_SERVICE_URL)

_STAGE_LABELS = {
    "pending": "Preparing inference",
    "extracting": "Uploading to cloud",
    "loading_models": "Loading ML models on cloud",
    "scoring": "Scoring images (tech + aes + obj)",
    "clustering": "Clustering embeddings (HDBSCAN)",
    "converting": "Converting results",
    "saving": "Saving embeddings",
    "retrieving": "Retrieving results from cloud",
    "tsne": "Computing t-SNE projection",
    "selecting": "Selecting champions",
    "completed": "Inference completed on cloud",
    "done": "Done",
    "error": "Error",
    "idle": "Idle",
}

_TOTAL_STAGES = 6


# ─── Helper: Extract top-level folder name from zip ──────────────────────────
def _get_zip_folder_name(zip_path: str) -> str:
    """Extract the top-level folder name from a zip file."""
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
        if not names:
            return "dataset"
        # Get the first path component (top-level folder)
        first = names[0].split('/')[0]
        return first if first else names[0].split('/')[1] if len(names[0].split('/')) > 1 else "dataset"


# ─── Pipeline State ───────────────────────────────────────────────────────────
@dataclass
class PipelineState:
    status: str = "idle"
    stage: str = "idle"
    stage_index: int = 0
    images_done: int = 0
    images_total: int = 0
    error: str | None = None
    run_id: str | None = None
    scores_df: pd.DataFrame | None = None
    embeddings: np.ndarray | None = None
    epsilon: float = 0.12
    min_cluster_size: int = 2
    ignore_object: bool = False
    results: dict[str, Any] | None = None


_state = PipelineState()
_state_lock = threading.Lock()


def _update_state(**kwargs: Any) -> None:
    with _state_lock:
        for k, v in kwargs.items():
            setattr(_state, k, v)


def _get_state() -> dict[str, Any]:
    with _state_lock:
        # Calculate progress percentage
        if _state.images_total > 0:
            pct = round((_state.images_done / _state.images_total) * 100)
        else:
            pct = 0
        
        return {
            "status": _state.status,
            "stage": _state.stage,
            "stage_label": _STAGE_LABELS.get(_state.stage, _state.stage),
            "stage_index": _state.stage_index,
            "images_done": _state.images_done,
            "images_total": _state.images_total,
            "pct": pct,
            "error": _state.error,
            "run_id": _state.run_id,
        }


# ─── Background Inference Thread ──────────────────────────────────────────────
async def _async_infer_and_poll(
    run_id: str,
    zip_path: str,
    epsilon: float,
    min_cluster_size: int,
    ignore_object: bool,
) -> None:
    """
    Run inference on cloud service and poll for results.
    This runs in a background thread via asyncio.run().
    """
    try:
        # Stage 1: Upload zip to cloud
        _update_state(stage="extracting", stage_index=1, images_done=0, images_total=0)
        logger.info(f"Uploading zip to cloud service for run {run_id}")
        
        ingest_resp = await cloud_client.ingest_zip(run_id, zip_path)
        file_count = ingest_resp.get("file_count", 0)
        logger.info(f"Ingested {file_count} files")
        
        _update_state(images_total=file_count)
        
        # Clean up local zip
        Path(zip_path).unlink(missing_ok=True)
        
        # Stage 2-6: Wait for cloud service to complete
        # Stage 2-6: Trigger inference on cloud service (returns immediately with 202)
        logger.info(f"Triggering inference on cloud service")
        _update_state(stage="loading_models", stage_index=2)
        
        # Call /infer but DON'T wait for response - it returns 202 Accepted immediately
        print(f"\n\n📡 CALLING /infer endpoint (should return immediately with 202)\n")
        try:
            infer_resp = await cloud_client.run_inference(
                run_id=run_id,
                epsilon=epsilon,
                min_cluster_size=min_cluster_size,
                ignore_object=ignore_object,
            )
            print(f"✅ /infer response received: {infer_resp}\n")
            logger.info(f"Inference triggered response: {infer_resp}")
        except Exception as e:
            print(f"⚠️  /infer call failed (but inference may still be running): {e}\n")
            logger.warning(f"Inference trigger warning: {e}")
        
        # Poll for progress from cloud service during inference
        # This should start immediately now that /infer returns 202
        print(f"\n🚀🚀🚀 STARTING PROGRESS POLL for run {run_id} 🚀🚀🚀\n")
        logger.info(f"🚀 STARTING PROGRESS POLL for run {run_id}")
        max_wait = 3600  # 1 hour timeout
        elapsed = 0
        poll_interval = 1  # Check every second
        poll_count = 0
        
        while elapsed < max_wait:
            poll_count += 1
            try:
                print(f"  [Poll #{poll_count}] Requesting progress... (elapsed: {elapsed}s)")
                progress = await cloud_client.get_progress(run_id)
                stage = progress.get("stage", "unknown")
                images_done = progress.get("images_done", 0)
                images_total = progress.get("images_total", file_count)
                percent = progress.get("percent_complete", 0)
                
                print(f"  [Poll #{poll_count}] ✓ Got: {stage} - {images_done}/{images_total} ({percent}%)")
                logger.info(f"📊 CLOUD PROGRESS: {stage} - {images_done}/{images_total} ({percent}%)")
                
                # Map cloud stages to dashboard stages
                if stage == "scoring":
                    _update_state(stage="scoring", stage_index=3, images_done=images_done, images_total=images_total)
                elif stage == "converting":
                    _update_state(stage="clustering", stage_index=4, images_done=images_total, images_total=images_total)
                elif stage == "saving":
                    _update_state(stage="selecting", stage_index=5, images_done=images_total, images_total=images_total)
                elif stage == "completed":
                    print(f"  [Poll #{poll_count}] ✅ Inference COMPLETED")
                    logger.info(f"✅ Inference completed on cloud service")
                    break
                
            except Exception as e:
                print(f"  [Poll #{poll_count}] ❌ ERROR: {e}")
                logger.error(f"❌ Error polling progress: {e}", exc_info=True)
            
            
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        
        # Poll for results (cloud may still be writing to disk)
        _update_state(stage="retrieving", stage_index=5.5)
        results = await cloud_client.wait_for_results(run_id, max_wait_seconds=120, poll_interval=2)
        
        if not results:
            raise Exception("Timeout waiting for inference results from cloud service")
        
        logger.info(f"Received results: {champion_count} champions from {file_count} images")
        
        # Process results for dashboard (do NOT compute TSNE upfront, store embeddings instead)
        results_list = results.get("results", [])
        
        # Reconstruct dataframe from results
        df_data = []
        embeddings_list = []
        
        for img_result in results_list:
            df_data.append({
                "idx": img_result["index"],
                "filename": img_result["filename"],
                "technical_score": img_result["scores"]["technical"],
                "aesthetic_score": img_result["scores"]["aesthetic"],
                "object_aesthetic_score": img_result["scores"]["object"],
                "tech_norm": img_result["normalized_scores"]["tech_norm"],
                "aes_norm": img_result["normalized_scores"]["aes_norm"],
                "obj_norm": img_result["normalized_scores"]["obj_norm"],
                "aggregated_score": img_result["aggregated_score"],
                "cluster_id": img_result["cluster_id"],
                "is_final_selection": img_result["is_champion"],
                "tech_penalized": img_result["tech_penalized"],
                "rejection_reason": img_result["rejection_reason"],
            })
            embeddings_list.append(np.array(img_result["embedding"]))
        
        df = pd.DataFrame(df_data)
        embeddings = np.array(embeddings_list)
        
        _update_state(
            status="done",
            stage="done",
            stage_index=6,
            images_done=file_count,
            scores_df=df,
            embeddings=embeddings,
            results=results,
            epsilon=epsilon,
            min_cluster_size=min_cluster_size,
            ignore_object=ignore_object,
        )
        
        logger.info("Pipeline complete")

    except Exception as exc:
        _update_state(
            status="error",
            stage="error",
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
        logger.error(f"Pipeline error: {exc}", exc_info=True)


def _run_inference_thread(
    run_id: str, zip_path: str, epsilon: float, min_cluster_size: int, ignore_object: bool
) -> None:
    """Wrapper to run async code in a thread."""
    try:
        asyncio.run(_async_infer_and_poll(run_id, zip_path, epsilon, min_cluster_size, ignore_object))
    except Exception as e:
        logger.error(f"Thread error: {e}", exc_info=True)


# ─── Request/Response Models ──────────────────────────────────────────────────
class RunRequest(BaseModel):
    epsilon: float = Field(default=0.12, ge=0.01, le=0.99)
    min_cluster_size: int = Field(default=2, ge=2, le=50)
    ignore_object: bool = False


class ReclusterRequest(BaseModel):
    epsilon: float = Field(ge=0.01, le=0.99)
    min_cluster_size: int = Field(default=2, ge=2, le=50)
    ignore_object: bool = False


# ─── Serialization Helper ─────────────────────────────────────────────────────
def _df_to_image_list(df: pd.DataFrame) -> list[dict]:
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "idx": int(row["idx"]),
                "filename": str(row["filename"]),
                "tech_score": float(row.get("technical_score", 0)),
                "aes_score": float(row.get("aesthetic_score", 0)),
                "obj_score": float(row.get("object_aesthetic_score", 0)),
                "tech_norm": float(row.get("tech_norm", 0)),
                "aes_norm": float(row.get("aes_norm", 0)),
                "obj_norm": float(row.get("obj_norm", 0)),
                "aggregated_score": float(row.get("aggregated_score", 0)),
                "tech_penalized": bool(row.get("tech_penalized", False)),
                "cluster_id": int(row.get("cluster_id", -1)),
                "is_champion": bool(row.get("is_final_selection", 0) == 1),
                "rejection_reason": row.get("rejection_reason"),
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
    state = _get_state()
    cloud_healthy = await cloud_client.health()
    return {
        "status": "ok",
        "pipeline_status": state["status"],
        "cloud_service": "ok" if cloud_healthy else "unavailable",
    }


@app.post("/run", status_code=202)
async def run_pipeline(
    file: UploadFile = File(...),
    epsilon: float = Form(default=0.12),
    min_cluster_size: int = Form(default=2),
    ignore_object: bool = Form(default=False),
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

    # Save upload to temp file first
    tmp_upload = Path(tempfile.gettempdir()) / f"thirdeye_{uuid.uuid4().hex[:8]}.zip"
    
    try:
        with open(tmp_upload, "wb") as f:
            content = await file.read()
            f.write(content)
            if f.tell() == 0:
                raise ValueError("Empty file uploaded")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to save upload: {str(e)}")
    
    # Extract run ID from zip folder name
    try:
        run_id = _get_zip_folder_name(str(tmp_upload))
        # Sanitize the run_id (replace spaces and special chars)
        run_id = "".join(c if c.isalnum() or c in "_-" else "_" for c in run_id)
    except Exception as e:
        logger.warning(f"Could not extract folder name from zip: {e}, using random ID")
        run_id = f"dataset_{uuid.uuid4().hex[:8]}"
    
    _update_state(
        status="running",
        stage="extracting",
        stage_index=0,
        images_done=0,
        images_total=0,
        error=None,
        run_id=run_id,
        epsilon=epsilon,
        min_cluster_size=min_cluster_size,
        ignore_object=ignore_object,
    )

    # Start background thread for inference
    thread = threading.Thread(
        target=_run_inference_thread,
        args=(run_id, str(tmp_upload), epsilon, min_cluster_size, ignore_object),
        daemon=True,
    )
    thread.start()

    return {
        "run_id": run_id,
        "status": "starting",
        "message": "Pipeline started, streaming to cloud service",
    }


@app.get("/status")
async def get_status() -> dict:
    return _get_state()


@app.get("/results")
async def get_results_endpoint() -> dict:
    with _state_lock:
        if _state.scores_df is None or _state.embeddings is None:
            raise HTTPException(status_code=404, detail="No results available")
        
        df = _state.scores_df
        images = _df_to_image_list(df)
        champions = [img for img in images if img["is_champion"]]
        
        # Count clusters and noise
        cluster_count = int(df[df["cluster_id"] >= 0]["cluster_id"].nunique())
        noise_count = int((df["cluster_id"] == -1).sum())
        
        return {
            "total_images": len(images),
            "total_champions": len(champions),
            "images": images,
            "champions": champions,
            "cluster_count": cluster_count,
            "noise_count": noise_count,
            "parameters": {
                "epsilon": _state.epsilon,
                "min_cluster_size": _state.min_cluster_size,
                "ignore_object": _state.ignore_object,
            },
        }


@app.post("/recluster")
async def recluster(request: ReclusterRequest) -> dict:
    """Re-cluster results with different parameters (local operation)."""
    with _state_lock:
        if _state.embeddings is None or _state.scores_df is None:
            raise HTTPException(status_code=400, detail="No embeddings available to recluster")
        
        df = _state.scores_df.copy()
        embeddings = _state.embeddings
    
    try:
        # Import pipeline utilities for local reclustering
        import pipeline as pl
        
        # Recalculate aggregated_score based on ignore_object parameter
        # Constants from cloud service
        W_AES = 0.6
        W_OBJ = 0.4
        TECH_FLOOR = 3.0
        
        if request.ignore_object:
            df["aggregated_score"] = df["aes_norm"]
        else:
            df["aggregated_score"] = W_AES * df["aes_norm"] + W_OBJ * df["obj_norm"]
        
        # Apply tech floor penalty
        tech_penalized = df["tech_norm"] < TECH_FLOOR
        df.loc[tech_penalized, "aggregated_score"] *= 0.5
        df["tech_penalized"] = tech_penalized
        
        # Re-cluster with new parameters
        df = pl.run_clustering(embeddings, df, request.epsilon, request.min_cluster_size)
        df = pl.select_champions(df)
        df = pl.assign_rejection_reasons(df)
        
        _update_state(
            scores_df=df,
            epsilon=request.epsilon,
            min_cluster_size=request.min_cluster_size,
            ignore_object=request.ignore_object,
        )
        
        # Build update list for each image
        updates = []
        for _, row in df.iterrows():
            updates.append({
                "idx": int(row["idx"]),
                "aggregated_score": float(row.get("aggregated_score", 0)),
                "cluster_id": int(row.get("cluster_id", -1)),
                "is_champion": bool(row.get("is_final_selection", 0) == 1),
                "rejection_reason": row.get("rejection_reason"),
            })
        
        # Get champions
        champions = [img for img in updates if img["is_champion"]]
        
        # Count clusters and noise
        cluster_count = int(df[df["cluster_id"] >= 0]["cluster_id"].nunique())
        noise_count = int((df["cluster_id"] == -1).sum())
        
        logger.info(f"Recluster: eps={request.epsilon}, mcs={request.min_cluster_size}, ignore_obj={request.ignore_object} -> {cluster_count} clusters, {len(champions)} champions")
        
        return {
            "updates": updates,
            "champions": champions,
            "cluster_count": cluster_count,
            "noise_count": noise_count,
            "parameters": {
                "epsilon": request.epsilon,
                "min_cluster_size": request.min_cluster_size,
                "ignore_object": request.ignore_object,
            },
        }
    except Exception as e:
        logger.error(f"Recluster failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Reclustering failed: {str(e)}")


@app.get("/image/{index}")
async def get_image(index: int) -> dict:
    """Get details for a specific image."""
    with _state_lock:
        if _state.scores_df is None:
            raise HTTPException(status_code=404, detail="No results available")
        
        df = _state.scores_df
        if index < 0 or index >= len(df):
            raise HTTPException(status_code=404, detail=f"Image {index} not found")
        
        row = df.iloc[index]
        
        # Get embedding if available
        embedding = None
        if _state.embeddings is not None and index < len(_state.embeddings):
            embedding = _state.embeddings[index].tolist()
        
        return {
            "idx": int(row["idx"]),
            "filename": str(row["filename"]),
            "scores": {
                "technical": float(row.get("technical_score", 0)),
                "aesthetic": float(row.get("aesthetic_score", 0)),
                "object": float(row.get("object_aesthetic_score", 0)),
            },
            "aggregated_score": float(row.get("aggregated_score", 0)),
            "cluster_id": int(row.get("cluster_id", -1)),
            "tsne": {
                "x": float(row.get("tsne_x", 0)),
                "y": float(row.get("tsne_y", 0)),
            },
            "is_champion": bool(row.get("is_final_selection", 0) == 1),
            "rejection_reason": row.get("rejection_reason"),
            "embedding": embedding,
        }


@app.get("/images/{filename}")
async def get_image_file(filename: str):
    """Serve image file from cloud service run directory."""
    from fastapi.responses import FileResponse
    
    with _state_lock:
        if not _state.run_id:
            raise HTTPException(status_code=404, detail="No active run")
        
        # Construct absolute path to image file
        # Dashboard is at /demo/dashboard, data is at /demo/data
        img_path = Path(__file__).parent.parent / "data" / "runs" / _state.run_id / "images" / filename
        
        if not img_path.exists():
            logger.warning(f"Image not found: {img_path}")
            raise HTTPException(status_code=404, detail=f"Image {filename} not found")
        
        return FileResponse(img_path, media_type="image/jpeg")


@app.get("/runs")
async def list_available_runs() -> dict:
    """List all available scored datasets."""
    runs_dir = Path(__file__).parent.parent / "data" / "runs"
    
    if not runs_dir.exists():
        return {"runs": []}
    
    available_runs = []
    for run_dir in sorted(runs_dir.iterdir()):
        if run_dir.is_dir():
            # Check if this run has results
            images_dir = run_dir / "images"
            results_file = run_dir / "results.json"
            
            if images_dir.exists() and results_file.exists():
                try:
                    with open(results_file) as f:
                        results_data = json.load(f)
                    image_count = len(list(images_dir.glob("*.jpg")))
                    available_runs.append({
                        "run_id": run_dir.name,
                        "image_count": image_count,
                        "champion_count": results_data.get("champion_count", 0),
                    })
                except Exception as e:
                    logger.warning(f"Error reading run {run_dir.name}: {e}")
    
    return {"runs": available_runs}


@app.post("/load-run/{run_id}")
async def load_run(run_id: str) -> dict:
    """Load a previously scored run without re-running inference."""
    runs_dir = Path(__file__).parent.parent / "data" / "runs"
    run_dir = runs_dir / run_id
    
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    try:
        results_file = run_dir / "results.json"
        
        if not results_file.exists():
            raise HTTPException(status_code=404, detail=f"No results for run {run_id}")
        
        with open(results_file) as f:
            results = json.load(f)
        
        # Reconstruct the dataframe from the results
        results_list = results.get("results", [])
        embeddings_list = []
        df_data = []
        
        for img_result in results_list:
            df_data.append({
                "idx": img_result["index"],
                "filename": img_result["filename"],
                "technical_score": img_result["scores"]["technical"],
                "aesthetic_score": img_result["scores"]["aesthetic"],
                "object_aesthetic_score": img_result["scores"]["object"],
                "tech_norm": img_result["normalized_scores"]["tech_norm"],
                "aes_norm": img_result["normalized_scores"]["aes_norm"],
                "obj_norm": img_result["normalized_scores"]["obj_norm"],
                "aggregated_score": img_result["aggregated_score"],
                "cluster_id": img_result["cluster_id"],
                "is_final_selection": img_result["is_champion"],
                "tech_penalized": img_result["tech_penalized"],
                "rejection_reason": img_result["rejection_reason"],
            })
            embeddings_list.append(np.array(img_result["embedding"]))
        
        df = pd.DataFrame(df_data)
        embeddings = np.array(embeddings_list) if embeddings_list else np.array([])
        
        # Update state
        epsilon = results.get("epsilon", 0.12)
        min_cluster_size = results.get("min_cluster_size", 2)
        ignore_object = results.get("ignore_object", False)
        
        _update_state(
            status="done",
            stage="done",
            stage_index=6,
            images_done=len(results_list),
            images_total=len(results_list),
            run_id=run_id,
            scores_df=df,
            embeddings=embeddings,
            results=results,
            epsilon=epsilon,
            min_cluster_size=min_cluster_size,
            ignore_object=ignore_object,
            error=None,
        )
        
        # Return image list
        images = _df_to_image_list(df)
        champions = [img for img in images if img["is_champion"]]
        
        return {
            "run_id": run_id,
            "total_images": len(images),
            "total_champions": len(champions),
            "images": images,
            "champions": champions,
            "parameters": {
                "epsilon": epsilon,
                "min_cluster_size": min_cluster_size,
                "ignore_object": ignore_object,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading run {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load run: {str(e)}")


@app.post("/compute-tsne")
async def compute_tsne_endpoint() -> dict:
    """Compute t-SNE coordinates for current embeddings and clustering."""
    with _state_lock:
        if _state.embeddings is None or _state.scores_df is None:
            raise HTTPException(status_code=404, detail="No embeddings available")
        
        embeddings = _state.embeddings
        df = _state.scores_df
    
    try:
        # Compute t-SNE from embeddings
        from sklearn.manifold import TSNE
        logger.info(f"Computing t-SNE for {len(embeddings)} embeddings...")
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
        coords = tsne.fit_transform(embeddings)
        
        # Build result with image indices, cluster IDs, and t-SNE coordinates
        result_coords = []
        for i, row in df.iterrows():
            result_coords.append({
                "index": int(row.get("idx", i)),
                "filename": str(row.get("filename", "")),
                "x": float(coords[i][0]),
                "y": float(coords[i][1]),
                "cluster_id": int(row.get("cluster_id", -1)),
                "is_champion": bool(row.get("is_final_selection", 0) == 1),
                "aggregated_score": float(row.get("aggregated_score", 0)),
            })
        
        logger.info(f"t-SNE computed successfully: {len(result_coords)} points")
        return {
            "tsne_coordinates": result_coords,
        }
    except Exception as e:
        logger.error(f"Failed to compute t-SNE: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to compute t-SNE: {str(e)}")

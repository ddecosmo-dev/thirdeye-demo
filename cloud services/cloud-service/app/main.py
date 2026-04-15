from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .cycles import notify_edge
from .ingest import ingest_zip
from .models import CycleActionResponse, IngestResponse, StartCycleRequest, StartCycleResponse
from .storage import create_metadata, read_metadata, run_dir, runs_root, write_metadata
from .utils import create_run_id, now_iso, safe_label, sha256_file


app = FastAPI(title="ThirdEye Cloud Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/cycle/start", response_model=StartCycleResponse)
async def start_cycle(payload: StartCycleRequest) -> StartCycleResponse:
    label = safe_label(payload.label)
    run_id = create_run_id(label)

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

    await notify_edge("start", payload.model_dump())

    return StartCycleResponse(run_id=run_id, label=label, status="running")


@app.post("/cycle/stop", response_model=CycleActionResponse)
async def stop_cycle(run_id: str) -> CycleActionResponse:
    meta = read_metadata(run_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Run not found")
    meta["status"] = "stopped"
    meta["ended_at"] = now_iso()
    write_metadata(run_id, meta)

    await notify_edge("stop", {"run_id": run_id})

    return CycleActionResponse(run_id=run_id, status="stopped")


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

    await notify_edge("abort", {"run_id": run_id, "reason": reason})

    return CycleActionResponse(run_id=run_id, status="aborted")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(
    run_id: str | None = Form(default=None),
    metadata_json: str | None = Form(default=None),
    file: UploadFile = File(...),
) -> IngestResponse:
    if run_id is None:
        run_id = create_run_id("ingest")
        create_metadata(run_id, label="ingest", status="uploaded")
    else:
        run_dir(run_id)

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

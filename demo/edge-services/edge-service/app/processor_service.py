"""Image processor service for storing images and metadata."""

from __future__ import annotations

import json
import os
import threading
import zipfile
from dataclasses import dataclass
from typing import Any

from flask import Flask, jsonify, request

from .config import settings
from .storage import (
    append_image_metadata,
    create_run_metadata,
    run_dir,
    temp_dir,
    temp_images_dir,
    update_run_metadata,
)
from .utils import configure_logging, ensure_dir, now_iso


@dataclass
class ProcessorState:
    active_run_id: str | None = None
    status: str = "idle"
    image_count: int = 0
    bytes_written: int = 0


class ImageProcessor:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = ProcessorState()
        self._logger = configure_logging("image-processor")

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self._state.status,
                "run_id": self._state.active_run_id,
                "image_count": self._state.image_count,
                "bytes_written": self._state.bytes_written,
            }

    def start_run(self, run_id: str, label: str | None) -> dict[str, Any]:
        with self._lock:
            if self._state.status == "running":
                raise ValueError("Processor already running")
            ensure_dir(settings.data_dir)
            temp_dir(run_id)
            temp_images_dir(run_id)
            create_run_metadata(
                run_id,
                label=label or "run",
                status="running",
                extra={"started_at": now_iso()},
            )
            self._state = ProcessorState(active_run_id=run_id, status="running")
            self._logger.info("processor started run_id=%s", run_id)
        return {"run_id": run_id, "status": "running"}

    def store_image(self, run_id: str, filename: str, payload: dict[str, Any], image_bytes: bytes) -> None:
        with self._lock:
            if self._state.status != "running" or run_id != self._state.active_run_id:
                raise ValueError("No active run")
            output_path = os.path.join(temp_images_dir(run_id), filename)
            with open(output_path, "wb") as handle:
                handle.write(image_bytes)
            self._state.image_count += 1
            self._state.bytes_written += len(image_bytes)
            append_image_metadata(run_id, payload)
            self._logger.info("stored image run_id=%s filename=%s bytes=%s", run_id, filename, len(image_bytes))

    def finalize_run(self, run_id: str, aborted: bool) -> dict[str, Any]:
        with self._lock:
            if run_id != self._state.active_run_id:
                raise ValueError("No active run")
            self._state.status = "finalizing"

        archive_path = self._create_archive(run_id)
        update_run_metadata(
            run_id,
            {
                "status": "aborted" if aborted else "stopped",
                "ended_at": now_iso(),
                "archive_path": archive_path,
                "image_count": self._state.image_count,
                "bytes_written": self._state.bytes_written,
            },
        )

        self._cleanup_temp(run_id)

        with self._lock:
            self._state = ProcessorState()

        self._logger.info("processor finalized run_id=%s", run_id)
        return {"run_id": run_id, "status": "archived", "archive_path": archive_path}

    def _create_archive(self, run_id: str) -> str:
        output_path = os.path.join(run_dir(run_id), "bundle.zip")
        temp_root = temp_dir(run_id)
        ensure_dir(os.path.dirname(output_path))

        with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(temp_root):
                for name in files:
                    full_path = os.path.join(root, name)
                    rel_path = os.path.relpath(full_path, temp_root)
                    zf.write(full_path, rel_path)

        return output_path

    def _cleanup_temp(self, run_id: str) -> None:
        root = temp_dir(run_id)
        for path, _, files in os.walk(root, topdown=False):
            for name in files:
                os.remove(os.path.join(path, name))
            if path != root:
                os.rmdir(path)
        if os.path.exists(root):
            os.rmdir(root)


processor = ImageProcessor()
app = Flask(__name__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/status")
def status():
    return jsonify(processor.status()), 200


@app.post("/run/start")
def run_start():
    payload = request.get_json(silent=True) or {}
    run_id = payload.get("run_id")
    label = payload.get("label")
    if not run_id:
        return jsonify({"error": "run_id is required"}), 400
    try:
        return jsonify(processor.start_run(run_id, label)), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/run/stop")
def run_stop():
    payload = request.get_json(silent=True) or {}
    run_id = payload.get("run_id")
    if not run_id:
        return jsonify({"error": "run_id is required"}), 400
    try:
        return jsonify(processor.finalize_run(run_id, aborted=False)), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/run/abort")
def run_abort():
    payload = request.get_json(silent=True) or {}
    run_id = payload.get("run_id")
    if not run_id:
        return jsonify({"error": "run_id is required"}), 400
    try:
        return jsonify(processor.finalize_run(run_id, aborted=True)), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/image")
def image_ingest():
    run_id = request.form.get("run_id")
    metadata_json = request.form.get("metadata_json")
    image_file = request.files.get("image")

    if not run_id or not metadata_json or not image_file:
        return jsonify({"error": "run_id, metadata_json, image are required"}), 400

    try:
        metadata = json.loads(metadata_json)
    except json.JSONDecodeError:
        processor._logger.error("invalid metadata_json run_id=%s", run_id)
        return jsonify({"error": "invalid metadata_json"}), 400

    filename = metadata.get("filename") or image_file.filename or "image.jpg"

    try:
        processor.store_image(run_id, filename, metadata, image_file.read())
        return jsonify({"status": "stored", "filename": filename}), 200
    except ValueError as exc:
        processor._logger.error("store image failed run_id=%s error=%s", run_id, exc)
        return jsonify({"error": str(exc)}), 400

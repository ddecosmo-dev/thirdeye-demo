"""Coordinator service for cycle control and pipeline execution."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import httpx
from flask import Flask, jsonify, request

from .config import settings
from .oak_controller import build_frame_source
from .pipeline import Pipeline
from .utils import configure_logging, create_run_id, now_iso, safe_label


@dataclass
class CoordinatorState:
    status: str = "idle"
    run_id: str | None = None
    label: str | None = None
    started_at: str | None = None
    frames_sent: int = 0
    frames_failed: int = 0


class Coordinator:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = CoordinatorState()
        self._stop_event = threading.Event()
        self._abort_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._logger = configure_logging("coordinator")
        self._pipeline = Pipeline()
        self._frame_source = None
        self._events: deque[str] = deque(maxlen=200)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._state.__dict__.copy()

    def events(self) -> list[str]:
        with self._lock:
            return list(self._events)

    def start_cycle(self, label: str | None, duration_seconds: int | None) -> dict[str, Any]:
        with self._lock:
            if self._state.status == "running":
                raise ValueError("Cycle already running")
            if not settings.oak_connected:
                raise ValueError("OAK_CONNECTED must be true for hardware inference")
            if duration_seconds is not None:
                duration_seconds = int(duration_seconds)
                if duration_seconds <= 0:
                    raise ValueError("duration_seconds must be positive")
            self._state = CoordinatorState(
                status="running",
                run_id=create_run_id(label),
                label=safe_label(label),
                started_at=now_iso(),
            )
            self._stop_event.clear()
            self._abort_event.clear()
            self._events.clear()

        self._frame_source = build_frame_source()
        self._logger.info(
            "cycle start run_id=%s label=%s duration_seconds=%s",
            self._state.run_id,
            self._state.label,
            duration_seconds,
        )
        self._logger.info("oak connected, starting DepthAI pipeline on device")

        self._notify_processor_start()

        self._worker = threading.Thread(
            target=self._run_capture,
            args=(duration_seconds,),
            daemon=True,
        )
        self._worker.start()

        return {"run_id": self._state.run_id, "status": "running"}

    def stop_cycle(self) -> dict[str, Any]:
        self._stop_event.set()
        self._log_event("cycle_stop_requested")
        self._logger.info("cycle stop requested run_id=%s", self._state.run_id)
        self._notify_processor_stop(aborted=False)
        return {"run_id": self._state.run_id, "status": "stopping"}

    def abort_cycle(self, reason: str | None) -> dict[str, Any]:
        self._abort_event.set()
        self._stop_event.set()
        self._logger.warning("abort requested reason=%s", reason or "none")
        self._log_event(f"cycle_abort_requested reason={reason or 'none'}")
        self._notify_processor_stop(aborted=True)
        return {"run_id": self._state.run_id, "status": "aborting"}

    def _notify_processor_start(self) -> None:
        if not self._state.run_id:
            return
        payload = {"run_id": self._state.run_id, "label": self._state.label}
        self._logger.info("notifying processor start run_id=%s", self._state.run_id)
        self._post_json("/run/start", payload)

    def _notify_processor_stop(self, aborted: bool) -> None:
        if not self._state.run_id:
            return
        endpoint = "/run/abort" if aborted else "/run/stop"
        payload = {"run_id": self._state.run_id}
        self._logger.info("notifying processor %s run_id=%s", endpoint, self._state.run_id)
        self._post_json(endpoint, payload)

    def _post_json(self, path: str, payload: dict[str, Any]) -> None:
        url = f"{settings.processor_base_url}{path}"
        try:
            with httpx.Client(timeout=10) as client:
                response = client.post(url, json=payload)
            if response.status_code >= 400:
                self._logger.error("processor call failed path=%s status=%s", path, response.status_code)
                self._log_event(f"processor_call_failed path={path} status={response.status_code}")
        except httpx.HTTPError as exc:
            self._logger.error("processor call failed path=%s error=%s", path, exc)
            self._log_event(f"processor_call_failed path={path} error={exc}")

    def _run_capture(self, duration_seconds: int | None) -> None:
        start_time = time.time()
        image_index = 0
        self._logger.info("capture loop start")
        self._frame_source.start()

        try:
            while not self._stop_event.is_set():
                if duration_seconds is not None:
                    if time.time() - start_time >= duration_seconds:
                        self._logger.info("duration elapsed, stopping cycle")
                        self._log_event("duration_elapsed")
                        self._stop_event.set()
                        break

                frame = self._frame_source.next_frame()
                if frame is None:
                    time.sleep(0.1)
                    continue

                image_index += 1
                filename = frame.filename_hint or f"image_{image_index:06d}.jpg"
                try:
                    processed = self._pipeline.process(frame, filename, image_index)
                    self._send_to_processor(processed)
                    with self._lock:
                        self._state.frames_sent += 1
                except Exception as exc:
                    self._logger.error("pipeline error image_index=%s error=%s", image_index, exc)
                    self._log_event(f"pipeline_error image_index={image_index} error={exc}")
                    with self._lock:
                        self._state.frames_failed += 1

        finally:
            self._frame_source.stop()
            with self._lock:
                self._state.status = "idle"
            self._log_event("cycle_complete")
            self._logger.info("capture loop ended frames_sent=%s frames_failed=%s", self._state.frames_sent, self._state.frames_failed)

    def _send_to_processor(self, processed: Any) -> None:
        if not self._state.run_id:
            return
        url = f"{settings.processor_base_url}/image"
        metadata = dict(processed.metadata)
        metadata["run_id"] = self._state.run_id
        metadata["label"] = self._state.label

        files = {"image": (processed.filename, processed.image_bytes, "image/jpeg")}
        data = {"run_id": self._state.run_id, "metadata_json": json_dumps(metadata)}

        try:
            with httpx.Client(timeout=20) as client:
                response = client.post(url, data=data, files=files)
            if response.status_code >= 400:
                self._logger.error("image ingest failed status=%s", response.status_code)
                self._log_event(f"image_ingest_failed status={response.status_code}")
        except httpx.HTTPError as exc:
            self._logger.error("image ingest failed error=%s", exc)
            self._log_event(f"image_ingest_failed error={exc}")

    def _log_event(self, message: str) -> None:
        with self._lock:
            self._events.append(f"{now_iso()} {message}")


def json_dumps(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, sort_keys=True)


coordinator = Coordinator()
app = Flask(__name__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/cycle/start")
def cycle_start():
    payload = request.get_json(silent=True) or {}
    label = payload.get("label")
    duration_seconds = payload.get("duration_seconds")

    try:
        return jsonify(coordinator.start_cycle(label, duration_seconds)), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/cycle/stop")
def cycle_stop():
    return jsonify(coordinator.stop_cycle()), 200


@app.post("/cycle/abort")
def cycle_abort():
    payload = request.get_json(silent=True) or {}
    reason = payload.get("reason")
    return jsonify(coordinator.abort_cycle(reason)), 200


@app.get("/status")
def status():
    return jsonify(coordinator.status()), 200


@app.get("/events")
def events():
    return jsonify({"events": coordinator.events()}), 200

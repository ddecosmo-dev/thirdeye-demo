"""Cycle state machine and capture/upload coordination for the edge device."""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

from .config import settings
from .oak_controller import Frame, build_frame_source
from .storage import append_event, create_metadata, images_dir, update_metadata
from .uploader import create_archive, upload_archive
from .utils import available_disk_bytes, create_run_id, ensure_dir, now_iso, safe_label
from .validation import validate_image_bytes


@dataclass
class CycleStats:
    frames_received: int = 0
    frames_written: int = 0
    frames_dropped: int = 0
    bytes_written: int = 0
    limit_reason: str | None = None


class CycleManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = "idle"
        self._run_id: str | None = None
        self._label: str | None = None
        self._stats = CycleStats()
        self._queue: queue.Queue[Frame | None] = queue.Queue(maxsize=settings.max_queue_frames)
        self._stop_event = threading.Event()
        self._abort_event = threading.Event()
        self._capture_thread: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None
        self._finalize_thread: threading.Thread | None = None
        self._frame_source = build_frame_source()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": self._state,
                "run_id": self._run_id,
                "label": self._label,
                "stats": self._stats.__dict__.copy(),
            }

    def start_cycle(self, label: str | None, duration_seconds: int | None, config: dict[str, Any] | None) -> dict[str, Any]:
        with self._lock:
            if self._state not in ("idle", "upload_failed"):
                raise ValueError("Cycle already running")

            ensure_dir(settings.data_dir)
            free_bytes = available_disk_bytes(settings.data_dir)
            if free_bytes < settings.min_free_disk_bytes:
                raise ValueError("Insufficient disk space")

            self._label = safe_label(label)
            self._run_id = create_run_id(self._label)
            self._stats = CycleStats()
            self._stop_event.clear()
            self._abort_event.clear()

            create_metadata(
                self._run_id,
                label=self._label,
                status="running",
                extra={
                    "requested_duration_seconds": duration_seconds,
                    "config": config or {},
                    "started_at": now_iso(),
                },
            )
            append_event(self._run_id, "cycle_started")

            self._state = "running"

            self._writer_thread = threading.Thread(target=self._writer_worker, daemon=True)
            self._capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
            self._writer_thread.start()
            self._capture_thread.start()

            if duration_seconds:
                threading.Thread(
                    target=self._auto_stop_after,
                    args=(duration_seconds,),
                    daemon=True,
                ).start()

            return {
                "run_id": self._run_id,
                "label": self._label,
                "state": self._state,
            }

    def stop_cycle(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            if self._state != "running" or run_id != self._run_id:
                raise ValueError("No active cycle")
            self._stop_event.set()
            append_event(run_id, "cycle_stop_requested")

            if self._finalize_thread is None or not self._finalize_thread.is_alive():
                self._finalize_thread = threading.Thread(target=self._finalize_cycle, args=(False,), daemon=True)
                self._finalize_thread.start()

            return {"run_id": run_id, "state": "stopping"}

    def abort_cycle(self, run_id: str, reason: str | None) -> dict[str, Any]:
        with self._lock:
            if self._state != "running" or run_id != self._run_id:
                raise ValueError("No active cycle")
            self._abort_event.set()
            self._stop_event.set()
            append_event(run_id, f"cycle_abort_requested reason={reason or 'none'}")

            if self._finalize_thread is None or not self._finalize_thread.is_alive():
                self._finalize_thread = threading.Thread(target=self._finalize_cycle, args=(True,), daemon=True)
                self._finalize_thread.start()

            return {"run_id": run_id, "state": "aborting"}

    def _auto_stop_after(self, duration_seconds: int) -> None:
        time.sleep(duration_seconds)
        with self._lock:
            if self._state == "running" and self._run_id:
                self._stop_event.set()
                append_event(self._run_id, "cycle_auto_stop")
                if self._finalize_thread is None or not self._finalize_thread.is_alive():
                    self._finalize_thread = threading.Thread(target=self._finalize_cycle, args=(False,), daemon=True)
                    self._finalize_thread.start()

    def _capture_worker(self) -> None:
        try:
            self._frame_source.start()
            while not self._stop_event.is_set() and not self._abort_event.is_set():
                frame = self._frame_source.next_frame()
                if frame is None:
                    continue
                self._stats.frames_received += 1
                try:
                    info = validate_image_bytes(frame.data, settings.max_image_bytes)
                except ValueError:
                    self._stats.frames_dropped += 1
                    continue

                try:
                    self._queue.put_nowait(Frame(data=frame.data, filename_hint=frame.filename_hint, ext=info.ext))
                except queue.Full:
                    self._stats.frames_dropped += 1
                    continue
        except Exception as exc:
            with self._lock:
                if self._run_id:
                    append_event(self._run_id, f"capture_error {exc}")
                self._state = "error"
        finally:
            self._frame_source.stop()

    def _writer_worker(self) -> None:
        while True:
            try:
                frame = self._queue.get(timeout=0.5)
            except queue.Empty:
                if self._stop_event.is_set() or self._abort_event.is_set():
                    break
                continue

            if frame is None:
                break

            if self._run_id is None:
                continue

            index = self._stats.frames_written + 1
            ext = frame.ext or os.path.splitext(frame.filename_hint)[1] or ".jpg"
            filename = f"frame_{index:06d}{ext}"
            output_path = os.path.join(images_dir(self._run_id), filename)

            with open(output_path, "wb") as handle:
                handle.write(frame.data)

            self._stats.frames_written += 1
            self._stats.bytes_written += len(frame.data)

            if self._stats.frames_written >= settings.max_run_images:
                self._stop_event.set()
                if self._stats.limit_reason is None:
                    self._stats.limit_reason = "max_run_images"
                    append_event(self._run_id, "limit_reached max_run_images")
            if self._stats.bytes_written >= settings.max_run_bytes:
                self._stop_event.set()
                if self._stats.limit_reason is None:
                    self._stats.limit_reason = "max_run_bytes"
                    append_event(self._run_id, "limit_reached max_run_bytes")

            if self._stats.frames_written % 10 == 0:
                update_metadata(
                    self._run_id,
                    {
                        "frames_written": self._stats.frames_written,
                        "bytes_written": self._stats.bytes_written,
                        "frames_dropped": self._stats.frames_dropped,
                        "limit_reason": self._stats.limit_reason,
                    },
                )

        if self._run_id:
            update_metadata(
                self._run_id,
                {
                    "frames_written": self._stats.frames_written,
                    "bytes_written": self._stats.bytes_written,
                    "frames_dropped": self._stats.frames_dropped,
                    "limit_reason": self._stats.limit_reason,
                },
            )

    def _finalize_cycle(self, aborted: bool) -> None:
        run_id = self._run_id
        if not run_id:
            return

        if self._capture_thread:
            self._capture_thread.join(timeout=5)

        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._writer_thread:
            self._writer_thread.join(timeout=5)

        status = "aborted" if aborted else "stopped"
        update_metadata(run_id, {"status": status, "ended_at": now_iso()})

        with self._lock:
            if aborted:
                self._state = "aborted"
                append_event(run_id, "cycle_aborted")
                return
            self._state = "uploading"

        try:
            archive_path, file_count, total_bytes = create_archive(run_id, settings.upload_max_bytes)
            response = upload_archive(
                run_id,
                archive_path,
                {
                    "file_count": file_count,
                    "total_bytes": total_bytes,
                    "frames_dropped": self._stats.frames_dropped,
                },
            )
            update_metadata(
                run_id,
                {
                    "status": "uploaded",
                    "upload_checksum": response["checksum_sha256"],
                },
            )
            append_event(run_id, "cycle_uploaded")
            with self._lock:
                self._state = "idle"
        except Exception as exc:
            update_metadata(run_id, {"status": "upload_failed", "upload_error": str(exc)})
            append_event(run_id, f"upload_failed {exc}")
            with self._lock:
                self._state = "upload_failed"

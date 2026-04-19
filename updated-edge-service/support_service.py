from __future__ import annotations

import datetime
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

app = Flask(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_SCRIPT = SCRIPT_DIR / "headless_iqa_v3.py"
RUNTIME_DATA_ROOT = SCRIPT_DIR / "runtime_data"
OUTBOUND_ROOT = SCRIPT_DIR / "outbound"
INSPECTION_ROOT = SCRIPT_DIR / "inspection"
RUNTIME_DATA_ROOT.mkdir(parents=True, exist_ok=True)
OUTBOUND_ROOT.mkdir(parents=True, exist_ok=True)
INSPECTION_ROOT.mkdir(parents=True, exist_ok=True)

_process_lock = threading.Lock()
_process: subprocess.Popen | None = None
_process_args: dict[str, Any] | None = None
_current_run_dir: Path | None = None
_intentional_stop = False
_stop_mode: str | None = None

_health: dict[str, Any] = {
    "running": False,
    "healthy": False,
    "state": "idle",
    "stop_reason": None,
    "last_exit_code": None,
    "last_error": None,
    "process_start_time": None,
    "last_update": None,
    "zip_path": None,
    "inspection_path": None,
}

HEALTH_CHECK_INTERVAL = 2.0


def _now_iso() -> str:
    return datetime.datetime.now().isoformat()


def _update_health(**kwargs: Any) -> None:
    _health.update({"last_update": _now_iso(), **kwargs})


def _get_status_snapshot() -> dict[str, Any]:
    with _process_lock:
        process_info: dict[str, Any] = {
            "running": False,
            "pid": None,
            "args": None,
        }

        if _process is not None:
            process_info["running"] = _process.poll() is None
            process_info["pid"] = _process.pid
            process_info["args"] = _process_args

        return {**process_info, "health": dict(_health)}


def _zip_run_dir(run_dir: Path) -> Path:
    target_zip = OUTBOUND_ROOT / f"{run_dir.name}.zip"
    with zipfile.ZipFile(target_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run_dir.rglob("*")):
            archive.write(path, arcname=path.relative_to(run_dir.parent))
    return target_zip


def _cleanup_run_dir(run_dir: Path) -> None:
    if run_dir.exists():
        shutil.rmtree(run_dir)


def _move_run_dir_to_inspection(run_dir: Path) -> Path:
    target_dir = INSPECTION_ROOT / run_dir.name
    if target_dir.exists():
        shutil.rmtree(target_dir)
    return Path(shutil.move(str(run_dir), str(target_dir)))


def _monitor_process() -> None:
    global _process, _process_args, _current_run_dir, _intentional_stop, _stop_mode
    while True:
        with _process_lock:
            if _process is not None:
                exit_code = _process.poll()
                if exit_code is not None:
                    if _stop_mode == "abort":
                        reason = "abort"
                        healthy = False
                        state = "aborted"
                    elif _stop_mode == "stop":
                        reason = "intentional"
                        healthy = True
                        state = "stopped"
                    elif _intentional_stop:
                        reason = "intentional"
                        healthy = exit_code == 0
                        state = "stopped" if exit_code == 0 else "crashed"
                    elif exit_code == 0:
                        reason = "completed"
                        healthy = True
                        state = "completed"
                    else:
                        reason = "crashed"
                        healthy = False
                        state = "crashed"

                    zip_path = None
                    inspection_path = None
                    if healthy and state in {"completed", "stopped"} and _current_run_dir is not None:
                        try:
                            zip_path = str(_zip_run_dir(_current_run_dir))
                            _cleanup_run_dir(_current_run_dir)
                        except Exception as exc:
                            inspection_path = str(_move_run_dir_to_inspection(_current_run_dir))
                            _update_health(last_error=str(exc))
                    elif _current_run_dir is not None:
                        inspection_path = str(_move_run_dir_to_inspection(_current_run_dir))

                    _update_health(
                        running=False,
                        healthy=healthy,
                        state=state,
                        stop_reason=reason,
                        last_exit_code=exit_code,
                        last_error=None if exit_code == 0 else f"Exit code {exit_code}",
                        zip_path=zip_path,
                        inspection_path=inspection_path,
                    )
                    _process = None
                    _process_args = None
                    _current_run_dir = None
                    _intentional_stop = False
                    _stop_mode = None
            else:
                if _health["state"] == "running":
                    _update_health(running=False, healthy=False, state="unknown")
        time.sleep(HEALTH_CHECK_INTERVAL)


@app.route("/status", methods=["GET"])
def status() -> Any:
    return jsonify(_get_status_snapshot())


@app.route("/health", methods=["GET"])
def health() -> Any:
    status_snapshot = _get_status_snapshot()
    return jsonify({
        "healthy": status_snapshot["health"]["healthy"],
        "state": status_snapshot["health"]["state"],
        "running": status_snapshot["running"],
        "stop_reason": status_snapshot["health"]["stop_reason"],
        "last_exit_code": status_snapshot["health"]["last_exit_code"],
        "last_error": status_snapshot["health"]["last_error"],
        "process_start_time": status_snapshot["health"]["process_start_time"],
        "last_update": status_snapshot["health"]["last_update"],
        "zip_path": status_snapshot["health"]["zip_path"],
        "inspection_path": status_snapshot["health"]["inspection_path"],
        "inspection_path": status_snapshot["health"]["inspection_path"],
        "state": status_snapshot["health"]["state"],
        "healthy": status_snapshot["health"]["healthy"],
        "stop_reason": status_snapshot["health"]["stop_reason"],
    })


@app.route("/start", methods=["POST"])
def start_pipeline() -> Any:
    data = request.get_json(silent=True) or {}
    with _process_lock:
        global _process, _process_args, _current_run_dir, _intentional_stop, _stop_mode
        if _process is not None and _process.poll() is None:
            return jsonify({"error": "Pipeline already running."}), 409

        run_seconds = int(data.get("run_seconds", 30))
        blur_thresh = float(data.get("blur_thresh", 100.0))
        min_intensity = float(data.get("min_intensity", 20.0))
        max_intensity = float(data.get("max_intensity", 235.0))
        run_id = data.get("run_id") or f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        log_dir = RUNTIME_DATA_ROOT
        _current_run_dir = log_dir / run_id
        _current_run_dir.mkdir(parents=True, exist_ok=False)

        args = [
            sys.executable,
            str(PIPELINE_SCRIPT),
            "--run-seconds",
            str(run_seconds),
            "--blur-thresh",
            str(blur_thresh),
            "--min-intensity",
            str(min_intensity),
            "--max-intensity",
            str(max_intensity),
            "--log-dir",
            str(log_dir),
            "--run-id",
            run_id,
        ]

        try:
            _process = subprocess.Popen(
                args,
                cwd=SCRIPT_DIR,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        except Exception as exc:
            _update_health(
                running=False,
                healthy=False,
                state="launch_failed",
                stop_reason="launch_failed",
                last_exit_code=None,
                last_error=str(exc),
                zip_path=None,
            )
            return jsonify({"error": "Failed to start pipeline.", "details": str(exc)}), 500

        _process_args = {
            "run_seconds": run_seconds,
            "blur_thresh": blur_thresh,
            "min_intensity": min_intensity,
            "max_intensity": max_intensity,
            "log_dir": str(log_dir),
            "run_id": run_id,
        }
        _intentional_stop = False
        _stop_mode = None
        _update_health(
            running=True,
            healthy=True,
            state="running",
            stop_reason=None,
            last_exit_code=None,
            last_error=None,
            process_start_time=_now_iso(),
            zip_path=None,
        )

    return jsonify({"status": "started", "pid": _process.pid, "run_dir": str(_current_run_dir)})


@app.route("/stop", methods=["POST"])
def stop_pipeline() -> Any:
    data = request.get_json(silent=True) or {}
    stop_mode = data.get("mode", "stop").lower()
    if stop_mode not in {"stop", "abort"}:
        return jsonify({"error": "Invalid stop mode. Use 'stop' or 'abort'."}), 400

    with _process_lock:
        global _process, _process_args, _intentional_stop, _stop_mode
        if _process is None or _process.poll() is not None:
            _process = None
            _process_args = None
            _intentional_stop = False
            _stop_mode = None
            _update_health(running=False, healthy=False, state="idle", stop_reason=None)
            return jsonify({"status": "not running"}), 400

        _intentional_stop = True
        _stop_mode = stop_mode
        _process.terminate()
        try:
            _process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _process.kill()
            _process.wait(timeout=5)
        pid = _process.pid

    return jsonify({"status": "terminated", "pid": pid, "mode": stop_mode})


if __name__ == "__main__":
    monitor = threading.Thread(target=_monitor_process, daemon=True, name="pipeline-health-monitor")
    monitor.start()
    app.run(host="0.0.0.0", port=5000)

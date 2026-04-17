"""Edge API entrypoint for cycle control and status."""

from __future__ import annotations

from flask import Flask, jsonify, request

from .cycle_manager import CycleManager


app = Flask(__name__)
manager = CycleManager()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# TODO: Add authentication (shared secret/JWT) before exposing publicly.
@app.post("/cycle/start")
def start_cycle():
    payload = request.get_json(silent=True) or {}
    label = payload.get("label")
    duration_seconds = payload.get("duration_seconds")
    config = payload.get("config")

    try:
        result = manager.start_cycle(label, duration_seconds, config)
        return jsonify(result), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/cycle/stop")
def stop_cycle():
    payload = request.get_json(silent=True) or {}
    run_id = payload.get("run_id")

    if not run_id:
        return jsonify({"error": "run_id is required"}), 400

    try:
        result = manager.stop_cycle(run_id)
        return jsonify(result), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.post("/cycle/abort")
def abort_cycle():
    payload = request.get_json(silent=True) or {}
    run_id = payload.get("run_id")
    reason = payload.get("reason")

    if not run_id:
        return jsonify({"error": "run_id is required"}), 400

    try:
        result = manager.abort_cycle(run_id, reason)
        return jsonify(result), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400


@app.get("/status")
def status():
    return jsonify(manager.status()), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8081)

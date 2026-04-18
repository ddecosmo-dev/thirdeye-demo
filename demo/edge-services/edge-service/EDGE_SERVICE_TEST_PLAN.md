# Edge Service Test Plan (Improved)

Goal: expose failures early, isolate them quickly, and provide explicit remediation steps with concrete commands to run on the edge device.

Top-level quick checks
- Verify OS, Docker (if used), and Python:

```bash
uname -a
python3 --version
docker --version || true
```

1) Setup & sanity
- Validate Python + DepthAI availability:

```bash
python3 -c "import depthai as dai; print('depthai OK', hasattr(dai, 'Device'))"
```
- Confirm required env vars (adjust paths as needed):

```bash
export OAK_CONNECTED=true
export BLOB_PATH=/path/to/student_mobilenet_v3.blob
export PREFILTER_BLOB_PATH=/path/to/prefilter.blob # optional
export CAPTURE_FPS=1
ls -lh "$BLOB_PATH"
```

- Start the service (local dev runner):

```bash
cd demo/edge-services/edge-service
python3 -m app.main
```

2) Build & run via Docker (if running containerized)
- Build image:

```bash
cd demo/edge-services/edge-service
docker build -t edge-service:local .
```
- Start stack (foreground):

```bash
docker-compose up
```

Or detached:

```bash
docker-compose up -d
```

3) API smoke & health checks
- Confirm container(s) running:

```bash
docker ps --filter name=edge-service --format "{{.Names}}: {{.Status}}"
```
- Health endpoint (adjust port/path from config):

```bash
curl -f http://localhost:8081/health || curl -f http://127.0.0.1:8081/health
```

4) Coordinator API tests (explicit commands)
- Start cycle (expect `status=running` + `run_id`):

```bash
curl -sS -X POST http://localhost:8081/cycle/start \
  -H 'Content-Type: application/json' \
  -d '{"label":"field","duration_seconds":10}' | jq
```
- Stop cycle:

```bash
curl -sS -X POST http://localhost:8081/cycle/stop
```
- Abort cycle (expect graceful response):

```bash
curl -sS -X POST http://localhost:8081/cycle/abort -H 'Content-Type: application/json' -d '{"reason":"test"}'
```

Remedies: inspect `docker-compose logs coordinator` or `journalctl` for stack traces.

5) Processor API tests (explicit commands)
- Start run and POST an image (use a small sample image):

```bash
curl -sS -X POST http://localhost:8082/run/start -H 'Content-Type: application/json' -d '{"run_id":"TEST","label":"manual"}'
curl -sS -X POST http://localhost:8082/image -F run_id=TEST -F image=@tests/sample_input.jpg
curl -sS -X POST http://localhost:8082/run/stop -H 'Content-Type: application/json' -d '{"run_id":"TEST"}'
```

Expect a `bundle.zip` under the run directory; if missing, check filesystem permissions and disk space (`df -h`).

6) ML inference validation
- Trigger inference via REST (adjust path/port):

```bash
curl -sS -X POST -F "file=@tests/sample_input.jpg" http://localhost:8081/infer | jq
```

- Local run (no Docker):

```bash
python3 app/inference/local_infer.py --model models/test_model.onnx --input tests/sample_input.npy
```

Verify response JSON includes `model_score` and `model_passed`.

7) Camera/hardware checks
- Confirm device on USB and DepthAI sees it:

```bash
lsusb
python3 -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"
```
- Quick camera script (if present):

```bash
python3 app/utils/test_camera.py --duration 5
```

8) Upload / connector tests
- Trigger uploader path and watch logs:

```bash
curl -sS -X POST http://localhost:8081/cycles -H 'Content-Type: application/json' -d '{"test":true}'
docker-compose logs uploader --tail=200
```

9) Logs & diagnostics
- Tail combined logs:

```bash
docker-compose logs -f
```

- Grab last 200 lines of app logs for bug reports:

```bash
docker-compose logs --no-color app | tail -n 200 > ~/edge_app_last.log
```

10) Disk pressure & data integrity
- Check disk free space and run file counts:

```bash
df -h /data || df -h /
wc -l /data/runs/<run_id>/temp/metadata.jsonl || true
ls /data/runs/<run_id>/temp/images | wc -l || true
unzip -l /data/runs/<run_id>/bundle.zip | head -n 20 || true
```

11) Fault injection quick cases
- Kill processor to ensure coordinator records failure:

```bash
pkill -f processor_service.py
docker-compose logs coordinator --tail=200
```

12) Cleanup & recovery
- Stop stack and remove images (if needed):

```bash
docker-compose down
docker image rm edge-service:local || true
```

Automated smoke script suggestion
- Create `demo/edge-services/edge-service/scripts/smoke_check.sh` to run: env checks, `docker-compose up -d`, health curl, inference curl, and exit non-zero on failures.

Notes
- Replace ports/endpoints and file paths with values in `app/config.py` or `docker-compose.yml`.
- If Docker is unavailable on target hardware, run Python-based tests and hardware checks directly.

Next steps I can take (choose any):
- Add `scripts/smoke_check.sh` to the repo
- Add `app/inference/local_infer.py` wrapper if missing
- Create a small `tests/sample_input.jpg` and `tests/sample_input.npy`

End of updated test plan.

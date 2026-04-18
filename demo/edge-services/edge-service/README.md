# ThirdEye Edge Service

Single-container edge services designed for the Pi Zero 2 W.

## What it provides
- Coordinator service for cycle control and pipeline execution
- Image processor service for storage and packaging
- Per-run directories with temp images and metadata.jsonl

## Data layout
Runs are stored under DATA_DIR (default: /data):

```
/data/runs/<run_id>/
  run.json
  bundle.zip
  temp/
    metadata.jsonl
    images/
```

## Environment variables
- DATA_DIR (default: /data)
- COORDINATOR_PORT (default: 8081)
- PROCESSOR_PORT (default: 8082)
- PROCESSOR_BASE_URL (default: http://127.0.0.1:8082)
- CAPTURE_FPS (default: 1)
- BLOB_PATH (optional): path to the DepthAI blob on device
- OAK_CONNECTED (default: false): set true to enable hardware logs
- NORMALIZE_INPUTS (default: false)

## Run locally

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.main
```

## Build and run container

```
docker build -t thirdeye-edge .
docker run --rm -p 8081:8081 -v $(pwd)/data:/data thirdeye-edge
```

## Example requests

Start a cycle (coordinator):
```
curl -X POST http://localhost:8081/cycle/start \
  -H 'Content-Type: application/json' \
  -d '{"label":"hike","duration_seconds":120}'
```

Stop a cycle:
```
curl -X POST http://localhost:8081/cycle/stop
```

Abort a cycle:
```
curl -X POST http://localhost:8081/cycle/abort \
  -H 'Content-Type: application/json' \
  -d '{"reason":"user_abort"}'
```

Coordinator status:
```
curl http://localhost:8081/status
```

Processor status:
```
curl http://localhost:8082/status
```

## TODO list
- Integrate Oak/DepthAI pipeline capture
- Add auth token or shared secret for edge endpoints
- Route bundles to the cloud ingest endpoint after edge testing

## Test cases (stress + reliability)
- Start/stop cycle with short duration (1-3 seconds) and verify archive creation
- Start cycle, abort immediately, ensure processor finalizes and temp directory is removed
- Start cycle with duration_seconds=0 or negative (expect 400)
- Start cycle twice without stopping (expect 400 on second)
- Send /cycle/stop when idle (expect graceful response)
- Processor /run/start twice with same run_id (expect 400)
- Processor /image with missing metadata_json (expect 400)
- Processor /image with invalid JSON (expect 400)
- Processor /image for unknown run_id (expect 400)
- Large image file over MAX_IMAGE_BYTES (expect validation failure)
- Mixed JPG/PNG inputs from the device camera
- 1 FPS and 5 FPS capture with 60+ seconds duration
- Disk full simulation (fill DATA_DIR) and start cycle
- Network failure between coordinator and processor (drop requests, expect events)
- Processor shutdown mid-run (coordinator logs ingest failure)
- OAK_CONNECTED=true with BLOB_PATH missing (check logs)
- OAK_CONNECTED=true with BLOB_PATH set (check logs)
- NORMALIZE_INPUTS=true with device images (ensure metadata normalization field)

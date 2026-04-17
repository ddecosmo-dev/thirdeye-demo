# ThirdEye Edge Service

Single-container edge service designed for the Pi Zero 2 W.

## What it provides
- Start/stop/abort cycle control endpoints
- Bounded capture -> disk write path with safety limits
- Zip + upload to cloud ingest endpoint
- Per-run directories with metadata.json and events.log

## Data layout
Runs are stored under DATA_DIR (default: /data):

```
/data/runs/<run_id>/
  metadata.json
  events.log
  images/
```

## Environment variables
- DATA_DIR (default: /data)
- CLOUD_INGEST_URL (default: http://localhost:8080/ingest)
- MOCK_IMAGE_DIR (optional): if set, reads images from this directory

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

Start a cycle:
```
curl -X POST http://localhost:8081/cycle/start \
  -H 'Content-Type: application/json' \
  -d '{"label":"hike","duration_seconds":120,"config":{"notes":"demo"}}'
```

Stop a cycle:
```
curl -X POST http://localhost:8081/cycle/stop \
  -H 'Content-Type: application/json' \
  -d '{"run_id":"RUN_ID"}'
```

Abort a cycle:
```
curl -X POST http://localhost:8081/cycle/abort \
  -H 'Content-Type: application/json' \
  -d '{"run_id":"RUN_ID","reason":"user_abort"}'
```

Status:
```
curl http://localhost:8081/status
```

## TODO list
- Integrate Oak/DepthAI pipeline capture
- Add auth token or shared secret for edge endpoints
- Add upload retry/backoff and receipt validation
- Enforce allowed image extensions in capture path
- Add cleanup policy after cloud confirms receipt

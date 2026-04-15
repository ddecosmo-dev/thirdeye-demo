# ThirdEye Cloud Service

Single-container cloud service for the ThirdEye demo.

## What it provides
- Start/stop/abort cycle control endpoints (for GUI to call)
- Image ingest endpoint that accepts zip uploads
- Run directories with metadata.json per cycle
- Lightweight REST API suitable for a demo GUI later

## Data layout
Runs are stored under DATA_DIR (default: /data):

```
/data/runs/<run_id>/
  metadata.json
  raw/upload.zip
  images/
```

## Run locally

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8080
```

## Build and run container

```
docker build -t thirdeye-cloud .
docker run --rm -p 8080:8080 -v $(pwd)/data:/data thirdeye-cloud
```

## Example requests

Start a cycle:
```
curl -X POST http://localhost:8080/cycle/start \
  -H 'Content-Type: application/json' \
  -d '{"label":"hike","duration_seconds":120,"config":{"notes":"demo"}}'
```

Stop a cycle:
```
curl -X POST "http://localhost:8080/cycle/stop?run_id=RUN_ID"
```

Abort a cycle:
```
curl -X POST "http://localhost:8080/cycle/abort?run_id=RUN_ID&reason=user_abort"
```

Upload images:
```
curl -X POST http://localhost:8080/ingest \
  -F "run_id=RUN_ID" \
  -F "metadata_json={\"notes\":\"oak upload\"}" \
  -F "file=@/path/to/images.zip"
```


## Next steps

Add image tag schema and a basic “inference done” update endpoint.
Wire the edge URL (EDGE_BASE_URL) and test the start/stop/abort callbacks.
Add auth token or shared secret to lock down /ingest and cycle endpoints.
Setup model pipeline 

TEST TEST

Setup GUI
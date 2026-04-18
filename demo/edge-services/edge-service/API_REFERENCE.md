# Edge Service API Reference

Complete API reference for the edge service coordinator and processor services running on localhost for local testing.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                  Camera (USB OAK-D Lite)                    │
│                    (via DepthAI pipeline)                   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         │ (frames @ CAPTURE_FPS)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               Coordinator (port 8081)                       │
│         Cycle control & ML inference orchestration          │
│  - Receives: start/stop/abort commands with duration        │
│  - Runs: capture loop, calls pipeline, sends to processor   │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      │ (HTTP POST /image)
                      │ (JPEG + metadata.json)
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                Processor (port 8082)                        │
│         Image storage & archive creation                    │
│  - Stores: JPEGs in temp/images/                            │
│  - Writes: metadata.jsonl (per-frame)                       │
│  - Creates: bundle.zip on stop/abort                        │
└─────────────────────────────────────────────────────────────┘
                      │
                      ▼
            /tmp/edge_data/runs/RUN_ID/
            ├── run.json (cycle metadata)
            ├── bundle.zip (final archive)
            └── temp/ (cleaned after archiving)
```

## Service Startup

### Environment Variables

```bash
# Required for hardware camera inference
export OAK_CONNECTED=true
export BLOB_PATH=/path/to/student_mobilenet_v3.blob

# Optional but recommended
export PREFILTER_BLOB_PATH=/path/to/prefilter.blob
export CAPTURE_FPS=1                  # Frames per second (default: 1)
export DOWNSAMPLE_WIDTH=320           # Inference input width
export DOWNSAMPLE_HEIGHT=240          # Inference input height
export MODEL_THRESHOLD=0.5            # Confidence threshold
export PREFILTER_THRESHOLD=0.25       # Prefilter confidence threshold

# Storage and networking
export DATA_DIR=/tmp/edge_data
export COORDINATOR_HOST=0.0.0.0
export COORDINATOR_PORT=8081
export PROCESSOR_HOST=0.0.0.0
export PROCESSOR_PORT=8082
```

### Start Services

```bash
# Start both coordinator and processor
python3 -m app.main

# Expected output:
# * Running on http://0.0.0.0:8082 (Processor)
# * Running on http://0.0.0.0:8081 (Coordinator)
```

## Coordinator API (port 8081)

The coordinator controls the capture cycle and runs ML inference on the camera pipeline.

### POST /cycle/start

**Start a new capture cycle**

Request:
```json
{
  "label": "test_run",           // Optional: friendly name for the run
  "duration_seconds": 30          // Optional: auto-stop after N seconds
}
```

Response (200 OK):
```json
{
  "run_id": "RUN_20240418_153022_test_run",
  "status": "running"
}
```

Error Cases:
- `400` - Cycle already running, OAK_CONNECTED not true, or invalid duration
- `500` - Device or pipeline error

Example:
```bash
curl -X POST http://localhost:8081/cycle/start \
  -H 'Content-Type: application/json' \
  -d '{
    "label": "field_test",
    "duration_seconds": 60
  }'
```

### POST /cycle/stop

**Stop the active cycle (gracefully finalize)**

Request:
```json
{}  // No payload required
```

Response (200 OK):
```json
{
  "run_id": "RUN_20240418_153022_test_run",
  "status": "stopping"
}
```

Notes:
- Triggers processor to finalize the archive
- Temp images/metadata are zipped into `bundle.zip`
- Temp directory is cleaned up after archiving

Example:
```bash
curl -X POST http://localhost:8081/cycle/stop
```

### POST /cycle/abort

**Abort the active cycle (emergency stop)**

Request:
```json
{
  "reason": "user_interrupt"  // Optional: abort reason for logging
}
```

Response (200 OK):
```json
{
  "run_id": "RUN_20240418_153022_test_run",
  "status": "aborting"
}
```

Notes:
- Same as `stop` but marks run as "aborted" instead of "stopped"
- Useful for testing error handling
- Archive is still created and cleaned up

Example:
```bash
curl -X POST http://localhost:8081/cycle/abort \
  -H 'Content-Type: application/json' \
  -d '{"reason":"manual_interrupt"}'
```

### GET /status

**Check current coordinator state**

Response (200 OK):
```json
{
  "status": "running",                // idle | running
  "run_id": "RUN_20240418_153022_test_run",
  "label": "field_test",
  "started_at": "2024-04-18T15:30:22Z",
  "frames_sent": 42,
  "frames_failed": 0
}
```

Example:
```bash
curl http://localhost:8081/status | jq
```

### GET /health

**Service health check**

Response (200 OK):
```json
{
  "status": "ok"
}
```

Example:
```bash
curl http://localhost:8081/health
```

### GET /events

**Get coordinator event log (up to 200 most recent)**

Response (200 OK):
```json
{
  "events": [
    "2024-04-18T15:30:22Z cycle_start",
    "2024-04-18T15:30:23Z image frame_000001.jpg processed",
    "2024-04-18T15:30:25Z duration_elapsed"
  ]
}
```

Example:
```bash
curl http://localhost:8081/events | jq
```

## Processor API (port 8082)

The processor receives images from the coordinator and manages storage + archiving.

### POST /run/start

**Initialize a new run (called by coordinator)**

Request:
```json
{
  "run_id": "RUN_20240418_153022_test_run",
  "label": "field_test"
}
```

Response (200 OK):
```json
{
  "run_id": "RUN_20240418_153022_test_run",
  "status": "running"
}
```

Notes:
- Creates temp directory structure: `runs/RUN_ID/temp/images/`
- Creates `run.json` metadata file
- Called automatically by coordinator on cycle start
- Can also be called directly for manual testing

### POST /run/stop

**Finalize and archive the run**

Request:
```json
{
  "run_id": "RUN_20240418_153022_test_run"
}
```

Response (200 OK):
```json
{
  "run_id": "RUN_20240418_153022_test_run",
  "status": "archived",
  "archive_path": "/tmp/edge_data/runs/RUN_20240418_153022_test_run/bundle.zip"
}
```

Effects:
- Creates `bundle.zip` with all images and metadata
- Cleans up temp directory
- Updates `run.json` with final status="stopped"

### POST /run/abort

**Abort and archive the run**

Request:
```json
{
  "run_id": "RUN_20240418_153022_test_run"
}
```

Response (200 OK):
```json
{
  "run_id": "RUN_20240418_153022_test_run",
  "status": "archived",
  "archive_path": "/tmp/edge_data/runs/RUN_20240418_153022_test_run/bundle.zip"
}
```

Effects:
- Same as `/run/stop` but sets status="aborted"
- Useful for testing error scenarios

### POST /image

**Ingest a single image and metadata (called by coordinator)**

Request (multipart/form-data):
```
run_id: RUN_20240418_153022_test_run
metadata_json: {"filename":"frame_000001.jpg","model_score":0.87,"model_passed":true,"tag":"model_passed"}
image: <binary JPEG data>
```

Response (200 OK):
```json
{
  "status": "stored",
  "filename": "frame_000001.jpg"
}
```

Notes:
- Called automatically by coordinator during capture
- Metadata is appended to `temp/metadata.jsonl`
- Image is written to `temp/images/frame_000001.jpg`

### GET /status

**Check processor state**

Response (200 OK):
```json
{
  "status": "running",          // idle | running
  "run_id": "RUN_20240418_153022_test_run",
  "image_count": 42,
  "bytes_written": 2048000
}
```

Example:
```bash
curl http://localhost:8082/status | jq
```

### GET /health

**Service health check**

Response (200 OK):
```json
{
  "status": "ok"
}
```

Example:
```bash
curl http://localhost:8082/health
```

## Complete Workflow Example

```bash
#!/bin/bash
set -e

# 1. Start services
echo "Starting services..."
export OAK_CONNECTED=true
export BLOB_PATH=$(pwd)/models/student_mobilenet_v3.blob
export DATA_DIR=/tmp/edge_data
python3 -m app.main &
SERVICE_PID=$!
sleep 2

# 2. Verify health
echo "Checking health..."
curl -s http://localhost:8081/health | jq
curl -s http://localhost:8082/health | jq

# 3. Start 30-second cycle
echo "Starting cycle..."
RESPONSE=$(curl -s -X POST http://localhost:8081/cycle/start \
  -H 'Content-Type: application/json' \
  -d '{"label":"demo","duration_seconds":30}')
RUN_ID=$(echo "$RESPONSE" | jq -r '.run_id')
echo "Run ID: $RUN_ID"

# 4. Monitor progress
echo "Monitoring..."
for i in {1..6}; do
  sleep 5
  curl -s http://localhost:8081/status | jq '.frames_sent'
done

# 5. Check results
echo "Checking archive..."
sleep 2
ls -lh /tmp/edge_data/runs/$RUN_ID/
unzip -l /tmp/edge_data/runs/$RUN_ID/bundle.zip | head -20

# 6. Cleanup
kill $SERVICE_PID
```

## Data Structure Reference

### Run Directory Layout

```
/tmp/edge_data/
└── runs/
    └── RUN_20240418_153022_test_run/
        ├── run.json                    # Cycle metadata (JSON)
        ├── bundle.zip                  # Final archive
        └── temp/                       # (cleaned after archiving)
            ├── metadata.jsonl          # Per-image metadata
            └── images/
                ├── frame_000001.jpg
                ├── frame_000002.jpg
                └── ...
```

### run.json Format

```json
{
  "archive_path": "/tmp/edge_data/runs/RUN_20240418_153022_test_run/bundle.zip",
  "bytes_written": 2048000,
  "created_at": "2024-04-18T15:30:22Z",
  "ended_at": "2024-04-18T15:30:52Z",
  "image_count": 30,
  "label": "test_run",
  "run_id": "RUN_20240418_153022_test_run",
  "started_at": "2024-04-18T15:30:22Z",
  "status": "stopped",
  "updated_at": "2024-04-18T15:30:52Z"
}
```

### metadata.jsonl Format (per line)

```json
{
  "filename": "frame_000001.jpg",
  "inference_source": "oak",
  "label": "test_run",
  "logged_at": "2024-04-18T15:30:23Z",
  "model_passed": true,
  "model_score": 0.87,
  "normalize_inputs": false,
  "prefilter_passed": true,
  "prefilter_blob_path": null,
  "prefilter_score": null,
  "run_id": "RUN_20240418_153022_test_run",
  "tag": "model_passed"
}
```

## Error Handling

### Common HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad request (invalid parameters, cycle already running, etc.) |
| 500 | Server error (device not found, pipeline error, disk full) |

### Retry Strategy

For production use (or extended local testing):
- Idempotent operations (start_cycle, stop_cycle) can be retried
- Network failures are logged but don't stop the capture loop
- Max image file size: 5MB (configurable via `MAX_IMAGE_BYTES`)
- Min free disk space: 200MB (configurable via `MIN_FREE_DISK_BYTES`)

## Performance Tips

### For Raspberry Pi Zero 2 W

- Set `CAPTURE_FPS=0.5` for one frame every 2 seconds
- Set `DOWNSAMPLE_WIDTH=240, DOWNSAMPLE_HEIGHT=180` for faster inference
- Keep `duration_seconds` under 300 (5 minutes) to avoid memory issues
- Monitor `/tmp/edge_data` disk usage during long runs

### For Local PC

- `CAPTURE_FPS=2-5` for faster testing
- Standard `DOWNSAMPLE_WIDTH=320, DOWNSAMPLE_HEIGHT=240`
- Can run for hours without issues

## Troubleshooting

### Camera Not Detected

```bash
python3 -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"
```

If empty, check USB connection and permissions:
```bash
# Linux only
lsusb | grep "03e7"  # Luxonis
sudo usermod -a -G plugdev $USER
```

### Images Not Being Captured

- Check `OAK_CONNECTED=true`
- Check `BLOB_PATH` exists and is readable
- Monitor `/tmp/edge_smoke_logs/edge_app.log` for errors
- Verify `CAPTURE_FPS` and `MODEL_THRESHOLD` aren't filtering everything

### Archive Not Created

- Check `/tmp/edge_data/runs/$RUN_ID/` exists
- Verify temp directory has images: `ls /tmp/edge_data/runs/$RUN_ID/temp/images/`
- Check disk space: `df -h /tmp`
- Review processor logs in stdout

### Metadata Missing

- Verify `metadata.jsonl` in archive: `unzip -p bundle.zip metadata.jsonl | wc -l`
- Each image should have one metadata line
- Check for coordination failures in event log: `curl http://localhost:8081/events | jq`

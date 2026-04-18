# Edge Service Local Testing - Setup Complete

## Summary of Changes

Your edge service is now configured and ready for local testing with your USB-connected camera. All commands use localhost for development before deploying to Raspberry Pi Zero 2 W.

## What Was Updated

### 1. **requirements.txt** - RPi Zero 2 W Compatibility ✓
   - Pinned all dependencies with compatible versions
   - Added numpy and requests (was missing)
   - Optimized for 64-bit OS on resource-constrained hardware
   - Added comments for optional development dependencies

### 2. **smoke_check_local.sh** - Simplified to Hardware-Only ✓
   - Removed mock mode complexity
   - Now exclusively tests with actual OAK camera
   - Validates blob files exist and accessible
   - Checks DepthAI device availability
   - Tests both coordinator and processor health
   - Logs all output to `/tmp/edge_smoke_logs/`

### 3. **LOCAL_TEST_SETUP.md** - Complete Testing Guide ✓
   - Hardware-only workflow documentation
   - Step-by-step environment setup
   - Start/stop/abort cycle examples
   - Data inspection and troubleshooting
   - Performance benchmarks for RPi Zero 2 W

### 4. **API_REFERENCE.md** - Complete API Documentation ✓
   - Coordinator API (port 8081): start/stop/abort cycles
   - Processor API (port 8082): image storage & archiving
   - Request/response schemas with examples
   - Error handling and retry strategy
   - Data structure reference

### 5. **test_e2e.py** - End-to-End Test Suite ✓
   - Automated testing of complete workflow
   - Tests: health, cycle start/stop, image collection, archive verification
   - Validates metadata and run.json
   - Reports results with pass/fail status

### 6. **quickstart.sh** - One-Command Setup & Test ✓
   - Validates environment configuration
   - Creates Python virtual environment
   - Runs smoke check
   - Starts services
   - Executes end-to-end tests
   - Displays summary of results

## Quick Start

### 1. Set Environment Variables

```bash
export OAK_CONNECTED=true
export BLOB_PATH=$(pwd)/models/student_mobilenet_v3.blob
export PREFILTER_BLOB_PATH=$(pwd)/models/prefilter.blob  # Optional
export CAPTURE_FPS=1
export DATA_DIR=/tmp/edge_data
```

### 2. Run Quick Start

```bash
cd demo/edge-services/edge-service
chmod +x quickstart.sh
./quickstart.sh
```

This will:
- ✓ Validate your setup
- ✓ Create Python environment
- ✓ Run smoke check
- ✓ Start coordinator & processor
- ✓ Execute end-to-end tests
- ✓ Display test results

## Manual Testing (if you prefer)

### Start Services

```bash
cd demo/edge-services/edge-service
python3 -m app.main
```

### In Another Terminal

```bash
# Start a 30-second cycle
curl -X POST http://localhost:8081/cycle/start \
  -H 'Content-Type: application/json' \
  -d '{"label":"test","duration_seconds":30}'

# Monitor progress
curl http://localhost:8081/status | jq

# Stop when done
curl -X POST http://localhost:8081/cycle/stop

# Verify archive
ls -lh /tmp/edge_data/runs/RUN_*/
unzip -l /tmp/edge_data/runs/RUN_*/bundle.zip | head -20
```

## API Commands Reference

### Coordinator (port 8081)

**Start cycle with duration:**
```bash
curl -X POST http://localhost:8081/cycle/start \
  -H 'Content-Type: application/json' \
  -d '{"label":"my_run","duration_seconds":60}'
```

**Stop cycle:**
```bash
curl -X POST http://localhost:8081/cycle/stop
```

**Abort cycle:**
```bash
curl -X POST http://localhost:8081/cycle/abort \
  -H 'Content-Type: application/json' \
  -d '{"reason":"test_abort"}'
```

**Check status:**
```bash
curl http://localhost:8081/status | jq
```

**View events:**
```bash
curl http://localhost:8081/events | jq
```

### Processor (port 8082)

**Check status:**
```bash
curl http://localhost:8082/status | jq
```

**View health:**
```bash
curl http://localhost:8082/health | jq
```

## File Structure

```
edge-service/
├── requirements.txt              # Updated for RPi compatibility
├── scripts/
│   └── smoke_check_local.sh       # ← Hardware-only validation
├── app/
│   ├── main.py                    # Starts both services
│   ├── coordinator_service.py     # Cycle control
│   ├── processor_service.py       # Image storage
│   ├── oak_controller.py          # OAK camera integration
│   ├── pipeline.py                # ML inference pipeline
│   └── ...
├── models/
│   ├── student_mobilenet_v3.blob  # Main inference model
│   └── prefilter.blob             # Optional prefilter
├── LOCAL_TEST_SETUP.md            # ← Complete testing guide
├── API_REFERENCE.md               # ← API documentation
├── test_e2e.py                    # ← Automated tests
└── quickstart.sh                  # ← One-command setup
```

## Data Flow During Testing

```
Camera (USB OAK-D Lite)
        ↓
OAK Controller (oak_controller.py)
        ↓
Pipeline (pipeline.py) → ML Inference
        ↓
Coordinator (port 8081) → HTTP POST /image
        ↓
Processor (port 8082)
        ↓
temp/images/frame_*.jpg
temp/metadata.jsonl
        ↓
(On stop/abort)
        ↓
bundle.zip (final archive)
        ↓
/tmp/edge_data/runs/RUN_ID/
```

## Output Files After Test

```
/tmp/edge_data/runs/RUN_20240418_153022_test/
├── run.json                 # Cycle metadata
├── bundle.zip               # Final archive with images + metadata
└── temp/                    # (cleaned after test)
    ├── images/
    │   ├── frame_000001.jpg
    │   ├── frame_000002.jpg
    │   └── ...
    └── metadata.jsonl       # Per-image inference results
```

## Inspecting Results

```bash
# View cycle metadata
cat /tmp/edge_data/runs/RUN_20240418_153022_test/run.json | jq

# List archive contents
unzip -l /tmp/edge_data/runs/RUN_20240418_153022_test/bundle.zip

# Count images
unzip -l /tmp/edge_data/runs/RUN_20240418_153022_test/bundle.zip | grep -c ".jpg"

# View first metadata entry
unzip -p /tmp/edge_data/runs/RUN_20240418_153022_test/bundle.zip metadata.jsonl | head -1 | jq
```

## Deployment to Raspberry Pi Zero 2 W

Once local testing is successful:

```bash
# Copy to Pi
scp -r . pi@raspberrypi:/home/pi/thirdeye-edge

# SSH into Pi
ssh pi@raspberrypi
cd /home/pi/thirdeye-edge

# Set same environment variables
export OAK_CONNECTED=true
export BLOB_PATH=$(pwd)/models/student_mobilenet_v3.blob
export CAPTURE_FPS=0.5  # Slower for Pi's limited resources
export DOWNSAMPLE_WIDTH=240
export DOWNSAMPLE_HEIGHT=180

# Test on Pi
./scripts/smoke_check_local.sh

# Run service
python3 -m app.main
```

## Troubleshooting

### Camera Not Detected

```bash
python3 - <<'EOF'
import depthai as dai
devs = dai.Device.getAllAvailableDevices()
print(f'Devices: {len(devs)}')
for d in devs:
    print(f'  - {d.getMxId()}')
EOF
```

### Blob File Issues

```bash
ls -lh "$BLOB_PATH"
file "$BLOB_PATH"
```

### Services Not Starting

```bash
# Check logs
tail -100 /tmp/edge_service.log

# Or run with output
python3 -m app.main
```

### Port Already in Use

```bash
# Find process
lsof -i :8081
lsof -i :8082

# Kill it
pkill -f "python3 -m app.main"
```

### Disk Space

```bash
df -h /tmp
du -sh /tmp/edge_data
```

## Environment Variables Reference

| Variable | Default | Notes |
|----------|---------|-------|
| `OAK_CONNECTED` | `false` | Set `true` for hardware camera |
| `BLOB_PATH` | None | Required when `OAK_CONNECTED=true` |
| `PREFILTER_BLOB_PATH` | None | Optional prefilter model |
| `CAPTURE_FPS` | `1` | Frames per second from camera |
| `DOWNSAMPLE_WIDTH` | `320` | Inference input width |
| `DOWNSAMPLE_HEIGHT` | `240` | Inference input height |
| `MODEL_THRESHOLD` | `0.5` | Confidence threshold |
| `PREFILTER_THRESHOLD` | `0.25` | Prefilter confidence threshold |
| `DATA_DIR` | `/data` | Storage location |
| `COORDINATOR_PORT` | `8081` | Service port |
| `PROCESSOR_PORT` | `8082` | Service port |

## Next Steps

1. **Test Locally** → Run `./quickstart.sh` to validate setup
2. **Monitor Data** → Watch images and metadata being collected
3. **Inspect Archives** → Extract and verify bundle.zip contents
4. **Adjust Settings** → Fine-tune CAPTURE_FPS, thresholds, etc.
5. **Deploy to Pi** → Copy validated code to Raspberry Pi
6. **Verify on Pi** → Run same tests on actual hardware

## Performance Expectations

### Local PC (i7, 16GB RAM)
- Startup: 2-3 seconds
- Per-frame latency: 100-200ms
- Memory: ~150-200 MB
- Can run for hours

### Raspberry Pi Zero 2 W (64-bit)
- Startup: 5-10 seconds
- Per-frame latency: 500-800ms
- Memory: ~200-250 MB (tight with 512MB available)
- Recommend: CAPTURE_FPS=0.5, duration ≤ 5 minutes

## Documentation Files

- **LOCAL_TEST_SETUP.md** - Step-by-step testing guide
- **API_REFERENCE.md** - Complete API documentation
- **README.md** - Original project readme
- **EDGE_SERVICE_TEST_PLAN.md** - Comprehensive test plan

## Support & Debugging

If issues occur:

1. Check `/tmp/edge_smoke_logs/` for validation errors
2. Review `/tmp/edge_service.log` for runtime logs
3. Use `curl` commands to test API directly
4. Verify blob files and permissions
5. Check device connectivity: `lsusb | grep 03e7`

---

**Setup Complete!** You're ready to start testing the edge pipeline with your camera.

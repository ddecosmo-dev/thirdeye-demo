# Local Edge Service Testing Guide

This guide covers testing the edge pipeline on your PC with a USB-connected camera (OAK Lite) before deploying to Raspberry Pi Zero 2 W.

## Overview

The edge service consists of two microservices running on localhost:
- **Coordinator** (port 8081): Controls cycle start/stop/abort and camera capture
- **Processor** (port 8082): Stores images and metadata, creates final zip archive

## Prerequisites

### 1. System Requirements
- **OS**: Linux (Ubuntu 20.04+), macOS (Intel/Apple Silicon), or Windows (WSL2)
- **Python**: 3.9 or later
- **Camera**: OAK-D Lite or similar DepthAI-compatible camera (USB)
- **Disk space**: At least 1 GB for test data

### 2. Software Installation

```bash
# Clone the repository (if not already done)
cd /path/to/thirdeye-demo/demo/edge-services/edge-service

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Model Files

Place your model blob files in the `models/` directory:
```bash
ls -lh models/
# Expected files:
# - student_mobilenet_v3.blob (main inference model)
# - prefilter.blob (optional, for filtering low-quality frames)
```

## Local Testing with USB Camera

### Setup Environment

```bash
# Terminal 1: Set up environment variables
export OAK_CONNECTED=true
export BLOB_PATH=$(pwd)/models/student_mobilenet_v3.blob
export PREFILTER_BLOB_PATH=$(pwd)/models/prefilter.blob  # Optional
export DATA_DIR=/tmp/edge_data
export CAPTURE_FPS=1
export COORDINATOR_PORT=8081
export PROCESSOR_PORT=8082

# Verify blob file exists
ls -lh "$BLOB_PATH"
```

### Run Smoke Check

```bash
# This validates the setup before running full tests
chmod +x scripts/smoke_check_local.sh
./scripts/smoke_check_local.sh
```

**Expected output:**
- ✓ All required packages present
- ✓ Blob file found and readable
- ✓ DepthAI devices detected
- ✓ Coordinator health OK
- ✓ Processor health OK

### Start the Service

```bash
# Terminal 1: Run the edge service
python3 -m app.main

# Expected output:
# * Running on http://0.0.0.0:8082 (Processor)
# * Running on http://0.0.0.0:8081 (Coordinator)
```

### Test Start/Stop Cycle

```bash
# Terminal 2: Start a 30-second capture cycle
curl -X POST http://localhost:8081/cycle/start \
  -H 'Content-Type: application/json' \
  -d '{"label":"test_run","duration_seconds":30}'

# Expected response:
# {"run_id":"RUN_xxxxxx","status":"running"}

# Save the run_id for later use
RUN_ID="RUN_xxxxxx"

# Check coordinator status
curl http://localhost:8081/status | jq

# Check processor status
curl http://localhost:8082/status | jq
```

### Monitor Data Collection

```bash
# Terminal 2: Watch the data directory in real-time
watch -n 1 'ls -lh /tmp/edge_data/runs/'"$RUN_ID"'/temp/images/ | tail -20'

# In another terminal: Check metadata
tail -f /tmp/edge_data/runs/$RUN_ID/temp/metadata.jsonl
```

### Stop Cycle

```bash
# Stop the capture (cycle runs for 30s, but you can stop early)
curl -X POST http://localhost:8081/cycle/stop

# Expected response:
# {"run_id":"RUN_xxxxxx","status":"stopped"}

# Check for the final bundle
ls -lh /tmp/edge_data/runs/$RUN_ID/
# Should see: run.json, bundle.zip, temp/ (if not yet cleaned)
```

### Inspect Results

```bash
# Extract and inspect the zip archive
cd /tmp/edge_data/runs/$RUN_ID
unzip -l bundle.zip | head -20

# View metadata summary
unzip -p bundle.zip metadata.jsonl | jq '.' | head -10

# Count images
unzip -l bundle.zip | grep '.jpg' | wc -l
```

## API Testing Workflows

### Complete Test Workflow (with curl)

```bash
#!/bin/bash
set -e

# 1. Start a 20-second cycle with label
echo "[1/5] Starting cycle..."
RESPONSE=$(curl -s -X POST http://localhost:8081/cycle/start \
  -H 'Content-Type: application/json' \
  -d '{"label":"api_test","duration_seconds":20}')
RUN_ID=$(echo "$RESPONSE" | jq -r '.run_id')
echo "Run ID: $RUN_ID"

# 2. Check initial status
echo "[2/5] Checking status..."
curl -s http://localhost:8081/status | jq '.status'

# 3. Let cycle run for a bit
echo "[3/5] Collecting data for 10 seconds..."
sleep 10

# 4. Stop cycle early
echo "[4/5] Stopping cycle..."
curl -s -X POST http://localhost:8081/cycle/stop | jq

# 5. Verify archive creation
echo "[5/5] Verifying archive..."
sleep 2
if [ -f "/tmp/edge_data/runs/$RUN_ID/bundle.zip" ]; then
    echo "✓ Archive created successfully"
    unzip -l "/tmp/edge_data/runs/$RUN_ID/bundle.zip" | head -5
else
    echo "✗ Archive not found"
    exit 1
fi
```

Save as `test_workflow.sh` and run:
```bash
chmod +x test_workflow.sh
./test_workflow.sh
```

### Abort Cycle

```bash
# Start a cycle
RESPONSE=$(curl -s -X POST http://localhost:8081/cycle/start \
  -H 'Content-Type: application/json' \
  -d '{"label":"abort_test","duration_seconds":60}')
RUN_ID=$(echo "$RESPONSE" | jq -r '.run_id')

# Wait a few seconds
sleep 5

# Abort the cycle
curl -X POST http://localhost:8081/cycle/abort \
  -H 'Content-Type: application/json' \
  -d '{"reason":"manual_abort"}'

# Verify archive (should still be created)
ls -lh /tmp/edge_data/runs/$RUN_ID/
```

## Configuration Reference

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `/data` | Base directory for run storage |
| `OAK_CONNECTED` | `false` | Enable OAK camera inference |
| `BLOB_PATH` | None | Path to main inference model blob |
| `PREFILTER_BLOB_PATH` | None | Path to prefilter model (optional) |
| `CAPTURE_FPS` | `1` | Camera capture frame rate |
| `DOWNSAMPLE_WIDTH` | `320` | Input image width for inference |
| `DOWNSAMPLE_HEIGHT` | `240` | Input image height for inference |
| `MODEL_THRESHOLD` | `0.5` | Confidence threshold for main model |
| `PREFILTER_THRESHOLD` | `0.25` | Confidence threshold for prefilter |
| `NORMALIZE_INPUTS` | `false` | Apply ImageNet normalization |
| `COORDINATOR_PORT` | `8081` | Coordinator service port |
| `PROCESSOR_PORT` | `8082` | Processor service port |
| `COORDINATOR_HOST` | `0.0.0.0` | Coordinator bind address |
| `PROCESSOR_HOST` | `0.0.0.0` | Processor bind address |

## Data Layout

After a cycle completes, data is organized as:

```
/tmp/edge_data/
└── runs/
    └── RUN_20240418_153022_test_run/
        ├── run.json              # Cycle metadata
        ├── bundle.zip            # Final archive
        └── temp/
            ├── metadata.jsonl    # Line-delimited JSON per frame
            └── images/
                ├── frame_000001.jpg
                ├── frame_000002.jpg
                └── ...
```

### bundle.zip Contents

```
bundle.zip
├── metadata.jsonl        # Each line: {"image":"frame_000001.jpg","model_score":0.87,...}
├── images/
│   ├── frame_000001.jpg
│   ├── frame_000002.jpg
│   └── ...
└── run.json             # Summary of the cycle
```

## Troubleshooting

### Camera Not Detected

```bash
# Check if OAK device is visible
python3 - <<'EOF'
import depthai as dai
devs = dai.Device.getAllAvailableDevices()
print(f'Devices found: {len(devs)}')
for d in devs:
    print(f'  - {d.getMxId()}')
EOF

# On Linux, check USB permissions
lsusb | grep "03e7"  # Luxonis devices
sudo usermod -a -G dialout,plugdev $USER

# Restart udev rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

### Blob File Not Found

```bash
# Verify path is set correctly
echo "BLOB_PATH=$BLOB_PATH"
ls -lh "$BLOB_PATH"

# If relative path, ensure you're in the edge-service directory
cd /path/to/thirdeye-demo/demo/edge-services/edge-service
```

### Disk Space Issues

```bash
# Check available space
df -h /tmp

# Clear previous test data
rm -rf /tmp/edge_data

# Check minimum free space requirement
du -sh /tmp/edge_data/runs/
```

### High Memory Usage

For Raspberry Pi Zero 2 W (limited RAM):
- Lower `CAPTURE_FPS` (e.g., 0.5 for 1 frame every 2 seconds)
- Reduce `DOWNSAMPLE_WIDTH/HEIGHT` (e.g., 240x180)
- Set reasonable `duration_seconds` (< 300 seconds for low-end hardware)

### Port Already in Use

```bash
# Find process using port 8081 or 8082
lsof -i :8081
lsof -i :8082

# Kill it
pkill -f "python3 -m app.main"

# Or use different ports
export COORDINATOR_PORT=9081
export PROCESSOR_PORT=9082
```

## Next Steps: Deploying to Raspberry Pi Zero 2 W

Once local testing succeeds:

1. **Copy validated code to Pi**:
   ```bash
   scp -r . pi@raspberrypi:/home/pi/thirdeye-edge
   ```

2. **SSH into Pi and test**:
   ```bash
   ssh pi@raspberrypi
   cd thirdeye-edge
   export BLOB_PATH=$(pwd)/models/student_mobilenet_v3.blob
   export OAK_CONNECTED=true
   ./scripts/smoke_check_local.sh
   ```

3. **Run as service** (optional):
   ```bash
   # Create systemd service file
   sudo tee /etc/systemd/system/thirdeye-edge.service > /dev/null <<EOF
   [Unit]
   Description=ThirdEye Edge Service
   After=network.target
   
   [Service]
   Type=simple
   User=pi
   WorkingDirectory=/home/pi/thirdeye-edge
   ExecStart=/home/pi/thirdeye-edge/.venv/bin/python3 -m app.main
   Environment="OAK_CONNECTED=true"
   Environment="BLOB_PATH=/home/pi/thirdeye-edge/models/student_mobilenet_v3.blob"
   Restart=on-failure
   RestartSec=10
   
   [Install]
   WantedBy=multi-user.target
   EOF
   
   sudo systemctl daemon-reload
   sudo systemctl enable thirdeye-edge
   sudo systemctl start thirdeye-edge
   sudo systemctl status thirdeye-edge
   ```

## Performance Benchmarks

### Local PC (i7, 16GB RAM)
- Startup time: ~2-3 seconds
- Inference latency: ~100-200ms per frame
- Memory usage: ~150-200 MB

### Raspberry Pi Zero 2 W (64-bit OS, 512MB RAM)
- Startup time: ~5-10 seconds
- Inference latency: ~500-800ms per frame (depending on model)
- Memory usage: ~200-250 MB
- Recommended: 1 FPS capture rate

## Support

Check logs for detailed error information:
```bash
cat /tmp/edge_smoke_logs/run.log
tail -100 /tmp/edge_smoke_logs/edge_app.log
```

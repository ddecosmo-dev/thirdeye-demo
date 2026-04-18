#!/usr/bin/env bash
set -euo pipefail

# Local smoke-check script for edge service with actual OAK camera.
# Validates hardware, blob files, and service startup before testing.
#
# Usage:
#   BLOB_PATH=/path/to/model.blob ./scripts/smoke_check_local.sh

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

LOG_DIR=/tmp/edge_smoke_logs
mkdir -p "$LOG_DIR"
APP_LOG="$LOG_DIR/edge_app.log"
CAMERA_LOG="$LOG_DIR/camera_test.log"
ENV_LOG="$LOG_DIR/env_check.log"

COORDINATOR_PORT="${COORDINATOR_PORT:-8081}"
PROCESSOR_PORT="${PROCESSOR_PORT:-8082}"

echo "======================================" | tee "$LOG_DIR/run.log"
echo "Edge Service Smoke Check: $(date)" | tee -a "$LOG_DIR/run.log"
echo "Camera: OAK (hardware required)" | tee -a "$LOG_DIR/run.log"
echo "======================================" | tee -a "$LOG_DIR/run.log"

echo "[STEP 1] Environment & package checks" | tee -a "$LOG_DIR/run.log"
echo "Logging to $ENV_LOG" | tee -a "$LOG_DIR/run.log"

python3 - <<'PY' 2>&1 | tee "$ENV_LOG"
import sys, importlib
print('Python:', sys.version)
required = ('flask', 'numpy', 'requests', 'httpx', 'PIL')
missing = []
for pkg in required:
    try:
        if pkg == 'PIL':
            import PIL
        else:
            importlib.import_module(pkg)
        print(f'{pkg:15} OK')
    except Exception as e:
        print(f'{pkg:15} MISSING: {e}')
        missing.append(pkg)

# depthai is optional depending on mode
try:
    import depthai
    print(f'{"depthai":15} OK')
except Exception as e:
    print(f'{"depthai":15} OPTIONAL (needed for hardware mode)')

if missing:
    print('\nMissing critical packages:', missing)
    sys.exit(2)
print('\nAll critical packages present')
sys.exit(0)
PY

ENV_RC=$?
if [ "$ENV_RC" -ne 0 ]; then
    echo "ERROR: Environment check failed" | tee -a "$LOG_DIR/run.log"
    exit "$ENV_RC"
fi

echo "[STEP 2] Blob file validation" | tee -a "$LOG_DIR/run.log"

: ${BLOB_PATH:=}
if [ -z "$BLOB_PATH" ]; then
    echo "ERROR: BLOB_PATH not set" | tee -a "$LOG_DIR/run.log"
    echo "Usage: export BLOB_PATH=/path/to/student_mobilenet_v3.blob" | tee -a "$LOG_DIR/run.log"
    exit 3
fi
if [ ! -f "$BLOB_PATH" ]; then
    echo "ERROR: BLOB_PATH file not found: $BLOB_PATH" | tee -a "$LOG_DIR/run.log"
    exit 4
fi
echo "✓ Blob file found: $BLOB_PATH ($(stat -c%s "$BLOB_PATH" 2>/dev/null || echo "?") bytes)" | tee -a "$LOG_DIR/run.log"

echo "[STEP 3] Camera availability check" | tee -a "$LOG_DIR/run.log"
python3 - <<'PY' > "$CAMERA_LOG" 2>&1 || CAM_RC=$?
import sys
try:
    import depthai as dai
    devs = dai.Device.getAllAvailableDevices()
    print(f'Available OAK devices: {len(devs)}')
    for d in devs:
        print(f'  - {d.getMxId()}')
    if not devs:
        print('ERROR: No OAK camera detected')
        sys.exit(6)
    print('✓ OAK camera check passed')
    sys.exit(0)
except Exception as e:
    print(f'ERROR: DepthAI check failed: {e}')
    sys.exit(8)
PY
CAM_RC=${CAM_RC:-$?}
if [ "$CAM_RC" -ne 0 ]; then
    echo "✗ Camera check failed (code $CAM_RC)" | tee -a "$LOG_DIR/run.log"
    cat "$CAMERA_LOG" | tee -a "$LOG_DIR/run.log"
    exit "$CAM_RC"
fi
cat "$CAMERA_LOG" | tee -a "$LOG_DIR/run.log"

echo "[STEP 4] Start edge service (background) and collect logs" | tee -a "$LOG_DIR/run.log"
echo "Service log -> $APP_LOG" | tee -a "$LOG_DIR/run.log"

# Set up environment for hardware mode
export DATA_DIR="${DATA_DIR:-/tmp/edge_data}"
export OAK_CONNECTED=true
export COORDINATOR_PORT="$COORDINATOR_PORT"
export PROCESSOR_PORT="$PROCESSOR_PORT"

# Create data directory
mkdir -p "$DATA_DIR"

echo "Environment setup:" | tee -a "$LOG_DIR/run.log"
echo "  DATA_DIR=$DATA_DIR" | tee -a "$LOG_DIR/run.log"
echo "  OAK_CONNECTED=true" | tee -a "$LOG_DIR/run.log"
echo "  BLOB_PATH=$BLOB_PATH" | tee -a "$LOG_DIR/run.log"
echo "  COORDINATOR_PORT=$COORDINATOR_PORT" | tee -a "$LOG_DIR/run.log"
echo "  PROCESSOR_PORT=$PROCESSOR_PORT" | tee -a "$LOG_DIR/run.log"

# Clean up any previous instances
pkill -f "python3 -m app.main" || true
sleep 1

# Start service in background
nohup python3 -m app.main > "$APP_LOG" 2>&1 &
APP_PID=$!
echo "Started edge service PID=$APP_PID" | tee -a "$LOG_DIR/run.log"

# Wait for service to start
echo "Waiting 5s for service to start..." | tee -a "$LOG_DIR/run.log"
sleep 5

# Check if process is still running
if ! kill -0 "$APP_PID" 2>/dev/null; then
    echo "ERROR: Service exited unexpectedly. Check $APP_LOG" | tee -a "$LOG_DIR/run.log"
    cat "$APP_LOG" | tee -a "$LOG_DIR/run.log"
    exit 9
fi

echo "[STEP 5] API health checks" | tee -a "$LOG_DIR/run.log"
sleep 2

# Test coordinator health
echo "Testing coordinator health..." | tee -a "$LOG_DIR/run.log"
if curl -sSf "http://localhost:${COORDINATOR_PORT}/health" -m 3 > "$LOG_DIR/coord_health.json" 2>&1; then
    echo "✓ Coordinator health OK" | tee -a "$LOG_DIR/run.log"
    cat "$LOG_DIR/coord_health.json" | tee -a "$LOG_DIR/run.log"
else
    echo "✗ Coordinator health check failed" | tee -a "$LOG_DIR/run.log"
    cat "$LOG_DIR/coord_health.json" 2>/dev/null | tee -a "$LOG_DIR/run.log"
fi

# Test processor health
echo "Testing processor health..." | tee -a "$LOG_DIR/run.log"
if curl -sSf "http://localhost:${PROCESSOR_PORT}/health" -m 3 > "$LOG_DIR/proc_health.json" 2>&1; then
    echo "✓ Processor health OK" | tee -a "$LOG_DIR/run.log"
    cat "$LOG_DIR/proc_health.json" | tee -a "$LOG_DIR/run.log"
else
    echo "✗ Processor health check failed" | tee -a "$LOG_DIR/run.log"
    cat "$LOG_DIR/proc_health.json" 2>/dev/null | tee -a "$LOG_DIR/run.log"
fi

echo "[STEP 6] Coordinator status check" | tee -a "$LOG_DIR/run.log"
if curl -sSf "http://localhost:${COORDINATOR_PORT}/status" -m 3 > "$LOG_DIR/coord_status.json" 2>&1; then
    echo "Coordinator status:" | tee -a "$LOG_DIR/run.log"
    cat "$LOG_DIR/coord_status.json" | tee -a "$LOG_DIR/run.log"
fi

echo "[STEP 7] Processor status check" | tee -a "$LOG_DIR/run.log"
if curl -sSf "http://localhost:${PROCESSOR_PORT}/status" -m 3 > "$LOG_DIR/proc_status.json" 2>&1; then
    echo "Processor status:" | tee -a "$LOG_DIR/run.log"
    cat "$LOG_DIR/proc_status.json" | tee -a "$LOG_DIR/run.log"
fi

echo "[STEP 8] Stopping service" | tee -a "$LOG_DIR/run.log"
kill "$APP_PID" 2>/dev/null || true
wait "$APP_PID" 2>/dev/null || true
sleep 1

echo "======================================" | tee -a "$LOG_DIR/run.log"
echo "Smoke check finished: $(date)" | tee -a "$LOG_DIR/run.log"
echo "All logs available in: $LOG_DIR" | tee -a "$LOG_DIR/run.log"
echo "======================================" | tee -a "$LOG_DIR/run.log"

exit 0

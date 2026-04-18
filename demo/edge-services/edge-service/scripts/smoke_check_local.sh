++#!/usr/bin/env bash
set -euo pipefail

# Local smoke-check script (no Docker).
# - Verifies required Python packages (depthai, numpy, requests, Pillow)
# - Verifies blob file exists
# - Runs a single camera inference via app/utils/test_camera.py if present
#   otherwise performs a minimal DepthAI device probe
# - Starts the full app for 30s, captures verbose logs, and analyzes them for startup markers
# - Produces an exit code >0 on failures and writes logs to /tmp

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT_DIR"

LOG_DIR=/tmp/edge_smoke_logs
mkdir -p "$LOG_DIR"
APP_LOG="$LOG_DIR/edge_app.log"
CAMERA_LOG="$LOG_DIR/camera_test.log"
ENV_LOG="$LOG_DIR/env_check.log"

echo "Smoke check started: $(date)" | tee "$LOG_DIR/run.log"

echo "[STEP 1] Environment & package checks" | tee -a "$LOG_DIR/run.log"
echo "Logging to $ENV_LOG" | tee -a "$LOG_DIR/run.log"

python3 - <<'PY' 2>&1 | tee "$ENV_LOG"
import sys, importlib
print('Python:', sys.version)
required = ('depthai','numpy','requests','PIL')
missing = []
for pkg in required:
    try:
        if pkg == 'PIL':
            import PIL
        else:
            importlib.import_module(pkg)
        print(pkg, 'OK')
    except Exception as e:
        print(pkg, 'MISSING:', e)
        missing.append(pkg)
if missing:
    print('\nMissing packages:', missing)
    sys.exit(2)
print('\nAll required packages present')
sys.exit(0)
PY

echo "[STEP 2] Blob file check" | tee -a "$LOG_DIR/run.log"
: ${BLOB_PATH:=}
if [ -z "$BLOB_PATH" ]; then
  echo "ERROR: BLOB_PATH is not set. Export BLOB_PATH=/path/to/model.blob" | tee -a "$LOG_DIR/run.log"
  exit 3
fi
if [ ! -f "$BLOB_PATH" ]; then
  echo "ERROR: BLOB_PATH file not found: $BLOB_PATH" | tee -a "$LOG_DIR/run.log"
  exit 4
fi
echo "Found blob: $BLOB_PATH" | tee -a "$LOG_DIR/run.log"

echo "[STEP 3] Camera pipeline test (single inference)" | tee -a "$LOG_DIR/run.log"
echo "Logging camera output to $CAMERA_LOG" | tee -a "$LOG_DIR/run.log"

if [ -f app/utils/test_camera.py ]; then
  echo "Running app/utils/test_camera.py --single --blob \"$BLOB_PATH\"" | tee -a "$LOG_DIR/run.log"
  if python3 app/utils/test_camera.py --single --blob "$BLOB_PATH" > "$CAMERA_LOG" 2>&1; then
    echo "Camera test script returned success" | tee -a "$LOG_DIR/run.log"
  else
    echo "Camera test script FAILED - see $CAMERA_LOG" | tee -a "$LOG_DIR/run.log"
    tail -n 200 "$CAMERA_LOG" | sed -n '1,200p' >&2
    exit 5
  fi
else
  echo "No test_camera.py found; performing minimal DepthAI probe" | tee -a "$LOG_DIR/run.log"
  python3 - <<PY > "$CAMERA_LOG" 2>&1 || CAM_RC=$?
import sys
try:
    import depthai as dai
    devs = dai.Device.getAllAvailableDevices()
    print('DepthAI devices:', devs)
    if not devs:
        sys.exit(6)
    # optional: attempt a simple pipeline load if blob present
    try:
        p = dai.Pipeline()
        print('Pipeline created')
        sys.exit(0)
    except Exception as e:
        print('Pipeline creation failed:', e)
        sys.exit(7)
except Exception as e:
    print('DepthAI probe failed:', e)
    sys.exit(8)
PY
  CAM_RC=${CAM_RC:-$?}
  if [ "$CAM_RC" -ne 0 ]; then
    echo "DepthAI probe failed (code $CAM_RC). See $CAMERA_LOG" | tee -a "$LOG_DIR/run.log"
    tail -n 200 "$CAMERA_LOG" | sed -n '1,200p' >&2
    exit 6
  else
    echo "DepthAI probe passed" | tee -a "$LOG_DIR/run.log"
  fi
fi

echo "[STEP 4] Start full app for 30s and collect logs" | tee -a "$LOG_DIR/run.log"
echo "App log -> $APP_LOG" | tee -a "$LOG_DIR/run.log"

# Start app in background
pkill -f "python3 -m app.main" || true
nohup python3 -m app.main > "$APP_LOG" 2>&1 &
APP_PID=$!
echo "Started app PID=$APP_PID" | tee -a "$LOG_DIR/run.log"

sleep 30

echo "Stopping app PID=$APP_PID" | tee -a "$LOG_DIR/run.log"
kill "$APP_PID" || true
wait "$APP_PID" 2>/dev/null || true

echo "Analyzing app log for startup markers" | tee -a "$LOG_DIR/run.log"
if grep -qi -E "started|listening|serving|ready|http" "$APP_LOG"; then
  echo "Startup markers found in app log" | tee -a "$LOG_DIR/run.log"
else
  echo "No clear startup markers found in $APP_LOG" | tee -a "$LOG_DIR/run.log"
  echo "Last 200 lines of app log:" | tee -a "$LOG_DIR/run.log"
  tail -n 200 "$APP_LOG" | sed -n '1,200p' >&2
  exit 9
fi

echo "[STEP 5] Basic API/endpoint checks (best-effort)" | tee -a "$LOG_DIR/run.log"
# attempt coordinator health and processor health where configured
COORD_PORT=8081
PROC_PORT=8082
set +e
curl -sSf "http://localhost:${COORD_PORT}/health" -m 5 > "$LOG_DIR/coord_health.json" 2>&1
COORD_RC=$?
curl -sSf "http://localhost:${PROC_PORT}/health" -m 5 > "$LOG_DIR/proc_health.json" 2>&1
PROC_RC=$?
set -e

if [ "$COORD_RC" -ne 0 ]; then
  echo "Coordinator health check failed (rc=$COORD_RC). See $LOG_DIR/coord_health.json" | tee -a "$LOG_DIR/run.log"
else
  echo "Coordinator health OK" | tee -a "$LOG_DIR/run.log"
fi
if [ "$PROC_RC" -ne 0 ]; then
  echo "Processor health check failed (rc=$PROC_RC). See $LOG_DIR/proc_health.json" | tee -a "$LOG_DIR/run.log"
else
  echo "Processor health OK" | tee -a "$LOG_DIR/run.log"
fi

echo "Smoke-check finished: $(date)" | tee -a "$LOG_DIR/run.log"
echo "Logs are available in $LOG_DIR" | tee -a "$LOG_DIR/run.log"

exit 0

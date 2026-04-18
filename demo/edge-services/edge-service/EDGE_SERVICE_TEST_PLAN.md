# Edge Service Test Plan

Goal: expose failures early, isolate them quickly, and provide remedies for hardware + data pipeline issues.

## Setup and sanity
- Validate dependencies on device: `python -c "import depthai"`
- Confirm env vars:
	- `export OAK_CONNECTED=true`
	- `export BLOB_PATH=/path/to/student_mobilenet_v3.blob`
	- `export PREFILTER_BLOB_PATH=/path/to/prefilter.blob` (optional)
	- `export CAPTURE_FPS=1`
- Confirm model file exists: `ls -lh "$BLOB_PATH"`
- Confirm device is visible: `depthai_demo` or `python -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"`
- Start services: `python -m app.main`

## Coordinator API behavior
- Start cycle with valid payload -> `status=running`, `run_id` returned.
	- `curl -X POST http://localhost:8081/cycle/start -H 'Content-Type: application/json' -d '{"label":"field","duration_seconds":10}'`
- Start cycle twice without stopping -> second returns 400.
- Start cycle twice without stopping -> second returns 400.
	- repeat the start command immediately
- Start cycle with `duration_seconds=0` or negative -> 400.
- Start cycle with `duration_seconds=0` or negative -> 400.
	- `curl -X POST http://localhost:8081/cycle/start -H 'Content-Type: application/json' -d '{"label":"bad","duration_seconds":0}'`
- Stop cycle while idle -> returns a clean response without exception.
- Stop cycle while idle -> returns a clean response without exception.
	- `curl -X POST http://localhost:8081/cycle/stop`
- Abort cycle while idle -> clean response, no crash.
- Abort cycle while idle -> clean response, no crash.
	- `curl -X POST http://localhost:8081/cycle/abort -H 'Content-Type: application/json' -d '{"reason":"test"}'`
- Duration elapses -> `events` contains `duration_elapsed` and `cycle_complete`.
- Duration elapses -> `events` contains `duration_elapsed` and `cycle_complete`.
	- `curl http://localhost:8081/events`

Remedies:
- If `duration_seconds` rejects, inspect request JSON and log output.
- If `events` empty, verify coordinator logging and API version.

## Processor API behavior
- `/run/start` then `/image` -> stored image and metadata line written.
	- `curl -X POST http://localhost:8082/run/start -H 'Content-Type: application/json' -d '{"run_id":"TEST","label":"manual"}'`
- `/image` without `metadata_json` -> 400.
- `/image` without `metadata_json` -> 400.
	- `curl -X POST http://localhost:8082/image -F run_id=TEST -F image=@/path/to/image.jpg`
- `/image` with invalid JSON -> 400 and log error.
- `/image` with invalid JSON -> 400 and log error.
	- `curl -X POST http://localhost:8082/image -F run_id=TEST -F metadata_json='{bad}' -F image=@/path/to/image.jpg`
- `/run/stop` -> `bundle.zip` created, temp removed.
- `/run/stop` -> `bundle.zip` created, temp removed.
	- `curl -X POST http://localhost:8082/run/stop -H 'Content-Type: application/json' -d '{"run_id":"TEST"}'`
- `/run/abort` -> archive created, status marked aborted.
- `/run/abort` -> archive created, status marked aborted.
	- `curl -X POST http://localhost:8082/run/abort -H 'Content-Type: application/json' -d '{"run_id":"TEST"}'`

Remedies:
- If archive missing, check filesystem permissions and disk free space.
- If temp not removed, verify processor finalization logs.

## Luxonis camera pipeline
- Set `OAK_CONNECTED=true` and valid `BLOB_PATH` -> pipeline starts and logs.
	- `export OAK_CONNECTED=true`
	- `export BLOB_PATH=/path/to/student_mobilenet_v3.blob`
- Invalid `BLOB_PATH` -> startup error with descriptive message.
- Invalid `BLOB_PATH` -> startup error with descriptive message.
	- `export BLOB_PATH=/path/to/missing.blob`
- Unplug camera mid-run -> expect ingest failures and logged errors.
- Switch USB port/cable and retry -> confirm device reconnects.

Remedies:
- If device not detected, run `depthai_demo` to confirm driver/hardware.
- If model fails to load, verify blob compatibility with device.

## ML inference validation
- Verify `model_score` and `model_passed` populate from device outputs.
	- `tail -n 5 /data/runs/<run_id>/temp/metadata.jsonl`
- If `PREFILTER_BLOB_PATH` is set, verify prefilter fields exist.
- If `PREFILTER_BLOB_PATH` is set, verify prefilter fields exist.
	- `grep -m 1 prefilter_score /data/runs/<run_id>/temp/metadata.jsonl`
- If prefilter missing, confirm `prefilter_passed=true` and tag uses model score.

Remedies:
- If scores are always 0 or missing, verify output layer names and blob outputs.
- If prefilter never passes, lower `PREFILTER_THRESHOLD` and re-check.

## Image normalization check
- Enable `NORMALIZE_INPUTS=true` and ensure warnings are absent in logs.
	- `export NORMALIZE_INPUTS=true`
- If warnings appear, confirm DepthAI version supports mean/std in ImageManip.
- If still unsupported, use a blob that bakes ImageNet normalization.

Remedies:
- If normalization unsupported, rebuild blob with normalization baked in.

## Storage and disk pressure
- Fill disk to below `MIN_FREE_DISK_BYTES` -> cycle start should fail.
	- `dd if=/dev/zero of=/data/fill.bin bs=1M count=500`
- Large images near `MAX_IMAGE_BYTES` -> ensure validation rejects oversized frames.
- Run long duration -> verify `bundle.zip` size and file count.

Remedies:
- Adjust `MIN_FREE_DISK_BYTES` and `MAX_IMAGE_BYTES` in env.
- Add cleanup script for older runs if disk pressure is frequent.

## Performance and throughput
- Run at `CAPTURE_FPS=1`, `5`, `10` and compare dropped frames.
	- `export CAPTURE_FPS=5`
- Validate queue behavior when processor is slower than capture.
- Measure CPU and memory on Pi Zero 2 W during a 10-minute run.

Remedies:
- Reduce `CAPTURE_FPS` or lower camera resolution.
- If CPU high, consider offloading more operations to the camera.

## Fault injection
- Stop processor service during run -> coordinator should log ingest failure.
	- `pkill -f processor_service.py`
- Stop coordinator during run -> processor should remain stable and allow manual stop.
- Drop network between coordinator and processor -> events show HTTP errors.

Remedies:
- Restart service, then call `/run/stop` with the last `run_id`.
- Add retry/backoff if errors are frequent.

## Data integrity
- Compare `metadata.jsonl` count to images stored.
	- `wc -l /data/runs/<run_id>/temp/metadata.jsonl`
	- `ls /data/runs/<run_id>/temp/images | wc -l`
- Verify `bundle.zip` contains `metadata.jsonl` and all images.
- Verify `bundle.zip` contains `metadata.jsonl` and all images.
	- `unzip -l /data/runs/<run_id>/bundle.zip | head -n 20`
- Open stored images to confirm valid JPEGs from encoder.

Remedies:
- If mismatch, check for partial writes and file locking.
- If JPEGs are corrupted, inspect encoder settings and camera output.

## Regression checks

## Regression checks
- Device path should not run any ML on CPU.

Remedies:
- Ensure `OAK_CONNECTED` is accurate in each environment.

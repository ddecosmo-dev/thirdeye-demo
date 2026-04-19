# ~/thirdeye_project/headless_iqa_v3.py
"""
Third Eye — on-device IQA inference demo (headless).
Pipeline: Camera -> 320x240 resize -> NeuralNetwork (DINOv2-distilled MobileNetV3)
          -> host (prints scenic score per frame)
"""
import argparse
import datetime
import json
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import depthai as dai
import numpy as np

#fix if needed
#BLOB_PATH = Path("~/thirdeye_project/student_mobilenet_v3.blob").expanduser()
BLOB_PATH = Path(__file__).resolve().parent / "student_mobilenet_v3.blob"
RUN_SECONDS = 30
SET_FPS = 2.0

# Prefilter cutoff defaults, matching the shape of the example in steps.md
PREFILTER_MIN_INTENSITY = 20.0
PREFILTER_MAX_INTENSITY = 235.0
PREFILTER_BLUR_THRESHOLD = 100.0

# NN input shape (must match what the blob was compiled for)
NN_W, NN_H = 320, 240


def run_prefilter(
    frame: np.ndarray,
    blur_thresh: float = PREFILTER_BLUR_THRESHOLD,
    min_intensity: float = PREFILTER_MIN_INTENSITY,
    max_intensity: float = PREFILTER_MAX_INTENSITY,
) -> tuple[bool, str, float, float]:
    """Run a fast image-quality prefilter on raw BGR camera frames."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_intensity = float(np.mean(gray))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if mean_intensity < min_intensity:
        return False, "Too Dark", mean_intensity, laplacian_var
    if mean_intensity > max_intensity:
        return False, "Overexposed", mean_intensity, laplacian_var
    if laplacian_var < blur_thresh:
        return False, "Too Blurry", mean_intensity, laplacian_var

    return True, "Passed", mean_intensity, laplacian_var


def save_log(entries: list[dict[str, Any]], path: Path) -> None:
    payload = {
        "run_id": path.stem,
        "created_at": datetime.datetime.now().isoformat(),
        "entries": entries,
    }
    path.write_text(json.dumps(payload, indent=2))


parser = argparse.ArgumentParser(description="Run headless IQA with a raw-image prefilter.")
parser.add_argument("--run-seconds", type=int, default=RUN_SECONDS, help="How many seconds to run the pipeline.")
parser.add_argument("--blur-thresh", type=float, default=PREFILTER_BLUR_THRESHOLD, help="Minimum Laplacian variance for non-blurry frames.")
parser.add_argument("--min-intensity", type=float, default=PREFILTER_MIN_INTENSITY, help="Minimum mean intensity to avoid too-dark frames.")
parser.add_argument("--max-intensity", type=float, default=PREFILTER_MAX_INTENSITY, help="Maximum mean intensity to avoid overexposure.")
parser.add_argument("--log-dir", type=Path, default=None, help="Directory to store the JSON log file.")
args = parser.parse_args()

log_root = args.log_dir or Path(tempfile.gettempdir()) / "thirdeye_iqa_logs"
log_root.mkdir(parents=True, exist_ok=True)
run_id = f"run_{datetime.datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
log_file = log_root / f"{run_id}.json"
log_entries: list[dict[str, Any]] = []
print(f"Logging JSON to {log_file}")

pipeline = dai.Pipeline()

# Camera node
cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)

# Request a 320x240 BGR output directly from the camera
# (The blob has --reverse_input_channels baked in, so it wants BGR at the input
#  and flips to RGB internally before the ImageNet normalization.)
cam_out = cam.requestOutput(
    size=(NN_W, NN_H),
    type=dai.ImgFrame.Type.BGR888p,  # planar BGR, matches blob's -ip U8 expectation
    fps = SET_FPS,    # frames-per-second. Minimum FPS of the sensor config: 1.4. Do not set below 1.4
)

# Neural network node
nn = pipeline.create(dai.node.NeuralNetwork)
nn.setBlobPath(str(BLOB_PATH))
nn.setNumInferenceThreads(2)

# Wire camera output into the NN
cam_out.link(nn.input)

# Queue to pull NN results and raw frames to the host directly from the camera output
nn_queue = nn.out.createOutputQueue()
preview_queue = cam_out.createOutputQueue()

print(f"Starting pipeline — will run for {args.run_seconds}s")
pipeline.start()

start = time.monotonic()
frame_count = 0
score_sum = 0.0

try:
    while time.monotonic() - start < args.run_seconds:
        preview_packet = preview_queue.get()
        if preview_packet is None:
            continue

        raw_frame = preview_packet.getCvFrame()
        prefilter_passed, prefilter_reason, mean_intensity, blur_var = run_prefilter(
            raw_frame,
            blur_thresh=args.blur_thresh,
            min_intensity=args.min_intensity,
            max_intensity=args.max_intensity,
        )

        nn_data: dai.NNData = nn_queue.get()
        if nn_data is None:
            continue

        output_tensor = nn_data.getFirstTensor()
        score = float(output_tensor.flatten()[0])

        frame_count += 1
        score_sum += score
        entry = {
            "frame_index": frame_count,
            "timestamp": datetime.datetime.now().isoformat(),
            "prefilter_passed": prefilter_passed,
            "prefilter_reason": prefilter_reason,
            "mean_intensity": round(mean_intensity, 2),
            "blur_variance": round(blur_var, 2),
            "scenic_score": round(score, 4),
        }
        log_entries.append(entry)
        save_log(log_entries, log_file)

        print(
            f"Frame {frame_count:4d}  prefilter={prefilter_reason} "
            f"(mean={mean_intensity:.1f}, blur={blur_var:.1f})  scenic_score={score:.4f}"
        )

except KeyboardInterrupt:
    print("\nInterrupted by user.")

elapsed = time.monotonic() - start
pipeline.stop()

print(f"\n--- Summary ---")
print(f"Frames processed: {frame_count}")
print(f"Elapsed: {elapsed:.2f}s")
print(f"FPS: {frame_count / elapsed:.2f}")
if frame_count > 0:
    print(f"Average score: {score_sum / frame_count:.4f}")
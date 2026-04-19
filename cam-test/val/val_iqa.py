# ~/thirdeye_project/headless_iqa_v3.py
"""
Third Eye — on-device IQA inference demo (headless).
Pipeline: Camera -> 320x240 resize -> NeuralNetwork (DINOv2-distilled MobileNetV3)
          -> host (prints scenic score per frame)
"""
import depthai as dai
import time
from pathlib import Path

#fix if needed
#BLOB_PATH = Path("~/thirdeye_project/student_mobilenet_v3.blob").expanduser()
BLOB_PATH = Path(__file__).resolve().parent / "student_mobilenet_v3.blob"
RUN_SECONDS = 30  
SET_FPS = 2.0

# NN input shape (must match what the blob was compiled for)
NN_W, NN_H = 320, 240

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

# Queue to pull NN results to the host
nn_queue = nn.out.createOutputQueue()

print(f"Starting pipeline — will run for {RUN_SECONDS}s")
pipeline.start()

start = time.monotonic()
frame_count = 0
score_sum = 0.0

try:
    while time.monotonic() - start < RUN_SECONDS:
        # Block until the next NN result (with a short timeout so Ctrl+C works)
        nn_data: dai.NNData = nn_queue.get()
        if nn_data is None:
            continue

        # The blob has a single output, a scalar in [0,1] (sigmoid baked in)
        # Get the first output tensor as a numpy array
        output_tensor = nn_data.getFirstTensor()
        score = float(output_tensor.flatten()[0])

        frame_count += 1
        score_sum += score
        print(f"Frame {frame_count:4d}  scenic_score = {score:.4f}")

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
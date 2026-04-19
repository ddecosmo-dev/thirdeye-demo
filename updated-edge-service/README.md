Third Eye Project — Raspberry Pi + OAK-D Lite Setup
===================================================

OVERVIEW
--------
Third Eye is a hands-free hiking camera system that runs image quality
assessment (IQA) AI models on-device to automatically capture scenic
photos during a hike. The hardware consists of a Raspberry Pi 4 (host)
connected to a Luxonis OAK-D Lite camera (AI inference via on-board
MyriadX VPU), both powered from a portable USB power bank for mobile
use (or plugged into the wall when testing from home).

This README covers the hardware / Pi / DepthAI setup AND the
PyTorch-to-blob model conversion pipeline (Colab side). For the project
narrative, team structure, decisions, and lessons learned, see CLAUDE.md.


CURRENT STATUS
--------------
Done:
  - Pi SD card reflashed to Raspberry Pi OS Lite 64-bit (Debian 13
    Trixie, aarch64). Previously ran 32-bit Raspbian Bookworm.
  - WiFi, SSH, hostname, and user account pre-configured during flash
    via Raspberry Pi Imager customisation (no monitor/keyboard needed
    on first boot).
  - Python 3.13 virtual environment at ~/thirdeye_project/.venv
  - DepthAI v3 installed (currently pinned to 3.0.0 for compatibility
    with the emotion-recognition example; can be upgraded to 3.5.0
    anytime — see Known Version Caveats below).
  - depthai-nodes 0.3.4, opencv-python-headless 4.10, numpy 2.4
  - udev rule added so DepthAI can access the camera without sudo
  - oak-examples (v3 examples repo) cloned and working
  - Four demos verified end-to-end:
      * headless_mobilenet_v3.py — terminal-only object detection
      * EfficientViT — browser visualizer, 1000-class ViT classifier
      * emotion-recognition — browser visualizer, 2-stage face + emotion
      * headless_iqa_v3.py — **our Third Eye IQA inference**, terminal
        output, custom distilled MobileNetV3 blob, prints scenic
        score per frame
  - SSH access confirmed from Mac at hostname `thirdeye`
  - Custom IQA model deployed: DINOv2-Small teacher → MobileNetV3-Small
    student (knowledge distillation), converted to RVC2 .blob format
    (4.96 MB), running on-device. See "MODEL CONVERSION PIPELINE" below.

Why DepthAI v3 (vs the old v2 setup):
  v3 is Luxonis's current, actively-maintained API. It has a cleaner
  pipeline model (no explicit XLinkOut nodes), integrates directly with
  the Luxonis Model Zoo (no manual blob downloads), and supports both
  RVC2 (our OAK-D Lite) and RVC4 (future OAK4) hardware. v3 requires
  a 64-bit Pi OS because Luxonis doesn't publish 32-bit ARM wheels —
  which is why we reflashed.

To do:
  - Validate IQA accuracy on-device (scenic vs non-scenic scenes —
    confirm scores discriminate properly before building further)
  - Add image saving to disk when scenic score crosses a threshold
  - Benchmark inference FPS / latency
  - Test full battery-powered operation for hike duration
  - Add CMU campus WiFi to the Pi (WPA2-Enterprise, requires manual
    wpa_supplicant config — Imager only supports WPA2-Personal)
  - Try out OAK Viewer
  - Fix teammate's (H,W) dimension ordering bug in
    hiking_model_loading.ipynb (teacher model inference transform)


HARDWARE
--------
What you need:
  - Raspberry Pi 4 (hostname: thirdeye, username: pi)
  - Luxonis OAK-D Lite camera
  - Luxonis USB-C Y-adapter (small block with two USB-C sockets)
  - 2x USB-A to USB-C cables
  - Portable USB power bank with at least 2 USB-A ports (mobile use)
    OR a wall outlet with a Pi power supply (stationary testing)

Pi info:
  - Hostname: thirdeye (resolves via mDNS on most networks)
  - Username: pi
  - Password: thirdeye   (set during Imager customisation; change this
    here if yours differs)
  - Pre-configured WiFi: home network only (see "To do" for CMU)
  - SSH enabled by default
  - No auto-login on Lite — requires password at console

Physical connection — plug-in sequence (do in this order):
  1. Verify power bank is charged and turned on.
  2. Plug Y-adapter's USB-C side into the OAK-D Lite camera.
  3. Plug one USB-A->USB-C cable from the power bank into one socket
     of the Y-adapter. The camera should come alive.
  4. Plug the other USB-A->USB-C cable from a USB 3.0 port on the Pi
     (BLUE ports, not black USB 2.0) into the other socket of the
     Y-adapter.
  5. Make sure the Pi itself is powered (power bank's 2nd USB-A port,
     or wall adapter for stationary testing).

IMPORTANT: The OAK-D Lite is powered separately from the Pi's USB
output. Do NOT power the camera only from the Pi — AI inference
creates current spikes that can exceed the Pi's 1.2 A USB budget and
cause brownouts / crashes.


COLD-START FROM POWERED-OFF (for reproducing any demo)
------------------------------------------------------
Prereqs:
  - You are on the same local network as the Pi.
  - You have the Pi password.

Step 1 — Power everything on:
  1. Plug the Pi into wall power.
  2. Wait about 60–90 seconds for boot.
  3. Connect the camera using the plug-in sequence above.

Step 2 — SSH into the Pi from your Mac:
    ssh pi@thirdeye
  If `thirdeye` doesn't resolve, use the IP directly (check your
  router's connected-devices page):
    ssh pi@192.168.1.152   # example — varies by network

  First time on a new network, SSH asks for fingerprint confirmation
  — type `yes`, then the password (which is: thirdeye)

Step 3 — Activate the Python venv:
    cd ~/thirdeye_project
    source .venv/bin/activate
  Prompt should now start with `(.venv)`.

Step 4 — Verify the camera is accessible:
    lsusb
  Look for `Intel Movidius MyriadX`. Then:
    python -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"
  Non-empty list means you're good. Empty list means udev / cable
  issue (see Troubleshooting).

From here, pick one of the four demos below.


DEMO 1 — headless_mobilenet_v3.py (terminal, no browser)
--------------------------------------------------------
What it does: runs MobileNet-SSD on the camera feed for 30 seconds,
prints detected objects (label, confidence, bbox) to the terminal.
No display required. Good for quick sanity checks over SSH.

Run:
    python ~/thirdeye_project/headless_mobilenet_v3.py

Point the camera at yourself, a chair, a bottle, a TV. MobileNet-SSD
recognizes 20 everyday classes. The script exits cleanly after 30s
and prints a final FPS summary.


DEMO 2 — EfficientViT (browser visualizer, ViT-based classifier)
---------------------------------------------------------------
What it does: runs a transformer-based 1000-class ImageNet classifier
on the camera feed, serving a live-annotated visualizer over HTTP.

Run (on the Pi):
    cd ~/thirdeye_project/oak-examples/neural-networks/generic-example
    python main.py --model luxonis/efficientvit:b1-224x224

First run downloads the model from the Luxonis Zoo (~30s). Then on
your Mac, open Chrome and go to:
    http://thirdeye:8082
or if mDNS isn't working:
    http://192.168.1.152:8082   # use the Pi's actual IP

You'll see the live camera feed with the top predicted ImageNet
classes and confidences. Point the camera at objects to see them
classified in real time.

To stop: Ctrl+C in the Pi's SSH terminal.


DEMO 3 — Emotion recognition (browser visualizer, 2-stage pipeline)
-------------------------------------------------------------------
What it does: detects faces using YuNet, then classifies emotion
(anger/contempt/disgust/fear/happy/neutral/sad/surprise) per face.

Run (on the Pi):
    cd ~/thirdeye_project/oak-examples/neural-networks/face-detection/emotion-recognition
    python main.py

First run downloads both models from the Zoo. Then open Chrome to
the same visualizer URL as above:
    http://thirdeye:8082

Look at the camera and make different expressions to see them
recognized live.

To stop: Ctrl+C in the Pi's SSH terminal.


DEMO 4 — Third Eye IQA inference (headless, custom blob)
--------------------------------------------------------
What it does: runs our custom MobileNetV3 IQA model (distilled from
DINOv2-Small) on the camera feed. Prints a scenic quality score
(0 to 1) per frame to the terminal. This is the actual Third Eye
inference pipeline.

The blob at `~/thirdeye_project/student_mobilenet_v3.blob` was
compiled with preprocessing baked in:
  - Input format: BGR U8 at 320x240 (camera-native, NO host-side
    normalization required)
  - Mean subtraction + scale division (ImageNet values) happen inside
    the blob
  - BGR -> RGB channel flip happens inside the blob
  - Final sigmoid activation also baked in (output is already 0-1)

**CRITICAL:** If you modify the DepthAI pipeline, do NOT add any
ImageManip normalization before the NeuralNetwork node — the model
will then be normalized twice and produce garbage scores.

Run:
    python ~/thirdeye_project/headless_iqa_v3.py

Point the camera at different scenes and watch the scores:
  - Keyboard, blank wall, floor, your hand up close → should score low
  - Plants, view out a window, landscapes → should score higher

Configurable: the script sets `cam.setFps(5)` by default to make
terminal output readable. Change to 10-30 for higher throughput
measurements.

To stop: Ctrl+C (or wait 30s for auto-exit).


CLEAN SHUTDOWN
--------------
Always do this before unplugging power:
    sudo shutdown now
Wait ~15s, then unplug. Never yank power on a running Pi — the SD
card can corrupt.


INSTALLED SOFTWARE
------------------
System (apt packages):
  - Raspberry Pi OS Lite 64-bit / Debian 13 (trixie), aarch64
  - Python 3.13.5
  - git, python3-venv, python3-pip
  - SSH server enabled

Python venv (~/thirdeye_project/.venv):
  - depthai 3.0.0           (can upgrade to 3.5.0; see caveats below)
  - depthai-nodes 0.3.4
  - numpy 2.4.4
  - opencv-python-headless 4.10.0.84
  - python-dotenv, pyyaml, requests

udev rule at /etc/udev/rules.d/80-movidius.rules:
    SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"
  Grants the `pi` user access to the Movidius USB device without sudo.

Repos inside ~/thirdeye_project/:
  - oak-examples/  — DepthAI v3 examples repo. Demos 1-3 live here.

Our own code (in ~/thirdeye_project/):
  - headless_mobilenet_v3.py            Demo 1 (stock MobileNet-SSD).
  - headless_mobilenet_v2_REFERENCE.py  Old v2 script, kept as a
                                        frozen reference. Do NOT run
                                        it — it uses the v2 API and
                                        will error out on v3.
  - student_mobilenet_v3.blob           Our custom IQA model,
                                        4.96 MB, compiled for MyriadX.
                                        Input: [1,3,240,320] BGR U8.
                                        Output: [1] scalar in [0,1]
                                        (scenic quality score,
                                        sigmoid baked in).
  - test_blob_load.py                   Sanity-check: creates a
                                        minimal Pipeline+NeuralNetwork
                                        and confirms DepthAI loads the
                                        blob. Requires camera
                                        connected (v3 limitation).
  - headless_iqa_v3.py                  Demo 4 (our IQA pipeline).
  - support_service.py                  Local REST coordinator to start/stop the pipeline.
  - README.md                           This file.

SUPPORT SERVICE
---------------
Run the local coordinator service to start and stop the headless IQA pipeline via HTTP POST.

Start the service:
    python support_service.py

How the service stores run data:
  - Per-test data is saved under `runtime_data/`
  - Completed runs are archived to `outbound/` as a ZIP containing the run directory, images, and JSON metadata
  - Aborted or failed runs are preserved under `inspection/` for later debugging and review
  - After successful archive creation, the temporary `runtime_data/` run directory is removed automatically

Endpoints:
  POST http://127.0.0.1:5000/start
    - JSON body: {"run_seconds": 30, "blur_thresh": 100.0, "min_intensity": 20.0, "max_intensity": 235.0}
    - Optional: {"run_id": "run_20260419_123456_abcd12"}

  POST http://127.0.0.1:5000/stop
    - JSON body: {"mode": "stop"}
    - Gracefully stops the current pipeline and archives the completed run to `outbound/`

  POST http://127.0.0.1:5000/stop
    - JSON body: {"mode": "abort"}
    - Forces the current pipeline to stop and marks it aborted without archiving

  GET  http://127.0.0.1:5000/status
    - Returns raw process status and logging metadata

  GET  http://127.0.0.1:5000/health
    - Returns service health, current state, and archive path when available

  GET  http://127.0.0.1:5000/archive-status
    - Returns whether a completed archive is available and the ZIP path

Example PowerShell calls:
    Invoke-RestMethod -Uri http://127.0.0.1:5000/start -Method Post -ContentType 'application/json' -Body '{"run_seconds":30,"blur_thresh":100.0,"min_intensity":20.0,"max_intensity":235.0}'
    Invoke-RestMethod -Uri http://127.0.0.1:5000/health
    Invoke-RestMethod -Uri http://127.0.0.1:5000/archive-status
    Invoke-RestMethod -Uri http://127.0.0.1:5000/stop -Method Post -Body '{"mode":"stop"}' -ContentType 'application/json'
    Invoke-RestMethod -Uri http://127.0.0.1:5000/stop -Method Post -Body '{"mode":"abort"}' -ContentType 'application/json'

Example curl calls:
    curl -X POST http://127.0.0.1:5000/start -H "Content-Type: application/json" -d '{"run_seconds":30,"blur_thresh":100.0,"min_intensity":20.0,"max_intensity":235.0}'
    curl http://127.0.0.1:5000/health
    curl http://127.0.0.1:5000/archive-status
    curl -X POST http://127.0.0.1:5000/stop -H "Content-Type: application/json" -d '{"mode":"stop"}'
    curl -X POST http://127.0.0.1:5000/stop -H "Content-Type: application/json" -d '{"mode":"abort"}'

The service launches `headless_iqa_v3.py` as a separate process and stores logs under `logs/` by default.

MODEL CONVERSION PIPELINE (PyTorch -> .blob)
---------------------------------------------
Context: our best IQA model uses a DINOv2-Small backbone + MLP head.
DINOv2 is a Vision Transformer and is too heavy for the OAK-D Lite's
MyriadX VPU. So we distill the DINOv2 teacher into a MobileNetV3-Small
student, then convert that student to a MyriadX .blob.

The full pipeline lives in the `knowledge_distiller.ipynb` Colab
notebook. Full details of the conversion saga (what went wrong and
why, multiple dead ends) are in CLAUDE.md under "The MobileNetV3 ->
MyriadX Blob Conversion Journey." This section is a short
reproduction guide.

Pipeline overview:
  (1) Train student via knowledge distillation from the DINOv2 teacher
      on the team's hiking dataset (notebook handles this).
  (2) Rewrite MobileNetV3's HardSigmoid/HardSwish activations using
      ReLU6 equivalents (Myriad doesn't support HardSigmoid directly).
  (3) Wrap the student with a sigmoid so the ONNX/blob output is
      directly a 0-1 score.
  (4) Export to ONNX with the right flags (see below).
  (5) Simplify ONNX with onnxsim.
  (6) ONNX -> OpenVINO IR (.xml + .bin) using OpenVINO 2022.3 in a
      Python 3.10 conda env on Colab. This step bakes in preprocessing.
  (7) IR -> .blob using Luxonis's blobconverter cloud API (because
      compile_tool requires a physical Myriad device, which Colab
      doesn't have but blobconverter's servers do).

KEY GOTCHAS anyone retracing this needs to know:

  * Replace HardSigmoid/HardSwish BEFORE ONNX export, using:
      HardSigmoid(x) = ReLU6(x + 3) / 6
      HardSwish(x)   = x * ReLU6(x + 3) / 6
    Otherwise the Myriad compile step will fail with
    "unsupported layer type HardSigmoid."

  * In torch.onnx.export(), set dynamo=False, opset_version=12,
    do_constant_folding=True. PyTorch 2.6+'s new exporter otherwise
    silently upgrades the opset to 18 which OpenVINO can't read.

  * Use transforms.Resize((240, 320)) for 240-tall x 320-wide
    landscape input. PyTorch's API is (H, W), not (W, H).

  * Do NOT send raw ONNX to blobconverter. Send the pre-compiled IR
    (from step 6) using blobconverter.from_openvino(xml=..., bin=...).
    blobconverter's server runs OpenVINO 2022.1 which has the same
    HardSigmoid problem; sending pre-compiled IR skips that step.

  * In the Model Optimizer (step 6), include these flags so
    preprocessing is baked into the blob:
      --mean_values [123.675,116.28,103.53]
      --scale_values [58.395,57.12,57.375]
      --reverse_input_channels
      --compress_to_fp16
    Combined with `-ip U8` at blobconverter time, this means the Pi
    pipeline feeds raw BGR U8 frames straight in — zero host-side
    preprocessing.

  * OpenVINO 2022.3 requires Python 3.7-3.10. Colab is 3.12. Solution:
    install Miniforge + create a conda env with Python 3.10. Only
    step 6 runs in that env; everything else stays on Colab's
    default Python.


KNOWN VERSION CAVEATS
---------------------
DepthAI is pinned to 3.0.0 (not the latest 3.5.0) because the
emotion-recognition example's requirements.txt hard-pins
`depthai==3.0.0` and `depthai-nodes==0.3.4`. On newer versions, the
example fails because depthai-nodes 0.4.0 changed the
GatherData.build() signature.

To upgrade DepthAI back to latest (works for headless_mobilenet_v3,
EfficientViT, AND our headless_iqa_v3.py — but breaks the emotion demo):
    pip install --upgrade depthai depthai-nodes

To re-pin for the emotion demo:
    pip install -r oak-examples/neural-networks/face-detection/emotion-recognition/requirements.txt

DepthAI v3 also cannot instantiate a `Pipeline()` object without a
camera connected — unlike v2, which could build pipelines offline.
Any test script that creates a pipeline (including test_blob_load.py)
requires the camera plugged in.


HOW THE PI WAS FLASHED (for future rebuilds)
--------------------------------------------
If this Pi ever needs to be reflashed from scratch:

  1. On a Mac/PC, install Raspberry Pi Imager from
     https://www.raspberrypi.com/software/
  2. Insert the microSD card.
  3. In Imager:
       Device:  Raspberry Pi 4
       OS:      Raspberry Pi OS (other) -> Raspberry Pi OS Lite (64-bit)
       Storage: (the SD card)
       Customisation:
         Hostname: thirdeye
         Username: pi
         Password: thirdeye (or your choice)
         WiFi SSID + password (home network; see To Do for CMU)
         SSH: enabled (password auth)
         Timezone: America/New_York (or as appropriate)
  4. Write, then reinsert SD card into the Pi and boot.
  5. SSH in, then run this bring-up sequence:
        sudo apt update
        sudo apt install -y python3-venv python3-pip git
        mkdir -p ~/thirdeye_project && cd ~/thirdeye_project
        python3 -m venv .venv
        source .venv/bin/activate
        pip install --upgrade pip
        pip install depthai depthai-nodes python-dotenv
        echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | \
            sudo tee /etc/udev/rules.d/80-movidius.rules
        sudo udevadm control --reload-rules && sudo udevadm trigger
        git clone --depth 1 https://github.com/luxonis/oak-examples.git
  6. Unplug and replug the camera, then verify with
     `python -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"`
  7. Copy over our custom files (student_mobilenet_v3.blob,
     headless_iqa_v3.py, test_blob_load.py, etc.) via scp from Mac.


TROUBLESHOOTING
---------------
SSH: "REMOTE HOST IDENTIFICATION HAS CHANGED"
  Expected after a Pi reflash — the new install has a different SSH
  host key. Clear the old one on your Mac and retry:
      ssh-keygen -R thirdeye
      ssh pi@thirdeye

SSH: "Connection closed" immediately
  Pi is still booting. Wait 30–60s and retry.

SSH: "Could not resolve hostname thirdeye"
  mDNS not working on your network. Use the Pi's IP directly. Find
  it via your router's connected-devices page, or on your Mac run
  `arp -a` after the Pi has been on the network ~1 minute.

`lsusb` shows camera but DepthAI returns empty device list
  Almost always a udev rule issue. Check:
      cat /etc/udev/rules.d/80-movidius.rules
  Should be exactly:
      SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"
  Watch for typos (we once had SUBSYSTM missing the E — rule silently
  didn't match, camera showed up in lsusb but DepthAI couldn't open
  it). To fix:
      echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | \
          sudo tee /etc/udev/rules.d/80-movidius.rules
      sudo udevadm control --reload-rules && sudo udevadm trigger
  Then unplug and replug the camera.

Visualizer in browser shows nothing / "refused to connect"
  Make sure the Python script is still running on the Pi — the
  visualizer only serves while the pipeline is active. Also try the
  IP URL instead of the hostname.

Example script errors with TypeError on `.build()`
  The depthai-nodes API changed between versions. See Known Version
  Caveats — you may need to re-pin to the example's required
  versions.

Camera shows up on Bus 001 / USB 2.0 instead of USB 3.0
  Cable is USB 2.0-only, or plugged into a black USB 2.0 port on the
  Pi. Works fine for small models but caps bandwidth. Use blue ports
  and a USB 3.0-rated cable for full speed.

test_blob_load.py or headless_iqa_v3.py errors with "No available
devices" or similar before even getting to inference
  The camera isn't being seen by DepthAI. Check the plug-in sequence:
  camera must be powered independently via the Y-adapter, and the
  data cable must go to a blue USB 3.0 port on the Pi. Also verify
  with `lsusb` that "Intel Movidius MyriadX" shows up.

Blob conversion fails with "unsupported layer type HardSigmoid"
  You're trying to convert a MobileNetV3-based model without
  rewriting HardSigmoid/HardSwish first. See MODEL CONVERSION
  PIPELINE above — the ReLU6 rewrite is mandatory, and must happen
  BEFORE ONNX export.

IQA scores are garbage / all similar / obviously wrong values
  Most likely cause: preprocessing is being applied twice. The blob
  has ImageNet normalization and BGR->RGB flip baked in. If the
  DepthAI pipeline also normalizes frames (via an ImageManip node
  with mean/scale, or by feeding float tensors instead of U8), the
  data hitting the model is wrong. Feed raw BGR U8 frames at
  320x240, nothing else.

IQA scores are inverted (scenic things score low, non-scenic high)
  The sigmoid is baked in, so this would indicate something flipped
  in the distillation training itself — not a deployment issue.
  Regenerate the student model from the notebook.


FILE LAYOUT REFERENCE
---------------------
~/thirdeye_project/
  .venv/                              Python virtual env
  oak-examples/                       DepthAI v3 examples repo
    neural-networks/
      generic-example/                used for EfficientViT demo
      face-detection/
        emotion-recognition/          used for emotion demo
  headless_mobilenet_v3.py            Demo 1 (stock MobileNet-SSD)
  headless_mobilenet_v2_REFERENCE.py  Old v2 script, reference only
  student_mobilenet_v3.blob           Our distilled IQA model (4.96 MB)
  test_blob_load.py                   Blob load sanity check
  headless_iqa_v3.py                  Demo 4 (our IQA inference)
  README.md                           This file

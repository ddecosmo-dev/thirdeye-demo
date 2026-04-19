# CLAUDE.md — Third Eye Project Context

## Purpose of this file
This file is context for future Claude conversations on the Third Eye project. It captures the project goals, the specific role of the user (Abdullah), the current state of the hardware/software setup, decisions made and why, and lessons learned the hard way. Read this first before responding to any Third Eye-related question.

---

## The Project

**Third Eye** is a hands-free hiking camera system that automatically captures and curates scenic photos during a hike, so the hiker can stay present in the moment instead of stopping to take pictures.

### Two-stage pipeline
1. **On-device (during hike):** A handheld OAK-D Lite camera with onboard MyriadX VPU compute, tethered to a Raspberry Pi 4 host, runs image quality assessment (IQA) AI models in real time to filter scenic frames as the hiker walks. Both devices run on a portable USB power bank.
2. **Cloud (after hike):** Once the hiker is back online, a cloud pipeline applies further filtering to select a diverse final set of best photos for the hiker's curated memory.

### IQA model approach
The team is benchmarking six architectures (NIMA, TANet, MUSIQ, TOPIQ, DINOv2-Small, DINOv3-Small) across the AVA, ScenicOrNot, and a self-collected Pittsburgh hiking validation dataset. Best results so far: **DINOv2 and DINOv3 Small with fine-tuned MLP heads** on ScenicOrNot. Validation showed no single dataset generalizes well to Pittsburgh hiking, motivating a proposed **multi-head architecture: shared DINOv2 backbone + multiple MLP heads trained on different datasets**.

For on-device deployment, the DINOv2-based teacher was **distilled into a MobileNetV3-Small student** (see "The MobileNetV3 → MyriadX Blob Conversion Journey" below) because DINOv2/v3-scale transformers are too heavy for the RVC2 MyriadX VPU.

### Abdullah's role on the team
Hardware integration owner. Specifically responsible for:
- Setting up the Raspberry Pi 4 + OAK-D Lite + power
- Getting custom AI models running on the OAK-D Lite for on-device IQA
- Pipeline: ONNX export → RVC2 conversion (via Luxonis ModelConverter / blobconverter) → DepthAI pipeline & nodes
- Other teammates handle the ML training, evaluation, and cloud-side curation. Teammate "ddecosmo" owns the GitHub repo (`ddecosmo-dev/24-782_Third_Eye_Project`).

---

## Hardware Inventory

- **Raspberry Pi 4** — hostname `thirdeye`, username `pi`, password `thirdeye` (set during reflash)
- **Luxonis OAK-D Lite camera** — RVC2 platform with MyriadX VPU. Device ID `1944301061AB097E00`.
- **Y-adapter** — small block with two USB-C sockets (for the camera). One side is data + power to Pi, the other is power-only to the bank.
- **2× USB-A to USB-C cables**
- **Portable USB power bank** with at least 2 USB-A ports
- **Mac (user's laptop)** — primary dev machine. SSH client. NOT a dev machine for Pi-side code (Abdullah edits files locally on Mac and `scp`s them over because VS Code Remote-SSH was unstable).

### Power architecture (important)
The OAK-D Lite is powered **independently** from the Pi via the Y-adapter, NOT from the Pi's USB port. AI inference creates current spikes that exceed the Pi's 1.2A USB budget and cause brownouts. This is documented in Luxonis's deployment guide and in the project paper.

---

## Network / Access

- SSH from Mac: `ssh pi@thirdeye`
- Pi is configured for **home WiFi only** at the moment. CMU campus WiFi (WPA2-Enterprise) was deferred — Raspberry Pi Imager only supports WPA2-Personal in its quick-config. Adding CMU is on the to-do list and requires manual `wpa_supplicant` config.
- IP on home WiFi at time of writing: `192.168.1.152`
- IP sticker on the Pi (legacy, from CMU campus): `172.26.254.242`

---

## Current Software State (as of April 16, 2026)

### OS
- **Raspberry Pi OS Lite 64-bit** (Debian 13 "trixie", aarch64, Python 3.13.5)
- Was previously 32-bit Raspbian Bookworm 12 (armv7l, Python 3.11). Reflashed in an earlier session to enable DepthAI v3.

### Python venv at `~/thirdeye_project/.venv`
- `depthai==3.0.0` (pinned — see Caveats)
- `depthai-nodes==0.3.4` (pinned)
- `numpy==2.4.4`
- `opencv-python-headless==4.10.0.84`
- `python-dotenv`, `pyyaml`, `requests`

### udev rule at `/etc/udev/rules.d/80-movidius.rules`
```
SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"
```

### Repos cloned in `~/thirdeye_project/`
- `oak-examples/` — Luxonis's DepthAI v3 examples repo (this is what's used now)

### Custom code in `~/thirdeye_project/`
- `headless_mobilenet_v3.py` — current working v3 demo. Runs MobileNet-SSD for 30s, prints detections to terminal. No GUI. Used as the structural template for the IQA pipeline below.
- `headless_mobilenet_v2_REFERENCE.py` — old v2 script, kept frozen as a reference. Do NOT run on v3.
- `student_mobilenet_v3.blob` — **our distilled IQA model**, 4.96 MB, compiled for MyriadX RVC2 via the conversion pipeline documented below. Outputs a single scalar in [0, 1] representing scenic quality.
- `test_blob_load.py` — sanity-check script. Creates a minimal `Pipeline` with a `NeuralNetwork` node that loads the blob, confirms DepthAI accepts it. Requires camera connected (v3 limitation).
- `headless_iqa_v3.py` — **the actual Third Eye IQA inference script.** Camera → 320×240 BGR → NeuralNetwork (our blob) → prints scenic score per frame to terminal. Configurable FPS via `cam.setFps(n)`. Runs for 30 seconds by default.
- `README.md` — comprehensive setup doc, written for teammates who haven't worked with the hardware.

### Verified working demos
1. **headless_mobilenet_v3.py** — terminal-only MobileNet-SSD, works.
2. **EfficientViT** (`oak-examples/neural-networks/generic-example/`, model `luxonis/efficientvit:b1-224x224`) — browser visualizer at `http://thirdeye:8082`, transformer-based 1000-class ImageNet classifier. Works.
3. **Emotion recognition** (`oak-examples/neural-networks/face-detection/emotion-recognition/`) — browser visualizer, 2-stage face + emotion. Works.
4. **Third Eye IQA inference** (`headless_iqa_v3.py`) — terminal-only, custom distilled blob. Prints scenic score (0–1) per frame. Working end-to-end. Needs accuracy validation on known scenic vs non-scenic scenes (see Open Questions).

---

## Decisions Made and Why

### Decision: DepthAI v3 instead of v2
- v2 examples in `oak-examples` repo wouldn't run on a v2 install (API incompatible).
- v2 had no published wheel for 32-bit ARM, so we initially stuck with v2 because the Pi was 32-bit. Then later in the same session, decided to reflash to 64-bit specifically to enable v3.
- v3 is Luxonis's actively-maintained API. v2 is in maintenance mode. v3 also unlocks the Luxonis Model Zoo, which is how `EfficientViT` and other modern models are deployed.

### Decision: Pinned to `depthai==3.0.0` (not 3.5.0)
- The emotion-recognition example pins `depthai==3.0.0` and `depthai-nodes==0.3.4` in its requirements.txt.
- On `depthai-nodes==0.4.0`, the example fails because `GatherData.build()` signature changed.
- Other demos (headless_mobilenet_v3, EfficientViT, our IQA script) work fine on either version.
- If Abdullah wants to upgrade to 3.5.0 for new work, the emotion demo will break — that's documented in the README.

### Decision: 64-bit Raspberry Pi OS Lite (no desktop)
- Lite has no GUI. Saves ~3 GB and avoids running unused services. We're SSH-only anyway.

### Decision: No Raspberry Pi Connect
- Disabled during Imager customization. Adds cloud dependency, attack surface, and resource usage for no benefit since we're on the same LAN as the Pi or physically co-located when hiking.

### Decision: Browser-based visualizer for v3 demos, not X11 forwarding
- Teammate's note in the original README said "avoid GUI tests, not really our speed" — this referred to running GUI on the Pi over SSH. The v3 visualizer is different: it's an HTTP server on the Pi that streams to a browser on the Mac. Works cleanly over SSH without X11.

### Decision: Edit files locally on Mac, scp to Pi
- Abdullah's VS Code Remote-SSH kept crashing in past sessions. Decided to use plain SSH from Mac terminal + scp for file transfers + nano on the Pi only when necessary.

### Decision: Distill DINOv2-Small → MobileNetV3-Small for on-device IQA
- The team's best-performing IQA model is a DINOv2-Small backbone + MLP head. DINOv2 is a Vision Transformer (ViT), and while Luxonis's `EfficientViT` proves *some* ViTs run on RVC2, DINOv2-Small is larger and the Luxonis Model Zoo's DINOv3 is RVC4-only.
- Rather than accept very slow inference or upgrade to OAK4 hardware mid-project, we distilled the DINOv2 teacher into a MobileNetV3-Small student via knowledge distillation on the team's hiking dataset. MobileNetV3 is CNN-based, Myriad-friendly, and orders of magnitude smaller.
- Training notebook: `knowledge_distiller.ipynb` (Google Colab, uses the DINOv2 teacher checkpoint from `hiking_model_loading.ipynb`).
- The student's final classifier layer outputs a single logit; we wrap it with sigmoid before export so the blob directly produces a 0–1 scenic score.

### Decision: Bake preprocessing into the blob
- The blob embeds ImageNet normalization (`--mean_values`, `--scale_values`), BGR→RGB channel flip (`--reverse_input_channels`), and U8→FP16 conversion (`-ip U8`).
- **Why:** this lets the DepthAI pipeline feed raw camera frames (BGR, 0–255, uint8) directly into the NeuralNetwork node with no host-side preprocessing. Lower CPU load on the Pi, simpler pipeline code.
- **Critical implication for on-device code:** the Pi-side pipeline must NOT normalize frames itself. Feed raw BGR U8. If a future teammate adds `ImageManip` with normalization, the model will produce garbage because normalization will be applied twice.
- The sigmoid is also baked in (via a `StudentForExport` wrapper in the ONNX export cell), so the blob output is already 0–1. No post-processing needed.

### Decision: Hybrid conversion pipeline — local OpenVINO 2022.3 + blobconverter cloud
This was forced by a compatibility sandwich (documented in full under "The MobileNetV3 → MyriadX Blob Conversion Journey"). Short version:
- **ONNX → OpenVINO IR** is done locally in a Colab conda Python 3.10 env running OpenVINO 2022.3. This step handles the `HardSigmoid`/`HardSwish` decomposition correctly.
- **OpenVINO IR → .blob** is done via Luxonis's blobconverter cloud API, because `compile_tool` requires a physical Myriad device (which Colab doesn't have, but their servers do).
- Sending the pre-compiled IR (not the raw ONNX) to blobconverter bypasses the fact that blobconverter's server runs OpenVINO 2022.1, which doesn't support `HardSigmoid` either.

---

## Lessons Learned the Hard Way

1. **Typos in udev rules fail silently.** The original rule had `SUBSYSTM` (missing the E). `lsusb` showed the camera but DepthAI returned an empty device list. Took a careful read of the file to spot. Always verify with `cat`.

2. **NumPy 1.x vs 2.x incompatibility with OpenCV wheels.** On the v2 setup, OpenCV's wheel metadata claimed NumPy ≥ 2 was required, but the binary inside was actually compiled against NumPy 1.x. Forcing `pip install "numpy<2"` fixed it. Pip warned about a "dependency conflict" — that warning is a lie based on the wrong metadata.

3. **OpenCV needs system-level shared libs that pip can't install.** On 32-bit Raspbian, had to `sudo apt install libopenblas0 libatlas-base-dev` to get past `ImportError: libopenblas.so.0` and then `libcblas.so.3`.

4. **First-time SSH after a reflash will warn about changed host key.** This is expected. Fix: `ssh-keygen -R thirdeye` then re-SSH and accept the new key.

5. **DepthAI v3's `DetectionNetwork.build()` takes the Camera *node*, not its Output.** First v3 port attempt passed `cam_out` (a `Camera.Output`) and got a TypeError. Correct call passes `cam` (the `Camera` node) and lets `build()` request the output internally.

6. **Luxonis Model Zoo example requirements.txt files often pin OLD versions.** Don't blindly run `pip install -r requirements.txt` if you have newer libs working — read the file first. We almost downgraded a fresh depthai 3.5.0 install to 3.0.0 unnecessarily.

7. **DINOv3 from the Luxonis Model Zoo is RVC4-only.** Can't run on the OAK-D Lite (RVC2). This is the reason we ended up distilling into MobileNetV3 rather than deploying DINOv2/v3 directly.

8. **EfficientViT does run on RVC2** — confirmed at ~35 inf/sec FP16 from Luxonis's published benchmarks, and verified on-device. So *some* transformer-based vision models work on the OAK-D Lite, just not DINOv2/v3-scale ones.

9. **The Pi's USB ports matter.** Camera connected to a black USB 2.0 port shows up as `Bus 001` (USB 2.0, 480 Mbps cap). Use the **blue USB 3.0** ports for full bandwidth. Also requires a USB 3.0-rated cable, not a USB 2.0-only cable.

10. **`sudo` caches credentials for ~15 min.** Don't be alarmed when commands prefixed with `sudo` don't re-prompt for a password — they're using cached creds from a recent prior `sudo`.

11. **`transforms.Resize((H, W))` takes height first, width second.** The teammate's code in `hiking_model_loading.ipynb` (the teacher training notebook) had `transforms.Resize((320, 240))` with a comment saying "320x240 camera-native resolution." That comment uses the informal W×H convention, but the PyTorch API expects (H, W). The flipped value silently resizes landscape images (320 wide × 240 tall) into portrait (320 tall × 240 wide), distorting aspect ratios during both training and inference. **This needs to be fixed in the teammate's notebook** — it affects teacher model evaluation numbers and any re-exports of the teacher. Correct call: `transforms.Resize((240, 320))`. Same fix applied to our student notebook and ONNX export dummy input.

12. **DepthAI v3 can't instantiate `Pipeline()` without a connected device.** In v2, you could build pipeline objects offline for testing. In v3, even a minimal `pipeline = dai.Pipeline()` with no nodes throws `RuntimeError: No available devices` if the camera isn't plugged in. Implication: sanity-check scripts that load blobs cannot be run off-device.

13. **The MobileNetV3 → MyriadX Blob Conversion Journey.** This took most of a session to untangle. Documenting the whole thing so nobody has to re-derive it:

    **Goal:** Convert a PyTorch MobileNetV3-Small (our distilled IQA student) into a `.blob` that runs on the OAK-D Lite's MyriadX VPU.

    **Obvious path (doesn't work):** PyTorch → ONNX → blobconverter → `.blob`.

    **What goes wrong, in order:**

    - **Issue 1: MobileNetV3 contains `HardSigmoid` and `HardSwish` ops** (used throughout its Squeeze-and-Excite blocks). The MyriadX VPU compiler does not implement `HardSigmoid`. The Luxonis "supported operations" docs *list* `HardSigmoid-1` as supported, but that's the OpenVINO IR op — the Myriad hardware plugin itself does not support it. Any toolchain that tries to compile MobileNetV3 to a Myriad blob without pre-processing will fail at the Myriad compile step with `unsupported layer type "HardSigmoid"`.

    - **Fix:** Replace `nn.Hardsigmoid` and `nn.Hardswish` modules in the model with mathematically equivalent versions built from `ReLU6` (which the Myriad *does* support), **before** ONNX export. Formulas:
      - `HardSigmoid(x) = ReLU6(x + 3) / 6`
      - `HardSwish(x)   = x × ReLU6(x + 3) / 6`
      - Recursive helper in the export cell walks the model tree and swaps these in place.

    - **Issue 2: PyTorch 2.6+ ignores `opset_version < 18`.** The new dynamo-based ONNX exporter in recent PyTorch silently overrides the requested opset. Warning in output says "Setting ONNX exporter to use operator set version 18 because the requested opset_version 12 is a lower version than we have implementations for." OpenVINO 2022.x only supports up to opset 15, so the exported ONNX is unusable.

    - **Fix:** Pass `dynamo=False` to `torch.onnx.export` to force the legacy exporter, which honors `opset_version=12`.

    - **Issue 3: `do_constant_folding=False` breaks SE block shape inference.** With constant folding disabled, the SE blocks' global-average-pool output shapes are left dynamic, causing OpenVINO's Model Optimizer to throw a channel-count mismatch (`Data batch channel count (1) does not match filter input channel count (16)`).

    - **Fix:** `do_constant_folding=True` in the export call.

    - **Issue 4: blobconverter cloud API runs OpenVINO 2022.1, which lacks `HardSigmoid` on Myriad.** Even with the ReLU6 rewrite, something in the ONNX→IR conversion on blobconverter's server was leaving `HardSigmoid` artifacts. More importantly, the 2022.1 toolchain itself has known issues that OpenVINO 2022.3 fixed.

    - **Fix approach:** Do the ONNX → IR step locally in OpenVINO 2022.3, then send the pre-compiled IR to blobconverter (which will only run the Myriad compile step).

    - **Issue 5: OpenVINO 2022.3 requires Python 3.7–3.10; Colab runs Python 3.12.** `pip install openvino-dev==2022.3` fails on Colab because it can't build `numpy<=1.23.4` on Python 3.12.

    - **Fix:** Install Miniforge (a lightweight conda) and create a Python 3.10 environment just for OpenVINO. Run `openvino.tools.mo` from that env. Everything else in the notebook stays on Colab's Python 3.12.

    - **Issue 6: OpenVINO's `compile_tool` (for the IR → blob step) requires a physical Myriad device connected.** Colab has no Myriad. Tried downloading the OpenVINO 2022.3.1 LTS archive and running `compile_tool` directly; it complains `Failed to allocate graph: MYRIAD device is not opened.`

    - **Fix:** Don't try to compile locally. Send the IR to blobconverter's cloud API, which has Myriad hardware. Crucially, use `blobconverter.from_openvino(xml=..., bin=...)` to submit the already-converted IR, **not** `blobconverter.from_onnx(...)`. The former only triggers the Myriad compile; the latter re-runs Model Optimizer with OpenVINO 2022.1 and hits Issue 4.

    - **Issue 7: Shared library hell when attempting local `compile_tool`.** Before we realized `compile_tool` couldn't work without a Myriad, we chased missing libraries: `libpugixml.so.1` (fixed with `apt install libpugixml1v5`), `libtbb.so.2` (fixed with `apt install libtbb2`), and finally the `MYRIAD device is not opened` error that made us pivot. Dead-end path; documenting it so nobody burns time re-discovering it.

    - **Issue 8: ModelConverter (the Luxonis-recommended modern tool) requires Docker.** Colab doesn't have Docker. Another dead end.

    **Final working pipeline (in the `knowledge_distiller.ipynb` notebook):**
    1. PyTorch model with `HardSigmoid`/`HardSwish` replaced by ReLU6 equivalents
    2. Wrapped in `StudentForExport` to bake in final sigmoid
    3. `torch.onnx.export(..., opset_version=12, do_constant_folding=True, dynamo=False)`
    4. `onnxsim` to clean the graph
    5. Miniforge conda env (Python 3.10) + `openvino-dev==2022.3.0` → `mo` → `.xml`+`.bin` IR
       - `mo` flags include `--mean_values`, `--scale_values`, `--reverse_input_channels`, `--compress_to_fp16`
    6. `blobconverter.from_openvino(xml=..., bin=...)` with `shaves=6`, `version="2022.1"`, `data_type="FP16"` → `.blob`

    **Input/output contract of the resulting blob:**
    - Input: `[1, 3, 240, 320]` BGR U8 (camera-native, no normalization needed on host)
    - Output: single float in [0, 1] — scenic quality score

---

## Open Questions / Things to Address Soon

1. ~~**Will custom DINOv2 + MLP heads even run on RVC2?**~~ **RESOLVED via distillation.** Deployed a MobileNetV3-Small student distilled from the DINOv2 teacher. 4.96 MB blob runs on the OAK-D Lite. Model conversion pipeline is fully documented. Remaining work is validating on-device *accuracy*, not *feasibility*.

2. **Validate IQA accuracy on-device.** Point the camera at known scenic vs non-scenic scenes (outdoors through a window, plants, vs keyboards/walls/floor) and verify scores discriminate properly. Three possible outcomes:
   - Clean separation (e.g., ~0.1–0.3 for non-scenic, ~0.6–0.9 for scenic) → model works, proceed.
   - Scores all similar → model isn't discriminating; debug. Could be overfitting in distillation, could be a preprocessing mismatch between training and deployment.
   - Scores inverted → sigmoid sign flip or mean/scale values wrong; fixable.

3. **Image-saving alongside inference.** When the scenic score exceeds a threshold, save the current frame to disk with a timestamp. This is the actual hiking-camera behavior. Depends on #2 first — no point saving frames based on a miscalibrated model.

4. **Fix teammate's `transforms.Resize((320, 240))` bug in `hiking_model_loading.ipynb`.** Should be `(240, 320)`. Affects teacher model evaluation numbers. Flag this to ddecosmo.

5. **Battery-powered runtime test.** How long can the Pi + camera + power bank actually run? Needs to be at least a few hours for a real hike.

6. **CMU campus WiFi setup.** Manual `wpa_supplicant` config required (WPA2-Enterprise, not supported by Raspberry Pi Imager's quick-config).

7. **Benchmark FPS / per-frame latency on the OAK-D Lite.** Informally, inference is fast; we throttled camera FPS down to 5 for readability. Worth a proper measurement.

8. **Revisit the multi-head architecture when distillation strategy is finalized.** The team proposed a shared-backbone + multiple-MLP-heads design. If that gets adopted, the distillation will need to be redone — either one student per head, or one student backbone + multiple heads (mirroring the teacher).

---

## How to Help Abdullah

- **He prefers step-by-step, one thing at a time, no walls of text.** He'll explicitly say "let's take it slow." Honor that.
- **He's an EE/hardware-minded person, comfortable with terminals, but will ask basic questions about commands or tools he hasn't used recently.** Treat all questions as legitimate; never condescend.
- **He cares about understanding *why*, not just running commands.** Briefly explain what each step does and what we're checking for. He's catching mistakes when explanations don't match what's happening — that's a good thing.
- **He wants to verify before assuming.** When in doubt, run a check command rather than relying on memory.
- **He's editing on his Mac, transferring with scp.** Don't push him toward editing in nano unless it's a one-line tweak.
- **Update the README at the end of major sessions.** It's the team's source of truth; teammates may try to reproduce his work without him present.
- **Keep an eye on time/scope creep.** If a side quest is going to eat hours, name the cost up front and ask if it's worth it.

---

## Quick Reference Commands

```bash
# SSH in
ssh pi@thirdeye

# Activate venv
cd ~/thirdeye_project && source .venv/bin/activate

# Verify camera
lsusb
python -c "import depthai as dai; print(dai.Device.getAllAvailableDevices())"

# Run terminal demos
python ~/thirdeye_project/headless_mobilenet_v3.py     # MobileNet-SSD (stock)
python ~/thirdeye_project/headless_iqa_v3.py           # Our IQA model (custom blob)

# Sanity-check the custom blob loads (camera must be connected)
python ~/thirdeye_project/test_blob_load.py

# Run browser-visualizer demos
cd ~/thirdeye_project/oak-examples/neural-networks/generic-example
python main.py --model luxonis/efficientvit:b1-224x224
# Then on Mac browser: http://thirdeye:8082

# Clean shutdown
sudo shutdown now
```

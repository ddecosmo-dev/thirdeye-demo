"""Frame source abstraction for Luxonis Oak camera."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import settings
from .utils import configure_logging

try:
    import depthai as dai
except ImportError:  # pragma: no cover - depends on device environment
    dai = None


@dataclass(frozen=True)
class Frame:
    data: bytes
    filename_hint: str
    ext: str | None = None
    inference: dict[str, Any] | None = None


class FrameSource:
    def start(self) -> None:
        return None

    def next_frame(self) -> Frame | None:
        raise NotImplementedError

    def stop(self) -> None:
        return None


class MockFrameSource(FrameSource):
class OakFrameSource(FrameSource):
    def __init__(self) -> None:
        self._logger = configure_logging("oak-frame-source")
        self._started = False
        self._device: Any | None = None
        self._jpeg_q: Any | None = None
        self._nn_q: Any | None = None
        self._prefilter_q: Any | None = None
        self._sequence = 0

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        if not settings.oak_connected:
            self._logger.info("oak not connected, waiting for hardware")
            return
        if dai is None:
            raise RuntimeError("DepthAI is not installed. Install 'depthai' on the device.")
        if not settings.blob_path:
            raise RuntimeError("BLOB_PATH is required when OAK_CONNECTED=true")

        pipeline = dai.Pipeline()

        cam = pipeline.createColorCamera()
        cam.setFps(settings.capture_fps)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setPreviewSize(settings.downsample_width, settings.downsample_height)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

        manip = pipeline.createImageManip()
        manip.initialConfig.setResize(settings.downsample_width, settings.downsample_height)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)
        cam.preview.link(manip.inputImage)

        if settings.normalize_inputs:
            applied = False
            if hasattr(manip.initialConfig, "setMean") and hasattr(manip.initialConfig, "setStd"):
                manip.initialConfig.setMean(list(settings.imagenet_mean))
                manip.initialConfig.setStd(list(settings.imagenet_std))
                applied = True
            if not applied:
                self._logger.warning("ImageNet normalization not supported in this DepthAI version")

        nn = pipeline.createNeuralNetwork()
        nn.setBlobPath(settings.blob_path)
        manip.out.link(nn.input)

        xout_nn = pipeline.createXLinkOut()
        xout_nn.setStreamName("nn")
        nn.out.link(xout_nn.input)

        if settings.prefilter_blob_path:
            prefilter = pipeline.createNeuralNetwork()
            prefilter.setBlobPath(settings.prefilter_blob_path)
            manip.out.link(prefilter.input)
            xout_prefilter = pipeline.createXLinkOut()
            xout_prefilter.setStreamName("prefilter")
            prefilter.out.link(xout_prefilter.input)
        else:
            xout_prefilter = None

        encoder = pipeline.createVideoEncoder()
        encoder.setDefaultProfilePreset(settings.capture_fps, dai.VideoEncoderProperties.Profile.MJPEG)
        cam.video.link(encoder.input)

        xout_jpeg = pipeline.createXLinkOut()
        xout_jpeg.setStreamName("jpeg")
        encoder.bitstream.link(xout_jpeg.input)

        self._device = dai.Device(pipeline)
        self._jpeg_q = self._device.getOutputQueue("jpeg", maxSize=4, blocking=True)
        self._nn_q = self._device.getOutputQueue("nn", maxSize=4, blocking=True)
        if xout_prefilter:
            self._prefilter_q = self._device.getOutputQueue("prefilter", maxSize=4, blocking=False)
        self._logger.info("oak pipeline started blob=%s", settings.blob_path)

    def next_frame(self) -> Frame | None:
        if not settings.oak_connected:
            return None
        if self._jpeg_q is None or self._nn_q is None:
            return None

        jpeg_packet = self._jpeg_q.get()
        nn_packet = self._nn_q.get()

        prefilter_score = None
        prefilter_passed = True
        if self._prefilter_q is not None:
            prefilter_packet = self._prefilter_q.tryGet()
            if prefilter_packet is not None:
                prefilter_raw = prefilter_packet.getFirstLayerFp16()
                prefilter_score = float(prefilter_raw[0]) if prefilter_raw else 0.0
                prefilter_passed = prefilter_score >= settings.prefilter_threshold

        nn_raw = nn_packet.getFirstLayerFp16()
        model_score = float(nn_raw[0]) if nn_raw else 0.0
        model_passed = model_score >= settings.model_threshold

        if prefilter_passed:
            tag = "model_passed" if model_passed else "model_failed"
        else:
            tag = "prefilter_failed"

        self._sequence += 1
        filename_hint = f"frame_{self._sequence:06d}.jpg"
        inference = {
            "prefilter_score": prefilter_score,
            "prefilter_passed": prefilter_passed,
            "model_score": model_score,
            "model_passed": model_passed,
            "tag": tag,
            "blob_path": settings.blob_path,
            "prefilter_blob_path": settings.prefilter_blob_path,
            "normalize_inputs": settings.normalize_inputs,
            "inference_source": "oak",
        }
        return Frame(data=bytes(jpeg_packet.getData()), filename_hint=filename_hint, ext=".jpg", inference=inference)

    def stop(self) -> None:
        if not self._started:
            return
        if self._device is not None:
            self._device.close()
            self._device = None
        self._logger.info("oak frame source stopped")
        self._started = False


def build_frame_source() -> FrameSource:
    if settings.oak_connected:
        return OakFrameSource()
    raise RuntimeError("No frame source configured. Set OAK_CONNECTED=true.")

"""Settings and resource limits for the edge services."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    data_dir: str = os.getenv("DATA_DIR", "/data")
    runs_dirname: str = "runs"
    temp_dirname: str = "temp"

    coordinator_host: str = os.getenv("COORDINATOR_HOST", "0.0.0.0")
    coordinator_port: int = int(os.getenv("COORDINATOR_PORT", "8081"))
    processor_host: str = os.getenv("PROCESSOR_HOST", "0.0.0.0")
    processor_port: int = int(os.getenv("PROCESSOR_PORT", "8082"))
    processor_base_url: str = os.getenv("PROCESSOR_BASE_URL", "http://127.0.0.1:8082")

    capture_fps: int = int(os.getenv("CAPTURE_FPS", "1"))
    max_image_bytes: int = int(os.getenv("MAX_IMAGE_BYTES", "5000000"))
    min_free_disk_bytes: int = int(os.getenv("MIN_FREE_DISK_BYTES", "200000000"))

    downsample_width: int = int(os.getenv("DOWNSAMPLE_WIDTH", "320"))
    downsample_height: int = int(os.getenv("DOWNSAMPLE_HEIGHT", "240"))
    prefilter_threshold: float = float(os.getenv("PREFILTER_THRESHOLD", "0.25"))
    model_threshold: float = float(os.getenv("MODEL_THRESHOLD", "0.5"))
    normalize_inputs: bool = os.getenv("NORMALIZE_INPUTS", "false").lower() == "true"

    blob_path: str | None = os.getenv("BLOB_PATH")
    prefilter_blob_path: str | None = os.getenv("PREFILTER_BLOB_PATH")
    oak_connected: bool = os.getenv("OAK_CONNECTED", "false").lower() == "true"

    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225)


settings = Settings()

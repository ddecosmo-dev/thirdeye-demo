"""Settings and resource limits for the edge service."""

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    data_dir: str = os.getenv("DATA_DIR", "/data")
    runs_dirname: str = "runs"

    max_queue_frames: int = int(os.getenv("MAX_QUEUE_FRAMES", "20"))
    max_image_bytes: int = int(os.getenv("MAX_IMAGE_BYTES", "5000000"))
    max_run_images: int = int(os.getenv("MAX_RUN_IMAGES", "1000"))
    max_run_bytes: int = int(os.getenv("MAX_RUN_BYTES", "2000000000"))
    min_free_disk_bytes: int = int(os.getenv("MIN_FREE_DISK_BYTES", "1000000000"))

    capture_fps_limit: int = int(os.getenv("CAPTURE_FPS_LIMIT", "5"))

    upload_max_bytes: int = int(os.getenv("UPLOAD_MAX_BYTES", "200000000"))
    upload_timeout_seconds: int = int(os.getenv("UPLOAD_TIMEOUT_SECONDS", "30"))

    cloud_ingest_url: str = os.getenv("CLOUD_INGEST_URL", "http://localhost:8080/ingest")
    mock_image_dir: str | None = os.getenv("MOCK_IMAGE_DIR")


settings = Settings()

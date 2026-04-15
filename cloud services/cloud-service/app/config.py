"""Environment-driven settings for upload limits and data paths."""

from pydantic import BaseModel
import os


class Settings(BaseModel):
    data_dir: str = os.getenv("DATA_DIR", "/data")
    # TODO: Move limits into a typed config file for easier deployment changes.
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", "52428800"))
    max_uncompressed_bytes: int = int(os.getenv("MAX_UNCOMPRESSED_BYTES", "104857600"))
    max_files_per_zip: int = int(os.getenv("MAX_FILES_PER_ZIP", "500"))
    edge_base_url: str | None = os.getenv("EDGE_BASE_URL")


settings = Settings()

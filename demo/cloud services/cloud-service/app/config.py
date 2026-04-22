"""Environment-driven settings for upload limits and data paths."""

from pydantic import BaseModel
import os


class Settings(BaseModel):
    data_dir: str = os.getenv(
        "DATA_DIR",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")),
    )
    # TODO: Move limits into a typed config file for easier deployment changes.
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", "2147483648"))  # 2GB
    max_uncompressed_bytes: int = int(os.getenv("MAX_UNCOMPRESSED_BYTES", "2147483648"))  # 2GB
    max_files_per_zip: int = int(os.getenv("MAX_FILES_PER_ZIP", "5000"))
    edge_base_url: str | None = os.getenv("EDGE_BASE_URL")


settings = Settings()

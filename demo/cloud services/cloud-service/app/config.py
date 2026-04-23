"""Environment-driven settings for upload limits and data paths."""

from pathlib import Path
from pydantic import BaseModel
import os


DEFAULT_REPO_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


class Settings(BaseModel):
    data_dir: str = str(Path(os.getenv("DATA_DIR", DEFAULT_REPO_DATA_DIR)))
    # TODO: Move limits into a typed config file for easier deployment changes.
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", "2147483648"))  # 2GB
    max_uncompressed_bytes: int = int(os.getenv("MAX_UNCOMPRESSED_BYTES", "2147483648"))  # 2GB
    max_files_per_zip: int = int(os.getenv("MAX_FILES_PER_ZIP", "5000"))
    edge_base_url: str | None = os.getenv("EDGE_BASE_URL")


settings = Settings()

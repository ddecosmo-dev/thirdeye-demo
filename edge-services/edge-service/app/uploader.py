"""Archive creation and upload helper for completed runs."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from typing import Any

import httpx

from .config import settings
from .storage import images_dir


def _sha256_file(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _archive_path(run_id: str) -> str:
    return os.path.join(images_dir(run_id), "..", "upload.zip")


def create_archive(run_id: str, max_bytes: int) -> tuple[str, int, int]:
    image_root = images_dir(run_id)
    archive_path = _archive_path(run_id)

    file_count = 0
    total_bytes = 0

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(image_root):
            for name in files:
                full_path = os.path.join(root, name)
                rel_path = os.path.relpath(full_path, image_root)
                size = os.path.getsize(full_path)
                total_bytes += size
                file_count += 1

                if total_bytes > max_bytes:
                    raise ValueError("Archive exceeds max allowed bytes")

                zf.write(full_path, rel_path)

    return archive_path, file_count, total_bytes


def upload_archive(run_id: str, archive_path: str, metadata: dict[str, Any]) -> dict[str, Any]:
    checksum = _sha256_file(archive_path)

    with open(archive_path, "rb") as handle:
        files = {"file": ("images.zip", handle, "application/zip")}
        data = {
            "run_id": run_id,
            "metadata_json": json.dumps(metadata),
        }
        # TODO: Add authentication header or shared secret.
        with httpx.Client(timeout=settings.upload_timeout_seconds) as client:
            response = client.post(settings.cloud_ingest_url, data=data, files=files)

    if response.status_code >= 400:
        raise ValueError(f"Cloud ingest failed: {response.text}")

    return {"checksum_sha256": checksum, "status_code": response.status_code}

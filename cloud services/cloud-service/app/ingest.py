from __future__ import annotations

import io
import json
import os
import zipfile
from typing import Any

from fastapi import HTTPException, UploadFile

from .config import settings
from .storage import images_dir, raw_dir, read_metadata, write_metadata
from .utils import ensure_dir


class ZipInspectionResult:
    def __init__(self, file_count: int, total_uncompressed: int) -> None:
        self.file_count = file_count
        self.total_uncompressed = total_uncompressed


def _safe_zip_members(zf: zipfile.ZipFile) -> ZipInspectionResult:
    file_count = 0
    total_uncompressed = 0

    for info in zf.infolist():
        if info.is_dir():
            continue
        file_count += 1
        total_uncompressed += info.file_size

        if file_count > settings.max_files_per_zip:
            raise HTTPException(status_code=413, detail="Zip contains too many files")
        if total_uncompressed > settings.max_uncompressed_bytes:
            raise HTTPException(status_code=413, detail="Zip is too large when expanded")

        name = info.filename
        if name.startswith("/") or name.startswith("\\"):
            raise HTTPException(status_code=400, detail="Zip contains absolute paths")
        if ".." in name.replace("\\", "/").split("/"):
            raise HTTPException(status_code=400, detail="Zip contains unsafe paths")

    return ZipInspectionResult(file_count, total_uncompressed)


def _write_upload_to_disk(upload: UploadFile, destination: str) -> int:
    ensure_dir(os.path.dirname(destination))
    size = 0
    with open(destination, "wb") as handle:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > settings.max_upload_bytes:
                raise HTTPException(status_code=413, detail="Upload exceeds size limit")
            handle.write(chunk)
    return size


def ingest_zip(run_id: str, upload: UploadFile, metadata_json: str | None) -> dict[str, Any]:
    if not upload.filename or not upload.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip uploads are supported")

    raw_path = os.path.join(raw_dir(run_id), "upload.zip")
    _write_upload_to_disk(upload, raw_path)

    try:
        with zipfile.ZipFile(raw_path, "r") as zf:
            inspection = _safe_zip_members(zf)
            target_dir = images_dir(run_id)
            zf.extractall(target_dir)
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Invalid zip file") from exc

    meta = read_metadata(run_id) or {}
    meta.setdefault("ingest", {})
    meta["ingest"].update(
        {
            "file_count": inspection.file_count,
            "total_uncompressed_bytes": inspection.total_uncompressed,
        }
    )

    if metadata_json:
        try:
            metadata_payload = json.loads(metadata_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="metadata_json is not valid JSON") from exc
        meta["ingest"]["metadata"] = metadata_payload

    write_metadata(run_id, meta)

    return {
        "file_count": inspection.file_count,
        "total_uncompressed_bytes": inspection.total_uncompressed,
        "raw_path": raw_path,
    }

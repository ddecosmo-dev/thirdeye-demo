"""Image ingest helpers with zip validation and size limits."""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import zipfile
from typing import Any

from fastapi import HTTPException, UploadFile

from .config import settings
from .storage import images_dir, raw_dir, read_metadata, write_metadata
from .utils import ensure_dir

logger = logging.getLogger(__name__)


# TODO: Extend to include per-file checksums if needed for audit.
class ZipInspectionResult:
    def __init__(self, file_count: int, total_uncompressed: int) -> None:
        self.file_count = file_count
        self.total_uncompressed = total_uncompressed


def _safe_zip_members(zf: zipfile.ZipFile) -> ZipInspectionResult:
    file_count = 0
    total_uncompressed = 0

    # TODO: Enforce allowed file extensions (e.g., .jpg, .png).
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
    logger.info(f"📥 Starting upload to {destination}")
    with open(destination, "wb") as handle:
        # TODO: Add per-request rate limiting or throttling if needed.
        chunk_count = 0
        while True:
            chunk = upload.file.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            size += len(chunk)
            chunk_count += 1
            if chunk_count % 100 == 0:  # Log every 100MB
                size_mb = size / (1024 * 1024)
                logger.info(f"📥 Uploaded {size_mb:.1f}MB so far...")
            if size > settings.max_upload_bytes:
                raise HTTPException(status_code=413, detail="Upload exceeds size limit")
            handle.write(chunk)
    size_mb = size / (1024 * 1024)
    size_gb = size / (1024 * 1024 * 1024)
    logger.info(f"✅ Upload complete: {size_gb:.2f}GB ({size_mb:.0f}MB)")
    return size


def ingest_zip(run_id: str, upload: UploadFile, metadata_json: str | None) -> dict[str, Any]:
    if not upload.filename or not upload.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip uploads are supported")

    raw_path = os.path.join(raw_dir(run_id), "upload.zip")
    logger.info(f"📦 Ingesting zip file: {upload.filename}")
    _write_upload_to_disk(upload, raw_path)

    try:
        logger.info("🔍 Inspecting zip file contents...")
        with zipfile.ZipFile(raw_path, "r") as zf:
            inspection = _safe_zip_members(zf)
            logger.info(f"📋 Zip contains {inspection.file_count} files ({inspection.total_uncompressed / (1024**3):.2f}GB uncompressed)")
            
            target_dir = images_dir(run_id)
            
            # Extract all files to a temp location first
            logger.info("🗜️  Extracting zip to temporary location...")
            temp_extract = os.path.join(raw_dir(run_id), "extracted")
            ensure_dir(temp_extract)
            zf.extractall(temp_extract)
            logger.info("✅ Zip extraction complete")
            
            # Now find all image files recursively and copy them to target_dir
            logger.info("🖼️  Copying images to final location...")
            image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic"}
            extracted_count = 0
            
            for root, dirs, files in os.walk(temp_extract):
                for filename in files:
                    if os.path.splitext(filename.lower())[1] in image_extensions:
                        src = os.path.join(root, filename)
                        dst = os.path.join(target_dir, filename)
                        # Handle duplicate filenames by appending a counter
                        if os.path.exists(dst):
                            base, ext = os.path.splitext(filename)
                            counter = 1
                            while os.path.exists(os.path.join(target_dir, f"{base}_{counter}{ext}")):
                                counter += 1
                            dst = os.path.join(target_dir, f"{base}_{counter}{ext}")
                        shutil.copy2(src, dst)
                        extracted_count += 1
                        if extracted_count % 100 == 0:
                            logger.info(f"📁 Processed {extracted_count} images...")
            
            logger.info(f"✅ Image copy complete: {extracted_count} images extracted to {target_dir}")
                        
    except zipfile.BadZipFile as exc:
        logger.error(f"❌ Invalid zip file: {exc}", exc_info=True)
        raise HTTPException(status_code=400, detail="Invalid zip file") from exc

    meta = read_metadata(run_id) or {}
    meta.setdefault("ingest", {})
    meta["ingest"].update(
        {
            "file_count": inspection.file_count,
            "total_uncompressed_bytes": inspection.total_uncompressed,
            "extracted_images": extracted_count,
        }
    )

    if metadata_json:
        try:
            metadata_payload = json.loads(metadata_json)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="metadata_json is not valid JSON") from exc
        # TODO: Validate metadata payload schema once defined.
        meta["ingest"]["metadata"] = metadata_payload

    write_metadata(run_id, meta)

    logger.info(f"🎉 Ingest complete for run {run_id}: {extracted_count} images ready for inference")
    return {
        "file_count": inspection.file_count,
        "total_uncompressed_bytes": inspection.total_uncompressed,
        "extracted_images": extracted_count,
        "raw_path": raw_path,
    }

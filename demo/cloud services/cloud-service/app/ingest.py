"""Image ingest helpers with zip validation and size limits."""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

from fastapi import HTTPException, UploadFile

from .config import settings
from .storage import edge_results_path, images_dir, raw_dir, read_metadata, run_dir, write_metadata
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


def _is_valid_edge_results_json(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if "entries" not in value or not isinstance(value["entries"], list):
        return False
    required_keys = {
        "frame_index",
        "timestamp",
        "prefilter_passed",
        "prefilter_reason",
        "mean_intensity",
        "blur_variance",
        "scenic_score",
    }

    for entry in value["entries"]:
        if not isinstance(entry, dict):
            return False
        if not required_keys.issubset(entry.keys()):
            return False
        if not isinstance(entry["prefilter_passed"], bool):
            return False
    return True


def _find_edge_results_json_in_zip(zip_path: str) -> tuple[str | None, dict[str, Any] | None]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if not info.filename.lower().endswith(".json"):
                continue

            try:
                with zf.open(info, "r") as handle:
                    content = handle.read().decode("utf-8")
                    candidate = json.loads(content)
            except Exception:
                continue

            if _is_valid_edge_results_json(candidate):
                return info.filename, candidate
    return None, None


def _write_edge_results(run_id: str, data: dict[str, Any]) -> str:
    path = edge_results_path(run_id)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return path


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

    edge_results_file: str | None = None
    edge_results_source: str = "simulated"

    try:
        logger.info("🔍 Inspecting zip file contents...")
        with zipfile.ZipFile(raw_path, "r") as zf:
            inspection = _safe_zip_members(zf)
            logger.info(f"📋 Zip contains {inspection.file_count} files ({inspection.total_uncompressed / (1024**3):.2f}GB uncompressed)")

            edge_filename, edge_data = _find_edge_results_json_in_zip(raw_path)
            if edge_filename and edge_data is not None:
                logger.info(f"🟢 Found existing edge JSON in upload: {edge_filename}")
                edge_results_file = _write_edge_results(run_id, edge_data)
                edge_results_source = "uploaded"

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
            image_paths = []

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
                        image_paths.append(Path(src))
                        if extracted_count % 100 == 0:
                            logger.info(f"📁 Processed {extracted_count} images...")

            logger.info(f"✅ Image copy complete: {extracted_count} images extracted to {target_dir}")

            if edge_results_file is None and extracted_count > 0:
                from .inference.edge_simulator import simulate_edge_inference_from_paths

                logger.info("🧪 Running edge simulator because no edge JSON was included in the upload.")
                run_root = run_dir(run_id)
                edge_path, _ = simulate_edge_inference_from_paths(
                    image_paths=image_paths,
                    output_root=Path(run_root).parent,
                    run_id=run_id,
                    results_filename="edge_results.json",
                )
                edge_results_file = str(edge_path)
                edge_results_source = "simulated"
            elif edge_results_file is None:
                edge_results_source = None
                logger.warning("⚠️ No images were extracted, so edge simulation was skipped.")

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
    meta["edge"] = {
        "results_path": edge_results_file,
        "source": edge_results_source,
    }

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
        "edge_results_path": edge_results_file,
        "edge_results_source": edge_results_source,
    }

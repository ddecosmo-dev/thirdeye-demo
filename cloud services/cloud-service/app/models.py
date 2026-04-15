"""API request/response models for cycle control and ingest."""

from pydantic import BaseModel, Field
from typing import Any


class StartCycleRequest(BaseModel):
    label: str | None = Field(default=None, max_length=64)
    duration_seconds: int | None = Field(default=None, ge=1, le=86400)
    # TODO: Replace with a structured model once edge config is finalized.
    config: dict[str, Any] | None = None


class StartCycleResponse(BaseModel):
    run_id: str
    label: str
    status: str


class CycleActionResponse(BaseModel):
    run_id: str
    status: str


class IngestResponse(BaseModel):
    run_id: str
    status: str
    file_count: int
    total_uncompressed_bytes: int
    checksum_sha256: str

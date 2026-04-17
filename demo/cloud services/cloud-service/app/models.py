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


class InferenceRequest(BaseModel):
    """Request to run inference on uploaded images."""
    run_id: str = Field(..., description="Run ID containing images to process")
    epsilon: float = Field(default=0.12, ge=0.0, le=1.0, description="HDBSCAN clustering epsilon")
    min_cluster_size: int = Field(default=2, ge=2, le=100, description="Minimum cluster size for HDBSCAN")
    ignore_object: bool = Field(default=False, description="Ignore object scores in aggregation")
    device: str | None = Field(default=None, description="Device for inference (cpu, cuda, mps) - defaults to auto-detected")


class InferenceResponse(BaseModel):
    """Response after inference completes."""
    run_id: str
    status: str
    message: str
    image_count: int
    champion_count: int
    results_path: str


class TSNERequest(BaseModel):
    """Request to compute t-SNE from saved embeddings."""
    run_id: str = Field(..., description="Run ID with saved inference results")
    perplexity: int = Field(default=30, ge=5, le=100, description="t-SNE perplexity parameter")
    seed: int = Field(default=42, description="Random seed for reproducibility")


class TSNEResponse(BaseModel):
    """Response with computed t-SNE coordinates."""
    run_id: str
    status: str
    image_count: int
    perplexity: int
    tsne_coordinates: list[dict[str, float]]  # [{x, y, index, filename}, ...]


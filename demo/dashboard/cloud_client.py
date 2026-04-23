"""Cloud service client for the dashboard."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class CloudServiceClient:
    """Client for interacting with the ThirdEye Cloud Service."""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=600.0)

    async def health(self) -> bool:
        """Check if cloud service is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Cloud service health check failed: {e}")
            return False

    async def get_runs(self) -> dict[str, Any]:
        """List available runs from the cloud service."""
        try:
            response = await self.client.get(f"{self.base_url}/runs")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get runs: {e}")
            raise

    async def ingest_zip(self, run_id: str, zip_path: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Upload a zip file to the cloud service."""
        try:
            with open(zip_path, "rb") as f:
                files = {"file": f}
                data = {"run_id": run_id}
                if metadata:
                    data["metadata_json"] = json.dumps(metadata)
                response = await self.client.post(f"{self.base_url}/ingest", files=files, data=data)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to ingest zip: {e}")
            raise

    async def run_inference(
        self,
        run_id: str,
        epsilon: float = 0.12,
        min_cluster_size: int = 2,
        ignore_object: bool = False,
        device: str = "cpu",
    ) -> dict[str, Any]:
        """Trigger inference on uploaded images."""
        try:
            payload = {
                "run_id": run_id,
                "epsilon": epsilon,
                "min_cluster_size": min_cluster_size,
                "ignore_object": ignore_object,
                "device": device,
            }
            response = await self.client.post(f"{self.base_url}/infer", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to run inference: {e}")
            raise

    async def get_status(self, run_id: str) -> dict[str, Any]:
        """Get the status of a run."""
        try:
            response = await self.client.get(f"{self.base_url}/runs/{run_id}")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            raise

    async def get_progress(self, run_id: str) -> dict[str, Any]:
        """Get real-time inference progress from cloud service."""
        try:
            url = f"{self.base_url}/progress/{run_id}"
            logger.info(f"📡 Requesting progress from: {url}")
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()
            logger.info(f"📡 Got progress: {data}")
            return data
        except Exception as e:
            logger.warning(f"❌ Failed to get progress from {self.base_url}/progress/{run_id}: {e}")
            # Return default progress if cloud service doesn't have it yet
            return {
                "stage": "pending",
                "images_done": 0,
                "images_total": 0,
                "percent_complete": 0,
            }

    async def get_results(self, run_id: str) -> dict[str, Any]:
        """Get inference results for a run."""
        try:
            response = await self.client.get(f"{self.base_url}/runs/{run_id}/results")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        except Exception as e:
            logger.error(f"Failed to get results: {e}")
            raise

    async def get_image(self, run_id: str, filename: str) -> httpx.Response:
        """Retrieve an image file for a run from the cloud service."""
        try:
            response = await self.client.get(f"{self.base_url}/runs/{run_id}/images/{filename}")
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch image {filename} for run {run_id}: {e}")
            raise

    async def wait_for_results(self, run_id: str, max_wait_seconds: float = 3600, poll_interval: float = 5) -> dict[str, Any] | None:
        """Poll for results until they become available."""
        elapsed = 0
        while elapsed < max_wait_seconds:
            try:
                results = await self.get_results(run_id)
                if results:
                    return results
            except Exception as e:
                logger.debug(f"Polling for results: {e}")
            
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
        
        return None

    async def compute_tsne(self, run_id: str, perplexity: int | None = None) -> dict[str, Any]:
        """Compute t-SNE projection from saved embeddings."""
        try:
            payload = {"run_id": run_id}
            if perplexity is not None:
                payload["perplexity"] = perplexity
            response = await self.client.post(f"{self.base_url}/tsne", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to compute TSNE: {e}")
            raise

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()

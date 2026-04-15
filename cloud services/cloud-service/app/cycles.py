from __future__ import annotations

import httpx

from fastapi import HTTPException

from .config import settings


async def notify_edge(action: str, payload: dict) -> None:
    if not settings.edge_base_url:
        return
    url = f"{settings.edge_base_url.rstrip('/')}/cycle/{action}"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code >= 400:
                raise HTTPException(status_code=502, detail="Edge device rejected the request")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail="Edge device unavailable") from exc

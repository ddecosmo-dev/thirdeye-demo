"""Technical quality scorer — nima-spaq via pyiqa."""

from __future__ import annotations

import torch
import pyiqa


class TechnicalScorer:
    def __init__(self) -> None:
        self._model = None

    def load(self, device: str) -> None:
        """Load nima-spaq metric from pyiqa."""
        self._model = pyiqa.create_metric("nima-spaq", device=device)

    def score(self, img_tensor: torch.Tensor) -> float:
        """
        Score a single image tensor.
        img_tensor: shape (1, C, H, W), values in [0, 1].
        Returns raw quality score (~0–100).
        """
        return self._model(img_tensor).item()

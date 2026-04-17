"""Aesthetic scorer — DINOv2-Small backbone + custom MLPHead."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, AutoModel


# ─── Model Classes ────────────────────────────────────────────────────────────

class MLPHead(nn.Module):
    def __init__(self, embed_dim: int = 384, hidden_dim: int = 256, dropout_p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DINOAestheticScorer(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.head(cls_token)


# ─── Scorer ───────────────────────────────────────────────────────────────────

class AestheticScorer:
    """
    DINOv2-Small + fine-tuned MLPHead aesthetic scorer.

    score() returns both the aesthetic score and the L2-normalised CLS embedding,
    because the embedding is used downstream for HDBSCAN clustering.
    """

    def __init__(self) -> None:
        self._backbone = None
        self._head = None
        self._device: str = "cpu"
        self.processor = None  # exposed so callers can build inputs

    def load(self, device: str) -> None:
        """Download weights from HF Hub and move model to device."""
        self._device = device
        b_name = "facebook/dinov2-small"
        self.processor = AutoImageProcessor.from_pretrained(b_name)
        backbone = AutoModel.from_pretrained(b_name)
        head = MLPHead(embed_dim=384)

        ckpt_path = hf_hub_download(
            repo_id="grantmwilkinson/dinov2-small-mlphead-aesthetic",
            filename="dinov2-small_MLPHead_best.pt",
        )
        head.load_state_dict(torch.load(ckpt_path, map_location=device))

        full_model = DINOAestheticScorer(backbone, head).to(device)
        full_model.eval()
        self._backbone = full_model.backbone
        self._head = full_model.head

    def score(self, inputs: dict) -> tuple[float, np.ndarray]:
        """
        Run inference on preprocessed inputs.
        inputs: output of self.processor(..., return_tensors='pt'), already on device.
        Returns (aesthetic_score, l2_normalised_embedding).
        Caller must wrap in torch.no_grad().
        """
        outputs = self._backbone(**inputs)
        cls_token = outputs.last_hidden_state[:, 0, :]
        aes_score = self._head(cls_token).item()
        embed = cls_token / cls_token.norm(p=2, dim=-1, keepdim=True)
        return aes_score, embed.cpu().numpy().flatten()

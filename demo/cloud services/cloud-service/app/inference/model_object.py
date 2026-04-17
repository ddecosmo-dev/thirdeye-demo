"""Object-based scenic scorer — MaskFormer segmentation + AestheticMLP."""

from __future__ import annotations

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor


# ─── Model Classes ────────────────────────────────────────────────────────────

class AestheticMLP(nn.Module):
    def __init__(self, input_dim: int = 847, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class ScenicAestheticModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, maskformer_id: str = "facebook/maskformer-resnet50-ade20k-full"):
        super().__init__()
        self.maskformer = MaskFormerForInstanceSegmentation.from_pretrained(maskformer_id)
        self.mlp = AestheticMLP(input_dim=847)

    def forward(self, pixel_values):
        return self.maskformer(pixel_values)


# ─── Scorer ───────────────────────────────────────────────────────────

class ObjectScorer:
    """
    MaskFormer semantic segmentation + fine-tuned AestheticMLP scenic scorer.
    Converts pixel class distributions (847 ADE20k classes) into a single scenic score.
    """

    def __init__(self) -> None:
        self._maskformer = None
        self._mlp = None
        self._device: str = "cpu"
        self.processor = None  # exposed so callers can build inputs

    def load(self, device: str) -> None:
        """Download weights from HF Hub and move model to device."""
        import logging
        logger = logging.getLogger(__name__)
        
        self._device = device
        repo_id = "grantmwilkinson/aesthetic-maskformer-scenicornot-mlp"
        
        logger.info(f"Loading ObjectScorer processor from {repo_id}")
        self.processor = MaskFormerImageProcessor.from_pretrained(repo_id)
        logger.info("✓ Processor loaded")
        
        logger.info(f"Loading ObjectScorer from {repo_id}")
        combined = ScenicAestheticModel.from_pretrained(repo_id)
        logger.info("✓ Model downloaded/loaded")
        
        logger.info(f"Moving model to device {device}")
        combined = combined.to(device)
        logger.info("✓ Model moved to device")
        
        logger.info("Setting model to eval mode")
        combined.eval()
        logger.info("✓ Model in eval mode")
        
        self._maskformer = combined.maskformer
        self._mlp = combined.mlp
        logger.info("✓ ObjectScorer fully loaded")

    def score(self, inputs: dict, img_size: tuple[int, int]) -> float:
        """
        Run inference on preprocessed inputs.
        inputs: output of self.processor(..., return_tensors='pt'), already on device.
        img_size: (height, width) of the input image for segmentation post-processing.
        Returns scenic score.
        Caller must wrap in torch.no_grad().
        """
        outputs = self._maskformer(**inputs)
        semantic_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[img_size]
        )[0]
        pixel_counts = torch.bincount(semantic_map.flatten(), minlength=847).float()
        pixel_dist = pixel_counts / semantic_map.numel()
        return self._mlp(pixel_dist.to(self._device)).item()

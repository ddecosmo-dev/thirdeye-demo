"""ML inference pipeline ported from Cloud_Architecture_Final.ipynb."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import pyiqa
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, PyTorchModelHubMixin
from PIL import Image, ImageEnhance
from pillow_heif import register_heif_opener
register_heif_opener()
from hdbscan import HDBSCAN
from sklearn.manifold import TSNE
from transformers import (
    AutoImageProcessor,
    AutoModel,
    MaskFormerForInstanceSegmentation,
    MaskFormerImageProcessor,
)

# ─── Constants (mirrored from notebook) ──────────────────────────────────────
TECH_FLOOR = 3.0
W_AES = 0.6
W_OBJ = 0.4
INFERENCE_SIZE = (320, 240)   # (width, height) for model input
TOP_N_CHAMPIONS = 20
TSNE_ITER = 500               # faster than notebook's 1000, negligible quality diff


# ─── Model Definitions (verbatim from notebook cell-21) ──────────────────────
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


# Combined model matching the HF hub artifact (notebook cell-21)
class ScenicAestheticModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, maskformer_id: str = "facebook/maskformer-resnet50-ade20k-full"):
        super().__init__()
        self.maskformer = MaskFormerForInstanceSegmentation.from_pretrained(maskformer_id)
        self.mlp = AestheticMLP(input_dim=847)

    def forward(self, pixel_values):
        return self.maskformer(pixel_values)


# ─── Preprocessing (verbatim from notebook cell-19) ──────────────────────────
def auto_enhance_image(pil_img: Image.Image) -> Image.Image:
    open_cv_image = np.array(pil_img.convert("RGB"))
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    lab = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_cv = cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)

    enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced_cv, cv2.COLOR_BGR2RGB))
    enhanced_pil = ImageEnhance.Color(enhanced_pil).enhance(1.3)
    enhanced_pil = ImageEnhance.Sharpness(enhanced_pil).enhance(1.0)
    return enhanced_pil


# ─── Model Loading ────────────────────────────────────────────────────────────
def load_models(device: str) -> dict:
    """Load all three scoring models. Returns a dict of model components."""
    models = {}

    # 1. Technical model
    models["tech"] = pyiqa.create_metric("nima-spaq", device=device)

    # 2. Aesthetic model (DINOv2-Small + MLP head)
    b_name = "facebook/dinov2-small"
    aes_processor = AutoImageProcessor.from_pretrained(b_name)
    aes_backbone = AutoModel.from_pretrained(b_name)
    aes_head = MLPHead(embed_dim=384)

    ckpt_path = hf_hub_download(
        repo_id="grantmwilkinson/dinov2-small-mlphead-aesthetic",
        filename="dinov2-small_MLPHead_best.pt",
    )
    aes_head.load_state_dict(torch.load(ckpt_path, map_location=device))

    full_aes_model = DINOAestheticScorer(aes_backbone, aes_head).to(device)
    full_aes_model.eval()

    models["aes_processor"] = aes_processor
    models["aes_backbone"] = full_aes_model.backbone
    models["aes_head"] = full_aes_model.head

    # 3. Object-based aesthetic model (ScenicAestheticModel from Hub — notebook cell-21)
    repo_id = "grantmwilkinson/aesthetic-maskformer-scenicornot-mlp"
    maskformer_processor = MaskFormerImageProcessor.from_pretrained(repo_id)
    combined_model = ScenicAestheticModel.from_pretrained(repo_id).to(device)
    combined_model.eval()

    models["maskformer_processor"] = maskformer_processor
    models["maskformer"] = combined_model.maskformer
    models["object_mlp"] = combined_model.mlp

    return models


# ─── Scoring ──────────────────────────────────────────────────────────────────
def score_images(
    image_paths: list[Path],
    models: dict,
    device: str,
    progress_cb: Callable[[int], None] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Run all three scoring models on every image.
    Resizes to INFERENCE_SIZE before inference; original files are untouched.
    Returns (scores_df, embeddings) where embeddings shape is (N, 384).
    """
    from torchvision import transforms
    to_tensor = transforms.ToTensor()

    records = []
    all_embeddings = []

    tech_model = models["tech"]
    aes_processor = models["aes_processor"]
    aes_backbone = models["aes_backbone"]
    aes_head = models["aes_head"]
    maskformer_processor = models["maskformer_processor"]
    maskformer = models["maskformer"]
    object_mlp = models["object_mlp"]

    for idx, img_path in enumerate(image_paths):
        original_pil = Image.open(img_path).convert("RGB")

        # Resize to inference size
        small_pil = original_pil.resize(INFERENCE_SIZE, Image.LANCZOS)
        pil_img = auto_enhance_image(small_pil)
        img_tensor = to_tensor(pil_img).unsqueeze(0).to(device)

        aes_inputs = aes_processor(images=pil_img, return_tensors="pt").to(device)
        obj_inputs = maskformer_processor(images=pil_img, return_tensors="pt").to(device)

        with torch.no_grad():
            tech_score = tech_model(img_tensor).item()

            outputs = aes_backbone(**aes_inputs)
            cls_token = outputs.last_hidden_state[:, 0, :]
            aes_score = aes_head(cls_token).item()

            embed = cls_token / cls_token.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.append(embed.cpu().numpy().flatten())

            outputs_obj = maskformer(**obj_inputs)
            semantic_map = maskformer_processor.post_process_semantic_segmentation(
                outputs_obj, target_sizes=[pil_img.size[::-1]]
            )[0]
            pixel_counts = torch.bincount(semantic_map.flatten(), minlength=847).float()
            pixel_dist = pixel_counts / semantic_map.numel()
            obj_score = object_mlp(pixel_dist.to(device)).item()

        records.append(
            {
                "idx": idx,
                "filename": img_path.name,
                "technical_score": tech_score,
                "aesthetic_score": aes_score,
                "object_aesthetic_score": obj_score,
            }
        )

        if progress_cb:
            progress_cb(idx + 1)

    df = pd.DataFrame(records)
    embeddings = np.array(all_embeddings)
    return df, embeddings


# ─── Score Aggregation (ported from notebook cell-25) ─────────────────────────
def _normalize_col(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([5.0] * len(series), index=series.index)
    return 10 * (series - mn) / (mx - mn)


def aggregate_scores(df: pd.DataFrame, ignore_object: bool = False) -> pd.DataFrame:
    df = df.copy()
    df["aes_norm"] = _normalize_col(df["aesthetic_score"])
    df["obj_norm"] = _normalize_col(df["object_aesthetic_score"])
    df["tech_norm"] = df["technical_score"].clip(0, 100) / 10.0

    if ignore_object:
        df["aggregated_score"] = df["aes_norm"]
    else:
        df["aggregated_score"] = W_AES * df["aes_norm"] + W_OBJ * df["obj_norm"]

    low_tech_mask = df["tech_norm"] < TECH_FLOOR
    df.loc[low_tech_mask, "aggregated_score"] *= 0.5
    df["tech_penalized"] = low_tech_mask

    return df


# ─── Clustering (ported from notebook cell-9) ─────────────────────────────────
def run_clustering(embeddings: np.ndarray, df: pd.DataFrame, epsilon: float, min_cluster_size: int = 2) -> pd.DataFrame:
    df = df.copy()
    clusterer = HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_cluster_size),  # keep min_samples == min_cluster_size
        metric="euclidean",  # embeddings are L2-normalized so euclidean ≡ cosine
        cluster_selection_epsilon=float(epsilon),
        cluster_selection_method="leaf",
    )
    cluster_labels = clusterer.fit_predict(embeddings)
    df["cluster_id"] = cluster_labels
    return df


# ─── t-SNE ───────────────────────────────────────────────────────────────────
def run_tsne(embeddings: np.ndarray) -> np.ndarray:
    perplexity = min(30.0, len(embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=42, max_iter=TSNE_ITER, perplexity=perplexity)
    return tsne.fit_transform(embeddings)


# ─── Champion Selection (ported from notebook cell-9) ────────────────────────
def select_champions(df: pd.DataFrame, top_n: int = TOP_N_CHAMPIONS) -> pd.DataFrame:
    """
    Select top_n champions with at most one image per cluster.
    One-off images (cluster_id == -1) are each treated as their own unique
    cluster, so they compete freely alongside cluster representatives.
    """
    df = df.copy()
    df["is_final_selection"] = 0
    df_sorted = df.sort_values(by="aggregated_score", ascending=False)

    selected_cluster_ids: set[int] = set()
    count = 0

    for idx, row in df_sorted.iterrows():
        cluster_id = row["cluster_id"]
        if cluster_id == -1:
            # One-off: unique by definition — always eligible
            df.at[idx, "is_final_selection"] = 1
            count += 1
        elif cluster_id not in selected_cluster_ids:
            # First (best) image from this cluster
            df.at[idx, "is_final_selection"] = 1
            selected_cluster_ids.add(cluster_id)
            count += 1
        if count >= top_n:
            break

    return df


def assign_rejection_reasons(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _reason(row):
        if row["is_final_selection"] == 1:
            return None
        if row["tech_norm"] < TECH_FLOOR:
            return "tech_floor_penalty"
        if row["cluster_id"] == -1:
            return "noise_not_selected"
        return "outscored_in_cluster"

    df["rejection_reason"] = df.apply(_reason, axis=1)
    return df


# ─── Full Pipeline (convenience wrapper) ──────────────────────────────────────
def run_full_pipeline(
    image_paths: list[Path],
    device: str,
    epsilon: float,
    models: dict,
    progress_cb: Callable[[int], None] | None = None,
    min_cluster_size: int = 2,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Run scoring → aggregation → clustering → t-SNE → selection.
    Returns (df, embeddings, tsne_coords).
    t-SNE coords are NOT stored in df here — caller merges them.
    """
    df, embeddings = score_images(image_paths, models, device, progress_cb)
    df = aggregate_scores(df)
    df = run_clustering(embeddings, df, epsilon, min_cluster_size)
    tsne_coords = run_tsne(embeddings)
    df["tsne_x"] = tsne_coords[:, 0]
    df["tsne_y"] = tsne_coords[:, 1]
    df = select_champions(df)
    df = assign_rejection_reasons(df)
    return df, embeddings, tsne_coords

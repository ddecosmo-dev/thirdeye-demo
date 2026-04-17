"""ML inference pipeline orchestrator — coordinates the three scorer modules."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from pillow_heif import register_heif_opener
from skimage import exposure

register_heif_opener()

from hdbscan import HDBSCAN
from sklearn.manifold import TSNE

from .model_technical import TechnicalScorer
from .model_aesthetic import AestheticScorer
from .model_object import ObjectScorer

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
TECH_FLOOR = 3.0
W_AES = 0.6
W_OBJ = 0.4
INFERENCE_SIZE = (320, 240)   # (width, height) for model input
TOP_N_CHAMPIONS = 20
TSNE_ITER = 500               # faster than notebook's 1000, negligible quality diff


# ─── Preprocessing ────────────────────────────────────────────────────────────
def auto_enhance_image(pil_img: Image.Image) -> Image.Image:
    """Apply adaptive histogram equalization + color enhancement to image."""
    # Convert to numpy array (uint8)
    rgb_array = np.array(pil_img.convert("RGB"), dtype=np.uint8)

    # Apply CLAHE-equivalent (adaptive histogram equalization) to each channel
    # clip_limit=0.02 ~ OpenCV's clipLimit=2.0 for normalized [0,1] range
    enhanced = np.zeros_like(rgb_array, dtype=np.uint8)
    for i in range(3):
        equalized = exposure.equalize_adapthist(rgb_array[:, :, i], clip_limit=0.02)
        enhanced[:, :, i] = (equalized * 255).astype(np.uint8)

    enhanced_pil = Image.fromarray(enhanced, mode='RGB')
    enhanced_pil = ImageEnhance.Color(enhanced_pil).enhance(1.3)
    enhanced_pil = ImageEnhance.Sharpness(enhanced_pil).enhance(1.0)
    return enhanced_pil


# ─── Model Loading ────────────────────────────────────────────────────────────
def load_models(device: str) -> dict:
    """Instantiate and load all three scorer modules."""
    logger.info(f"Loading models on device: {device}")
    
    logger.info("━" * 60)
    logger.info("Loading TechnicalScorer (NIMA)")
    logger.info("━" * 60)
    tech = TechnicalScorer()
    tech.load(device)
    logger.info("✓ Technical scorer loaded")

    logger.info("")
    logger.info("━" * 60)
    logger.info("Loading AestheticScorer (DINOv2)")
    logger.info("━" * 60)
    aes = AestheticScorer()
    aes.load(device)
    logger.info("✓ Aesthetic scorer loaded")

    logger.info("")
    logger.info("━" * 60)
    logger.info("Loading ObjectScorer (MaskFormer)")
    logger.info("━" * 60)
    obj = ObjectScorer()
    obj.load(device)
    logger.info("✓ Object scorer loaded")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("ALL MODELS LOADED SUCCESSFULLY")
    logger.info("=" * 60)

    return {"tech": tech, "aes": aes, "obj": obj}


# ─── Scoring ──────────────────────────────────────────────────────────────────
def score_images(
    image_paths: list[Path],
    models: dict,
    device: str,
    progress_cb: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Run all three scoring models on every image.
    Resizes to INFERENCE_SIZE before inference; original files are untouched.
    Returns (scores_df, embeddings) where embeddings shape is (N, 384).
    """
    # Validate device is a string
    device = str(device).strip()
    if device not in ("cuda", "cpu"):
        logger.error(f"❌ Invalid device: {device}, forcing to cuda")
        device = "cuda"
    
    from torchvision import transforms

    to_tensor = transforms.ToTensor()

    tech_scorer: TechnicalScorer = models["tech"]
    aes_scorer: AestheticScorer = models["aes"]
    obj_scorer: ObjectScorer = models["obj"]

    records = []
    all_embeddings = []
    total = len(image_paths)

    for idx, img_path in enumerate(image_paths):
        try:
            original_pil = Image.open(img_path).convert("RGB")

            small_pil = original_pil.resize(INFERENCE_SIZE, Image.LANCZOS)
            pil_img = auto_enhance_image(small_pil)
            img_tensor = to_tensor(pil_img).unsqueeze(0).to(device)

            aes_inputs = aes_scorer.processor(images=pil_img, return_tensors="pt")
            aes_inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in aes_inputs.items()}
            
            obj_inputs = obj_scorer.processor(images=pil_img, return_tensors="pt")
            obj_inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in obj_inputs.items()}

            with torch.no_grad():
                tech_score = tech_scorer.score(img_tensor)
                aes_score, embed = aes_scorer.score(aes_inputs)
                obj_score = obj_scorer.score(obj_inputs, pil_img.size[::-1])

            all_embeddings.append(embed)

            records.append(
                {
                    "idx": idx,
                    "filename": img_path.name,
                    "technical_score": float(tech_score),
                    "aesthetic_score": float(aes_score),
                    "object_aesthetic_score": float(obj_score),
                }
            )
        except Exception as e:
            logger.error(f"Error scoring {img_path.name}: {e}")
            records.append(
                {
                    "idx": idx,
                    "filename": img_path.name,
                    "technical_score": 0.0,
                    "aesthetic_score": 0.0,
                    "object_aesthetic_score": 0.0,
                    "error": str(e),
                }
            )
            all_embeddings.append(np.zeros(384))

        # Log progress every 10 images or on completion
        processed = idx + 1
        if progress_cb:
            progress_cb(processed, total)
        
        if processed % 10 == 0 or processed == total:
            pct = (processed / total) * 100
            logger.info(f"  ⏳ Scored {processed}/{total} images ({pct:.0f}%)")

    df = pd.DataFrame(records)
    embeddings = np.array(all_embeddings)
    return df, embeddings


# ─── Score Aggregation ────────────────────────────────────────────────────────
def _normalize_col(series: pd.Series) -> pd.Series:
    """Normalize series to 0-10 range."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series([5.0] * len(series), index=series.index)
    return 10 * (series - mn) / (mx - mn)


def aggregate_scores(df: pd.DataFrame, ignore_object: bool = False) -> pd.DataFrame:
    """Aggregate individual scores into final composite score."""
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


# ─── Clustering ───────────────────────────────────────────────────────────────
def run_clustering(
    embeddings: np.ndarray, df: pd.DataFrame, epsilon: float, min_cluster_size: int = 2
) -> pd.DataFrame:
    """Cluster embeddings using HDBSCAN."""
    df = df.copy()
    clusterer = HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_cluster_size),
        metric="euclidean",
        cluster_selection_epsilon=float(epsilon),
        cluster_selection_method="leaf",
    )
    cluster_labels = clusterer.fit_predict(embeddings)
    df["cluster_id"] = cluster_labels.astype(int)
    return df


# ─── t-SNE ────────────────────────────────────────────────────────────────────
def run_tsne(embeddings: np.ndarray, perplexity: int = 30, seed: int = 42) -> np.ndarray:
    """Compute t-SNE projection of embeddings at runtime.
    
    Separated from main inference pipeline to allow re-projection with different parameters.
    
    Args:
        embeddings: (N, 384) array of L2-normalized DINOv2 embeddings
        perplexity: t-SNE perplexity parameter (affects clustering)
        seed: Random seed for reproducibility
    
    Returns:
        (N, 2) array of t-SNE coordinates
    """
    actual_perplexity = min(float(perplexity), len(embeddings) - 1)
    tsne = TSNE(n_components=2, random_state=seed, max_iter=TSNE_ITER, perplexity=actual_perplexity)
    return tsne.fit_transform(embeddings)


# ─── Champion Selection ───────────────────────────────────────────────────────
def select_champions(df: pd.DataFrame, top_n: int = TOP_N_CHAMPIONS) -> pd.DataFrame:
    """
    Select top_n champions with at most one image per cluster.
    One-off images (cluster_id == -1) compete freely.
    """
    df = df.copy()
    df["is_final_selection"] = 0
    df_sorted = df.sort_values(by="aggregated_score", ascending=False)

    selected_cluster_ids: set[int] = set()
    count = 0

    for idx, row in df_sorted.iterrows():
        cluster_id = row["cluster_id"]
        if cluster_id == -1:
            df.at[idx, "is_final_selection"] = 1
            count += 1
        elif cluster_id not in selected_cluster_ids:
            df.at[idx, "is_final_selection"] = 1
            selected_cluster_ids.add(cluster_id)
            count += 1
        if count >= top_n:
            break

    return df


def assign_rejection_reasons(df: pd.DataFrame) -> pd.DataFrame:
    """Assign reason for rejection to non-selected images."""
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


# ─── Full Pipeline ────────────────────────────────────────────────────────────
class InferenceRunner:
    """Orchestrates the full inference pipeline."""

    def __init__(self, device: str = "cpu", pre_loaded_models: dict | None = None):
        self.device = device
        self.models = pre_loaded_models  # Use pre-loaded models if provided

    def load_models(self) -> None:
        """Load all ML models."""
        if self.models is None:
            self.models = load_models(self.device)

    def run(
        self,
        image_paths: list[Path],
        epsilon: float = 0.12,
        min_cluster_size: int = 2,
        ignore_object: bool = False,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """
        Run full pipeline: scoring → aggregation → clustering → selection.
        Note: TSNE is NOT computed here; use compute_tsne() at runtime for flexibility.
        Returns dict with 'df' and 'embeddings'.
        """
        if self.models is None:
            self.load_models()

        logger.info(f"Starting inference on {len(image_paths)} images")
        
        # Scoring
        logger.info("Stage 1: Scoring images")
        df, embeddings = score_images(image_paths, self.models, self.device, progress_cb)

        # Aggregation
        logger.info("Stage 2: Aggregating scores")
        df = aggregate_scores(df, ignore_object=ignore_object)

        # Clustering
        logger.info("Stage 3: Clustering embeddings")
        df = run_clustering(embeddings, df, epsilon, min_cluster_size)

        # Champion selection
        logger.info("Stage 4: Selecting champions")
        df = select_champions(df)
        df = assign_rejection_reasons(df)

        logger.info("Inference complete")
        return {
            "df": df,
            "embeddings": embeddings,
        }

    def compute_tsne(self, embeddings: np.ndarray, perplexity: int | None = None) -> np.ndarray:
        """Compute t-SNE projection from embeddings (runtime, after inference).
        
        This allows demonstrating how perplexity affects visualization without
        re-running the expensive scoring, clustering stages.
        """
        if perplexity is None:
            perplexity = min(30, len(embeddings) - 1)
        return run_tsne(embeddings, perplexity=perplexity)


def dataframe_to_results_json(df: pd.DataFrame, embeddings: np.ndarray) -> list[dict[str, Any]]:
    """
    Convert inference results to JSON-serializable format.
    Each image gets a record with all scores, embedding, and metadata.
    Note: TSNE coordinates are NOT included; compute at runtime via /tsne endpoint.
    """
    results = []
    for idx, row in df.iterrows():
        embedding_list = embeddings[idx].tolist() if idx < len(embeddings) else []
        record = {
            "index": int(row.get("idx", idx)),
            "filename": str(row["filename"]),
            "scores": {
                "technical": float(row.get("technical_score", 0)),
                "aesthetic": float(row.get("aesthetic_score", 0)),
                "object": float(row.get("object_aesthetic_score", 0)),
            },
            "normalized_scores": {
                "tech_norm": float(row.get("tech_norm", 0)),
                "aes_norm": float(row.get("aes_norm", 0)),
                "obj_norm": float(row.get("obj_norm", 0)),
            },
            "aggregated_score": float(row.get("aggregated_score", 0)),
            "embedding": embedding_list,
            "cluster_id": int(row.get("cluster_id", -1)),
            "is_champion": int(row.get("is_final_selection", 0)),
            "tech_penalized": bool(row.get("tech_penalized", False)),
            "rejection_reason": row.get("rejection_reason"),
        }
        if "error" in row:
            record["error"] = str(row["error"])
        results.append(record)
    return results

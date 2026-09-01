"""DINOv3 model and path defaults."""

from pathlib import Path

# --- Model presets (Hugging Face gated repos; accept license + huggingface-cli login) ---
VITB16_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
VITL16_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

# Production default for full-corpus CLS runs.
DEFAULT_MODEL_ID = VITL16_MODEL_ID

# Laptop / smoke-test fallback (86M params, 768-dim CLS).
LAPTOP_MODEL_ID = VITB16_MODEL_ID

MODEL_CLS_DIM: dict[str, int] = {
    VITB16_MODEL_ID: 768,
    VITL16_MODEL_ID: 1024,
}

DEFAULT_CLS_SIZE = 224

CSV_DEFAULT = Path("data/sampled_with_thumbnails.csv")
THUMB_DIR_DEFAULT = Path("data/thumbnails")
ARCHIVE_ROOT = Path("data/archive")

# CLS track: one embedding / one cluster label per thumbnail
CLS_EMBEDDINGS_ROOT = Path("data/dinov3_cls_embeddings")
CLS_CLUSTERS_ROOT = Path("data/dinov3_cls_clusters")
CLS_CLUSTERING_TYPE = "cls"

# Patch track: one embedding / one cluster label per 16×16 token
PATCH_EMBEDDINGS_ROOT = Path("data/dinov3_patch_embeddings")
PATCH_CLUSTERS_ROOT = Path("data/dinov3_patch_clusters")
PATCH_CLUSTERING_TYPE = "patch"

DEFAULT_PATCH_SIZE = 224  # letterboxed image input size
DEFAULT_VIT_PATCH_SIZE = 16  # ViT */16 spatial patch size for crops/montages
DEFAULT_KMEANS_CLUSTERS = 40

# Shared clustering defaults (full ~8.6k corpus; tune via CLI)
DEFAULT_PCA_COMPONENTS = 50
DEFAULT_UMAP_NEIGHBORS = 30
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_UMAP_CLUSTER_COMPONENTS = 10  # HDBSCAN space; 2D UMAP is plot-only
DEFAULT_CLUSTER_SPACE = "umap"  # CLS HDBSCAN; pass pca to reproduce older runs
DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE = 3
DEFAULT_HDBSCAN_MIN_SAMPLES = 1
DEFAULT_HDBSCAN_SELECTION_METHOD = "eom"
DEFAULT_SAMPLES_PER_CLUSTER = 12

# Composite used to rank a CLS HDBSCAN sweep (min-max over the grid).
SWEEP_DBCV_WEIGHT = 0.5
SWEEP_SILHOUETTE_WEIGHT = 0.3
SWEEP_NOISE_WEIGHT = 0.2


def expected_cls_dim(model_id: str) -> int | None:
    return MODEL_CLS_DIM.get(model_id)

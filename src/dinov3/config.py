"""DINOv3 model and path defaults."""

from pathlib import Path

# --- Model presets (Hugging Face gated repos; accept license + huggingface-cli login) ---
VITB16_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
VITL16_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"

# Production default for full-corpus CLS runs on this branch.
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
EMBEDDINGS_ROOT = Path("data/dinov3_embeddings")
PATCH_EMBEDDINGS_ROOT = Path("data/dinov3_patch_embeddings")
ARCHIVE_ROOT = Path("data/archive")

# CLS thumbnail clustering (one label per image) — compare against patch motifs below
CLS_CLUSTERING_TYPE = "cls_thumbnail"
CLUSTERS_ROOT = Path("data/dinov3_clusters")

# Patch motif clustering (recurring local visual units across the corpus)
PATCH_CLUSTERING_TYPE = "patch_motif"
PATCH_MOTIFS_ROOT = Path("data/dinov3_patch_motifs")
DEFAULT_PATCH_SIZE = 224  # letterboxed image input size
DEFAULT_VIT_PATCH_SIZE = 16  # ViT */16 spatial patch size for crops/montages

# Shared clustering defaults (full ~8.6k corpus; tune via CLI)
DEFAULT_PCA_COMPONENTS = 50
DEFAULT_UMAP_NEIGHBORS = 30
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE = 3
DEFAULT_HDBSCAN_MIN_SAMPLES = 1
DEFAULT_SAMPLES_PER_CLUSTER = 12


def expected_cls_dim(model_id: str) -> int | None:
    return MODEL_CLS_DIM.get(model_id)
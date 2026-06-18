"""DINOv3 model and path defaults."""

from pathlib import Path

# Smaller model for laptop dev; use vitl16 on HPC for full runs.
DEFAULT_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_CLS_SIZE = 224

CSV_DEFAULT = Path("data/sampled_with_thumbnails.csv")
THUMB_DIR_DEFAULT = Path("data/thumbnails")
EMBEDDINGS_ROOT = Path("data/dinov3_embeddings")
PATCH_EMBEDDINGS_ROOT = Path("data/dinov3_patch_embeddings")

# CLS thumbnail clustering (one label per image) — compare against patch motifs below
CLS_CLUSTERING_TYPE = "cls_thumbnail"
CLUSTERS_ROOT = Path("data/dinov3_clusters")

# Patch motif clustering (recurring local visual units across the corpus)
PATCH_CLUSTERING_TYPE = "patch_motif"
PATCH_MOTIFS_ROOT = Path("data/dinov3_patch_motifs")
DEFAULT_PATCH_SIZE = 224

# Shared clustering defaults
DEFAULT_PCA_COMPONENTS = 50
DEFAULT_UMAP_NEIGHBORS = 15
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE = 10
DEFAULT_HDBSCAN_MIN_SAMPLES = 2
DEFAULT_SAMPLES_PER_CLUSTER = 12
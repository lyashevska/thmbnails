"""DINOv3 model and path defaults."""

from pathlib import Path

# Smaller model for laptop dev; use vitl16 on HPC for full runs.
DEFAULT_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_CLS_SIZE = 224

CSV_DEFAULT = Path("data/sampled_with_thumbnails.csv")
THUMB_DIR_DEFAULT = Path("data/thumbnails")
EMBEDDINGS_ROOT = Path("data/dinov3_embeddings")
CLUSTERS_ROOT = Path("data/dinov3_clusters")

# Clustering defaults tuned for ~1.7k noisy thumbnails
DEFAULT_PCA_COMPONENTS = 50
DEFAULT_UMAP_NEIGHBORS = 15
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE = 10
DEFAULT_HDBSCAN_MIN_SAMPLES = 2
DEFAULT_SAMPLES_PER_CLUSTER = 12
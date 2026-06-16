"""DINOv3 model and path defaults."""

from pathlib import Path

# Smaller model for laptop dev; use vitl16 on HPC for full runs.
DEFAULT_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
DEFAULT_CLS_SIZE = 224

CSV_DEFAULT = Path("data/sampled_with_thumbnails.csv")
THUMB_DIR_DEFAULT = Path("data/thumbnails")
EMBEDDINGS_ROOT = Path("data/dinov3_embeddings")
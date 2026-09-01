# Thumbnail study

Visual analysis of adult-video thumbnails, with a focus on gender and racial stereotypes.

The project samples videos over time, downloads their thumbnails, and analyzes the images in two complementary ways:

- **Prompted annotation** with a local vision-language model (structured JSON)
- **Label-free visual features** with DINOv3 (embeddings and clustering)

DINOv3 is used in two parallel tracks: one embedding and cluster label **per thumbnail** (CLS), and one embedding and cluster label **per local patch**.

How to install dependencies, pull models, and run the scripts is in [README-dev.md](README-dev.md). Methods, data, and results live under [docs/](docs/).

## Layout

```
src/sample_per_year.py   year-stratified sampling
src/scraper.py           thumbnail download
src/vlm_annotate.py      VLM annotation
src/dinov3/              DINOv3 preprocess, extract, check, cluster
  extract.py / cluster.py           libraries
  extract_cls.py / cluster_cls.py / cluster_cls_sweep.py   thumbnail-level CLI
  extract_patch.py / cluster_patch.py   patch-level CLI
data/                    samples, thumbnails, embeddings, clusters
docs/                    methods, data notes, run log
prompt                   VLM system prompt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then see [README-dev.md](README-dev.md).

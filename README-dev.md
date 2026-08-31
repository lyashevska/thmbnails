# Developer notes

Commands for sampling, thumbnail download, VLM annotation, and DINOv3 extraction/clustering. Project overview is in [README.md](README.md). Recorded run IDs are in [docs/dinov3_runs.md](docs/dinov3_runs.md).


```bash
source venv/bin/activate
```

## Sampling and thumbnails

Year-stratified sample (latest five years in the source CSV; `n` per year is set in the script):

```bash
python src/sample_per_year.py
```

Download thumbnails for the sample. The scraper is resumable: existing successes are skipped, and files already on disk are reconciled into the CSV.

```bash
python src/scraper.py
```

Outputs:

- `data/sampled_data.csv`
- `data/sampled_with_thumbnails.csv`
- `data/thumbnails/{viewkey}.jpg`
- `data/scraper.log`

Downstream model scripts skip files smaller than 4 KB (placeholders) and expect 640×360 source images.

## VLM annotation (Ollama)

`src/vlm_annotate.py` sends each thumbnail plus its title to a local Ollama model and asks for JSON following `prompt`.

```bash
ollama serve
ollama pull huihui_ai/qwen3-vl-abliterated:4b-instruct

python src/vlm_annotate.py --dry-run --limit 10
python src/vlm_annotate.py --limit 3 --force
```

The script reads `data/sampled_with_thumbnails.csv`, skips thumbnails under 4 KB, and writes:

- `data/annotations_ollama/<image_id>.json`
- `data/annotations_ollama/<image_id>.raw.txt` (only when JSON parse fails)
- `data/annotations_ollama.jsonl`

Existing per-image JSON is skipped unless you pass `--force`. Postprocessing is minimal; spot-check outputs before scaling. Useful flags: `--csv`, `--out-dir`, `--results`, `--prompt`, `--model`.

## DINOv3

DINOv3 checkpoints on Hugging Face are gated. Accept the model license, then:

```bash
huggingface-cli login
```

Default checkpoint is ViT-L/16 (`facebook/dinov3-vitl16-pretrain-lvd1689m`, 1024-d CLS, typically 8–12 GB VRAM). Override with `--model` for a laptop smoke test:

```bash
python src/dinov3/extract_cls.py \
  --model facebook/dinov3-vitb16-pretrain-lvd1689m --limit 10
```

| Model | Parameters | CLS dim | VRAM | Use |
|-------|-----------|---------|------|-----|
| ViT-S / S+ | 21–50M | 384 | 1–3 GB | prototyping |
| ViT-B/16 | 86M | 768 | 3–6 GB | smoke tests |
| **ViT-L/16** | **300M** | **1024** | **8–12 GB** | **default / production** |
| ViT-H+ | 840M | 1280 | 20–30+ GB | high-end only |

ViT-L on GPU is on the order of minutes for a few thousand images, tens of minutes for ~10k. CPU works for a smoke test (seconds per image) but not a full corpus run.


### Preprocessing

`src/dinov3/preprocess.py` filters invalid files, converts to RGB, letterboxes 16:9 to a square (black pad), then resizes to 224px for both CLS and patch extraction.

Valid thumbnail: readable, ≥ 4096 bytes, 640×360.

```bash
python src/dinov3/inspect_preprocess.py --limit 10 --seed 42
```

### CLS track (one vector / one cluster per thumbnail)

Extract (resumable; skips existing `vectors/<image_id>.npy` unless `--force`):

```bash
python src/dinov3/extract_cls.py --dry-run --limit 10
python src/dinov3/extract_cls.py --limit 20
python src/dinov3/extract_cls.py
python src/dinov3/extract_cls.py --run-id <run_id>   # resume
```

`--dry-run` still creates an empty run directory. Always pass `--embeddings-run-id` when clustering so an empty dry-run folder is not picked as “latest”.

Validate:

```bash
python src/dinov3/check_cls.py --run-id <run_id>
```

A healthy ViT-L run prints `Embeddings shape: (N, 1024)`, matching ID count, finite norms, and `OK: basic embedding checks passed.`

Outputs under `data/dinov3_cls_embeddings/<run_id>/`:

- `vectors/<image_id>.npy`
- `cls_embeddings.npy` — stacked `(N, D)`
- `image_ids.json`
- `manifest.json`

Cluster (PCA → UMAP → HDBSCAN by default; also `--method kmeans` or `agglomerative`):

```bash
python src/dinov3/cluster_cls.py --embeddings-run-id <run_id>
python src/dinov3/cluster_cls.py \
  --embeddings-run-id <run_id> \
  --hdbscan-min-cluster-size 3 --hdbscan-min-samples 1 --umap-neighbors 30
```

Outputs under `data/dinov3_cls_clusters/<run_id>/`:

- `cluster_assignments.csv`
- `cluster_summary.csv`
- `umap.png`
- `samples/cluster_<id>/` (example thumbnails + `_grid.jpg`)
- `manifest.json` (`clustering_type: cls`)

### Patch track (one vector / one cluster per 16×16 token)

Reuse the CLS corpus image list when possible. Letterbox padding rows are masked.

```bash
python src/dinov3/extract_patch.py --embeddings-run-id <cls_run_id> --limit 50
python src/dinov3/extract_patch.py --embeddings-run-id <cls_run_id>
python src/dinov3/extract_patch.py --run-id <run_id>   # resume
python src/dinov3/check_patch.py --run-id <run_id>
```

Outputs under `data/dinov3_patch_embeddings/<run_id>/`:

- `vectors/<image_id>.npz`
- `image_ids.json`
- `manifest.json`

Cluster:

```bash
python src/dinov3/cluster_patch.py --patch-run-id <patch_run_id>
```

Outputs under `data/dinov3_patch_clusters/<run_id>/`:

- `patch_assignments.csv`
- `cluster_summary.csv`
- `image_cluster_histogram.csv`
- `patch_umap.png`
- `patch_crops/cluster_<id>/_grid.jpg`
- `manifest.json` (`clustering_type: patch`)

# DINOv3 run log

## Active target (ViT-L CLS, full corpus)

| Setting | Value |
|---------|-------|
| Model | `facebook/dinov3-vitl16-pretrain-lvd1689m` (ViT-L/16, ~300M) |
| CLS dim | 1024 |
| Input CSV | `data/sampled_with_thumbnails.csv` |
| Valid thumbnails | ~8,666 (≥ 4 KB, on disk) |
| Branch | `feature/dinov3-vitl-cls` |

### Extract CLS embeddings

```bash
python src/dinov3/extract_embeddings.py --dry-run
python src/dinov3/extract_embeddings.py --limit 20
python src/dinov3/extract_embeddings.py
python src/dinov3/check_embeddings.py   # uses latest run
```

Resume into an existing run:

```bash
python src/dinov3/extract_embeddings.py --run-id <run_id>
```

Laptop smoke test (ViT-B/16):

```bash
python src/dinov3/extract_embeddings.py \
  --model facebook/dinov3-vitb16-pretrain-lvd1689m \
  --limit 10
```

### Model inference timing

Each CLS embedding run records DINOv3 model time in `manifest.json` → `model_timing`:

- `model_load_seconds` — one-time model load to GPU/CPU
- `inference_seconds` / `inference_human` — forward passes this run (excludes load)
- `seconds_per_image` — inference / successful images
- `device` — e.g. `cuda:0` or `cpu`

### Cluster CLS thumbnails

Recommended starting point (from parameter sweep on pilot data; re-tune on full corpus):

```bash
python src/dinov3/cluster_embeddings.py \
  --hdbscan-min-cluster-size 3 \
  --hdbscan-min-samples 1 \
  --umap-neighbors 30
```

## Archived pilot (ViT-B/16, 1,743 images)

Moved to `data/archive/dinov3_vitb16_pilot_1743/` by `scripts/archive_dinov3_vitb_pilot.sh`.

| Run | ID | Notes |
|-----|-----|-------|
| CLS embeddings | `20260617T091002Z` | 768-dim, 1,743 images |
| CLS clusters | `20260617T123852Z` | 83% HDBSCAN noise |
| Patch embeddings | `20260618T125222Z` | 224px patches |
| Patch motifs | `20260618T140719Z` | 72% patch noise |

Do not mix ViT-B and ViT-L embeddings in the same clustering run.
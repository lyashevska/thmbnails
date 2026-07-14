# DINOv3 run log

## Active target (ViT-L CLS, full corpus)

| Setting | Value |
|---------|-------|
| Model | `facebook/dinov3-vitl16-pretrain-lvd1689m` (ViT-L/16, ~300M) |
| CLS dim | 1024 |
| Input CSV | `data/sampled_with_thumbnails.csv` |
| Valid thumbnails (CSV, ≥ 4 KB) | 10,957 |
| Embeddable (inference OK) | 8,666 |
| Branch | `feature/dinov3-vitl-cls` |

### Production CLS embeddings (complete)

| Run ID | `20260713T131720Z` |
|--------|---------------------|
| Model | `facebook/dinov3-vitl16-pretrain-lvd1689m` |
| Shape | `(8666, 1024)` |
| Inference failures | 2,291 (unreadable/corrupt despite ≥ 4 KB) |
| Timing | load 1.18s, inference 5m 1s, **0.035 s/image** (cuda) |

```bash
python src/dinov3/check_embeddings.py --run-id 20260713T131720Z
```

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

### Cluster CLS thumbnails (active)

Two complementary runs on the full ViT-L corpus. **Use both for now** — not agglomerative or HDBSCAN `leaf`.

| Role | Method | Run folder | Clusters | Noise |
|------|--------|------------|----------|-------|
| **Motif discovery** | HDBSCAN `eom` | `hdbscan-eom-vitl` | 539 | 69.2% (6,000) |
| **Corpus taxonomy** | K-means K=40 | `kmeans-k40-vitl` | 40 | 0% |

Shared settings: PCA 50 (59% variance), UMAP neighbors 30, embeddings `20260713T131720Z`.

**HDBSCAN eom** — tight visual micro-motifs (feet, close-ups, hentai, etc.); most images stay noise.  
**K-means K=40** — every thumbnail labeled; broader themes (~73–404 images per cluster).

```bash
# Compare methods/settings (no sample grids)
python src/dinov3/compare_cluster_methods.py --embeddings-run-id 20260713T131720Z

# Motif discovery (HDBSCAN eom)
python src/dinov3/cluster_embeddings.py \
  --embeddings-run-id 20260713T131720Z \
  --method hdbscan \
  --hdbscan-selection-method eom \
  --hdbscan-min-cluster-size 3 \
  --hdbscan-min-samples 1 \
  --run-id hdbscan-eom-vitl

# Corpus taxonomy (k-means)
python src/dinov3/cluster_embeddings.py \
  --embeddings-run-id 20260713T131720Z \
  --method kmeans \
  --n-clusters 40 \
  --run-id kmeans-k40-vitl
```

Review: `data/dinov3_clusters/<run_id>/umap.png` and `samples/cluster_*/_grid.jpg`.

Patch motifs (`extract_patch_embeddings.py` / `cluster_patch_motifs.py`) deferred until CLS runs are reviewed.

## Archived pilot (ViT-B/16, 1,743 images)

Moved to `data/archive/dinov3_vitb16_pilot_1743/` by `scripts/archive_dinov3_vitb_pilot.sh`.

| Run | ID | Notes |
|-----|-----|-------|
| CLS embeddings | `20260617T091002Z` | 768-dim, 1,743 images |
| CLS clusters | `20260617T123852Z` | 83% HDBSCAN noise |
| Patch embeddings | `20260618T125222Z` | 224px patches |
| Patch motifs | `20260618T140719Z` | 72% patch noise |

Do not mix ViT-B and ViT-L embeddings in the same clustering run.
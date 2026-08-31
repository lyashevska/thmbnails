# DINOv3 run log

Two parallel tracks, same two steps: extract embeddings, then cluster them.

```
data/dinov3_cls_embeddings/     one 1024-d vector per thumbnail
data/dinov3_cls_clusters/       one cluster label per thumbnail
data/dinov3_patch_embeddings/   112 × 1024-d vectors per thumbnail
data/dinov3_patch_clusters/     one cluster label per patch
```

## Active target (ViT-L, full corpus)

| Setting | Value |
|---------|-------|
| Model | `facebook/dinov3-vitl16-pretrain-lvd1689m` (ViT-L/16, ~300M) |
| Dim | 1024 |
| Input CSV | `data/sampled_with_thumbnails.csv` |
| Valid thumbnails (CSV, ≥ 4 KB) | 10,957 |
| Embeddable (inference OK) | 8,666 |

### Production CLS embeddings (complete)

| Run ID | `20260713T131720Z` |
|--------|---------------------|
| Path | `data/dinov3_cls_embeddings/20260713T131720Z` |
| Shape | `(8666, 1024)` |
| Inference failures | 2,291 (unreadable/corrupt despite ≥ 4 KB) |
| Timing | load 1.18s, inference 5m 1s, **0.035 s/image** (cuda) |

```bash
python src/dinov3/extract_cls.py --dry-run
python src/dinov3/extract_cls.py --limit 20
python src/dinov3/extract_cls.py
python src/dinov3/check_cls.py --run-id 20260713T131720Z
```

Resume into an existing run:

```bash
python src/dinov3/extract_cls.py --run-id <run_id>
```

Laptop smoke test (ViT-B/16):

```bash
python src/dinov3/extract_cls.py \
  --model facebook/dinov3-vitb16-pretrain-lvd1689m \
  --limit 10
```

Each CLS embedding run records `model_timing` in `manifest.json` (load, inference, seconds/image, device).

### Production CLS clusters (complete)

Two complementary runs on embeddings `20260713T131720Z`. Shared settings: PCA 50 (59% variance), UMAP neighbors 30.

| Role | Method | Run folder | Clusters | Noise |
|------|--------|------------|----------|-------|
| Tight visual groups | HDBSCAN `eom` | `hdbscan-eom-vitl` | 539 | 69.2% (6,000) |
| Full-corpus taxonomy | K-means K=40 | `kmeans-k40-vitl` | 40 | 0% |

```bash
python src/dinov3/cluster_cls.py \
  --embeddings-run-id 20260713T131720Z \
  --method hdbscan \
  --hdbscan-selection-method eom \
  --hdbscan-min-cluster-size 3 \
  --hdbscan-min-samples 1 \
  --run-id hdbscan-eom-vitl

python src/dinov3/cluster_cls.py \
  --embeddings-run-id 20260713T131720Z \
  --method kmeans \
  --n-clusters 40 \
  --run-id kmeans-k40-vitl
```

Review: `data/dinov3_cls_clusters/<run_id>/umap.png` and `samples/cluster_*/_grid.jpg`.

### Production patch embeddings (complete)

| Run ID | `20260714T101958Z` |
|--------|---------------------|
| Path | `data/dinov3_patch_embeddings/20260714T101958Z` |
| Images | 8,666 (same corpus as CLS run `20260713T131720Z`) |
| Tokens | 112 per image (8×14 after top/bottom letterbox rows are masked) |
| Timing | load 1.11s, inference 6m 55s, **0.048 s/image** (cuda) |

```bash
python src/dinov3/extract_patch.py \
  --embeddings-run-id 20260713T131720Z \
  --limit 20

python src/dinov3/extract_patch.py \
  --embeddings-run-id 20260713T131720Z

python src/dinov3/check_patch.py --run-id 20260714T101958Z
```

### Production patch clusters (complete)

| Run folder | `hdbscan-eom-vitl-patches` |
|------------|----------------------------|
| Path | `data/dinov3_patch_clusters/hdbscan-eom-vitl-patches` |
| Patch embeddings | `20260714T101958Z` |
| Method | HDBSCAN `eom`, min_cluster_size=30, min_samples=3 |
| Patches | 970,592 |
| Clusters | 1,285 |
| Noise | 83.7% (812,646 patches) |

```bash
python src/dinov3/cluster_patch.py \
  --patch-run-id 20260714T101958Z \
  --run-id hdbscan-eom-vitl-patches
```

Review: `patch_assignments.csv`, `cluster_summary.csv`, and `patch_crops/cluster_*/_grid.jpg`.

## Archived pilot (ViT-B/16, 1,743 images)

Moved with `scripts/archive_dinov3_vitb_pilot.sh` (not on disk in the current tree). Do not mix ViT-B and ViT-L embeddings in the same clustering run.

| Run | ID | Notes |
|-----|----|-------|
| CLS embeddings | `20260617T091002Z` | 768-dim, 1,743 images |
| CLS clusters | `20260617T123852Z` | 83% HDBSCAN noise |
| Patch embeddings | `20260618T125222Z` | 224px patches |
| Patch clusters | `20260618T140719Z` | 72% patch noise |

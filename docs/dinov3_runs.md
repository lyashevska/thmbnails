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

### Production CLS clusters

Embeddings `20260713T131720Z`. Commands: [README-dev.md](../README-dev.md).

#### Chosen density clustering (UMAP + HDBSCAN): `sweep-A-n15-mcs20-ms20`

Working labels for visual interpretation. **Not** a seed-proof taxonomy.

| Setting | Value |
|---------|-------|
| Path | `data/dinov3_cls_clusters/sweep-A-n15-mcs20-ms20` |
| Cluster space | 10-D UMAP (PCA 50 → 59% variance; 2-D UMAP is plot-only) |
| UMAP | `n_neighbors=15`, `min_dist=0.0` |
| HDBSCAN | `eom`, `min_cluster_size=20`, `min_samples=20` |
| Seed | 42 |
| Clusters | 36 |
| Noise | 38.6% (3,348) |
| Median cluster size | 65 |
| DBCV / silhouette | 0.32 / 0.59 |

**Why A.** A 99-cell sweep (`sweeps/20260831T172632Z`) ranked cells with DBCV 50% / silhouette 30% / (1 − noise) 20%. The composite winner was a **2-cluster, 0% noise** split (median size 4,333) and was discarded. Every cell with ≥ 20 clusters had `min_dist=0`. In that band, A had the best composite, tighter cores than B (same UMAP; `mcs=30`, `ms=10` → 32 clusters, 34% noise, median 92), and leftover mass for a later noise peel.

**Stability (UMAP seeds, knobs frozen).** Pairwise ARI/NMI. Raising `n_neighbors` to 50 or dropping UMAP to 5-D did not help.

| Probe | ARI | NMI |
|-------|-----|-----|
| A, 100 seeds (`sweep-A-stability`) | 0.45 ± 0.37 | 0.55 ± 0.29 |
| D (`n_neighbors=50`), 10 seeds | 0.51 ± 0.41 | 0.58 ± 0.33 |
| 5-D UMAP, `n_neighbors=50`, 10 seeds | 0.51 ± 0.41 | 0.58 ± 0.32 |

Treat A as an exploratory seed-42 clustering. Review `umap.png` and `samples/cluster_*/_grid.jpg`. Next analysis step: peel A’s 3,348 noise points (not implemented yet).

#### Companion / older CLS cluster runs

| Role | Method | Run folder | Clusters | Noise |
|------|--------|------------|----------|-------|
| Full-corpus taxonomy (deterministic) | K-means K=40 on PCA | `kmeans-k40-vitl` | 40 | 0% |
| Earlier PCA-space HDBSCAN (not the chosen cut) | HDBSCAN `eom` on PCA, `mcs=3`, `ms=1` | `hdbscan-eom-vitl` | 539 | 69.2% (6,000) |

```bash
python src/dinov3/cluster_cls.py \
  --embeddings-run-id 20260713T131720Z \
  --cluster-space pca --method kmeans --n-clusters 40 \
  --run-id kmeans-k40-vitl
```

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

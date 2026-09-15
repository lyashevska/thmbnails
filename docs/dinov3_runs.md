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

Embeddings `20260713T131720Z`. Commands: [README-dev.md](https://github.com/lyashevska/thmbnails/blob/main/README-dev.md).

#### Chosen density clustering (UMAP + HDBSCAN): `nopca-n15-mcs20-ms20`

Working labels for visual interpretation. **Not** a seed-proof taxonomy.

| Setting | Value |
|---------|-------|
| Path | `data/dinov3_cls_clusters/nopca-n15-mcs20-ms20` |
| Cluster space | 10-D UMAP of raw 1024-D CLS (no PCA; 2-D UMAP is plot-only) |
| UMAP | `n_neighbors=15`, `min_dist=0.0` |
| HDBSCAN | `eom`, `min_cluster_size=20`, `min_samples=20` |
| Seed | 42 |
| Clusters | 33 |
| Noise | 41.1% (3,558) |
| Median cluster size | 54 |
| DBCV / silhouette | 0.27 / 0.54 |

**Why this cut.** Same 99-cell UMAP/HDBSCAN grid as the earlier PCA-first sweep, but UMAP on raw CLS (`--skip-pca`). Sweep: `sweeps/nopca`. Composite leaders were **2-cluster, 0% noise** splits (median size 4,333) and were discarded. The usable band (`n_clusters` 20–80) had 12 cells; almost all had `min_dist=0`. The winner used `n_neighbors=15`, `min_dist=0`, `mcs=20`, `ms=20` — the same knobs as PCA-first cut A (`sweep-A-n15-mcs20-ms20`: 36 clusters, 38.6% noise).

Native UMAP scores slightly favour A. Shared-space silhouette (raw CLS cosine; PCA-50 euclidean) slightly favours this cut. Assigned-only ARI vs A is 0.90 (cores agree; noise assignment differs). The extra PCA step was dropped as unnecessary. Comparison: `compare-sweep-A-n15-mcs20-ms20-vs-nopca-n15-mcs20-ms20/comparison.json`.

**Stability (UMAP seeds, knobs frozen).** Pairwise ARI/NMI. Dropping PCA did not stabilise the typology. Raising `n_neighbors` to 50 or dropping UMAP to 5-D (PCA path) did not help.

| Probe | ARI | NMI |
|-------|-----|-----|
| Working cut, 100 seeds (`nopca-n15-mcs20-ms20-stability`) | 0.47 ± 0.35 | 0.57 ± 0.24 |
| A (PCA then 10-D UMAP), 100 seeds (`sweep-A-stability`) | 0.45 ± 0.37 | 0.55 ± 0.29 |
| D (`n_neighbors=50`, PCA), 10 seeds | 0.51 ± 0.41 | 0.58 ± 0.33 |
| 5-D UMAP, `n_neighbors=50`, 10 seeds | 0.51 ± 0.41 | 0.58 ± 0.32 |

Treat this as an exploratory seed-42 clustering. Review `umap.png` and `samples/cluster_*/_grid.jpg`.

**Noise peels.** Two rounds on `cluster_id=-1`, knobs inherited (`skip-pca`, 10-D UMAP, `n_neighbors=15`, `min_dist=0`, `mcs=20`, `ms=20`, `eom`):

```bash
python src/dinov3/cluster_cls_peel.py \
  --from-clusters-run-id nopca-n15-mcs20-ms20 --rounds 2
```

| Round | Folder | Input | New clusters | Remaining noise | DBCV / silhouette |
|-------|--------|------:|-------------:|----------------:|-------------------|
| 0 | `nopca-n15-mcs20-ms20` | 8,666 | 33 | 41.1% (3,558) | 0.27 / 0.54 |
| 1 | `nopca-n15-mcs20-ms20-r1` | 3,558 | 11 | 35.2% (1,254) | 0.07 / 0.13 |
| 2 | `nopca-n15-mcs20-ms20-r2` | 1,254 | 9 | 50.4% (632) | 0.23 / 0.45 |

Combined (`nopca-n15-mcs20-ms20-peels/`): **53** cluster ids, **8,034** assigned (92.7%), **632** still noise (7.3%). Round 0 remains the primary taxonomy (5,108 images). Round 1 residual pile is folder **3** (combined id **36**; 1,402 images, 39.4% of r1). Round 0 folders **0** and **18** (741 and 842) are oversized mixed groups. Inspect `samples/cluster_*/_grid.jpg` on r1 (other than 3) and r2 before naming extra types.

**Residual CLS (Stage 1).** Subtract the parent-cluster mean in raw 1024-d CLS (noise: nearest centroid), L2-normalise, refit UMAP+HDBSCAN on all 8,666 images. The **operating cut is leaf**, `min_samples=10`. Inherited eom / `min_samples=20` is a control: it collapses to illustrated vs live-action and is not used.

```bash
python src/dinov3/cluster_cls_residual.py \
  --from-clusters-run-id nopca-n15-mcs20-ms20 \
  --hdbscan-selection-method leaf --hdbscan-min-samples 10 \
  --run-id nopca-n15-mcs20-ms20-residual-leaf

python src/dinov3/cluster_cls_residual.py \
  --from-clusters-run-id nopca-n15-mcs20-ms20
```

| Role | Run | Folder | Clusters | Noise | DBCV / silhouette |
|------|-----|--------|---------:|------:|-------------------|
| **Operating Stage 1** | leaf, ms=10 | `nopca-n15-mcs20-ms20-residual-leaf/` | 68 | 56.2% (4,868) | 0.19 / 0.51 |
| Control (not used) | inherited knobs (eom, ms=20) | `nopca-n15-mcs20-ms20-residual/` | 2 (708, 7,958) | 0% | 0.71 / 0.60 |

Leaf: 3,798 assigned; median residual group mixes 6 parent ids (median majority-parent fraction 0.62). Mixed grids share clothing, body, or couple composition across parent sets. Unmixed residual groups still track parent 0 (drawing medium) or parent 18 (close-up). Review `samples/cluster_*/_grid.jpg` and `parent_residual_mix.csv`. Keep / skip list: [cls_residual_stage1.md](cls_residual_stage1.md). The control run’s residual 0 is 98% parent 0 (illustrated vs the rest).

#### Companion / older CLS cluster runs

| Role | Method | Run folder | Clusters | Noise |
|------|--------|------------|----------|-------|
| PCA-first usable-band winner (same knobs; extra PCA) | PCA → 10-D UMAP + HDBSCAN | `sweep-A-n15-mcs20-ms20` | 36 | 38.6% (3,348) |
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

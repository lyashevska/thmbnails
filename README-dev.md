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

Default HDBSCAN path is PCA → **10-D UMAP (cluster space)** → HDBSCAN. A separate 2D UMAP is only for `umap.png`. Pass `--skip-pca` (or `--pca-components 0`) to run UMAP on raw CLS. Older PCA-space runs (`hdbscan-eom-vitl`, `kmeans-k40-vitl`) need `--cluster-space pca`.

**Chosen CLS density clustering:** `sweep-A-n15-mcs20-ms20` (see [docs/dinov3_runs.md](docs/dinov3_runs.md) for why). Commands that produced the recorded results are below. `--embeddings-run-id` is always the CLS vector dump `20260713T131720Z`, not a cluster folder.

Outputs under `data/dinov3_cls_clusters/<run_id>/`:

- `cluster_assignments.csv`
- `cluster_summary.csv`
- `cluster_space.npy` — array HDBSCAN used (10-D UMAP or PCA)
- `umap.png`
- `samples/cluster_<id>/` (example thumbnails + `_grid.jpg`)
- `manifest.json` (`clustering_type: cls`, plus `dbcv` / `silhouette` / `noise_fraction`)
- `stability.json` — only when `--stability-runs` is set

#### Parameter sweep (grid, no plots)

99 cells: UMAP `n_neighbors` ∈ {15,30,50}, `min_dist` ∈ {0.0,0.1,0.25}, HDBSCAN `min_cluster_size` ∈ {10,20,30,50}, `min_samples` ∈ {5, 10, min_cluster_size} (duplicates dropped). Cluster space is 10-D UMAP, `eom`. Ranked by min-max **DBCV 50% / silhouette 30% / (1 − noise) 20%**. That ranking is not the winner: the composite favoured 2-cluster, 0% noise splits.

```bash
python src/dinov3/cluster_cls_sweep.py --embeddings-run-id 20260713T131720Z --dry-run
python src/dinov3/cluster_cls_sweep.py --embeddings-run-id 20260713T131720Z
```

Wrote `data/dinov3_cls_clusters/sweeps/20260831T172632Z/sweep.csv`.

Same grid with PCA skipped (UMAP on raw 1024-D CLS):

```bash
python src/dinov3/cluster_cls_sweep.py --embeddings-run-id 20260713T131720Z --skip-pca --dry-run
python src/dinov3/cluster_cls_sweep.py --embeddings-run-id 20260713T131720Z --skip-pca \
  --run-id nopca
```

Shortlist with the same rule as cut A (discard 2-cluster composite leaders; keep `n_clusters` 20–80). Then render and compare:

```bash
The no-PCA usable-band winner used the same knobs as cut A (`n_neighbors=15`, `min_dist=0`, `mcs=20`, `ms=20`):

```bash
python src/dinov3/cluster_cls.py --embeddings-run-id 20260713T131720Z --skip-pca \
  --umap-neighbors 15 --umap-min-dist 0.0 \
  --hdbscan-min-cluster-size 20 --hdbscan-min-samples 20 \
  --run-id nopca-n15-mcs20-ms20

python src/dinov3/compare_cluster_runs.py \
  --run-a sweep-A-n15-mcs20-ms20 --label-a PCA-UMAP \
  --run-b nopca-n15-mcs20-ms20 --label-b UMAP
```

#### Shortlist (render grids)

All usable cells (`n_clusters` 20–80) had `min_dist=0`. Rendered A (chosen) and B (same UMAP, coarser HDBSCAN):

```bash
python src/dinov3/cluster_cls.py --embeddings-run-id 20260713T131720Z \
  --umap-neighbors 15 --umap-min-dist 0.0 \
  --hdbscan-min-cluster-size 20 --hdbscan-min-samples 20 \
  --run-id sweep-A-n15-mcs20-ms20

python src/dinov3/cluster_cls.py --embeddings-run-id 20260713T131720Z \
  --umap-neighbors 15 --umap-min-dist 0.0 \
  --hdbscan-min-cluster-size 30 --hdbscan-min-samples 10 \
  --run-id sweep-B-n15-mcs30-ms10
```

A: 36 clusters, 38.6% noise (3,348), median size 65, DBCV 0.32. B: 32 clusters, 34.0% noise, median size 92. A was kept for tighter cores and more leftover mass for a later noise peel.

Reproduce A (or any later run) with the same flags; pass `--cluster-space pca` only for the old k-means / PCA-HDBSCAN folders.

#### Stability (after A was frozen, not during the search)

PCA frozen; only the UMAP seed changes. Pairwise ARI and NMI (noise as a label, and assigned-only).

```bash
# A — 100 seeds (reported)
python src/dinov3/cluster_cls.py --embeddings-run-id 20260713T131720Z \
  --umap-neighbors 15 --umap-min-dist 0.0 \
  --hdbscan-min-cluster-size 20 --hdbscan-min-samples 20 \
  --stability-runs 100 \
  --run-id sweep-A-stability

# D — same HDBSCAN as A, n_neighbors=50; 10 seeds (did not improve)
python src/dinov3/cluster_cls.py --embeddings-run-id 20260713T131720Z \
  --umap-neighbors 50 --umap-min-dist 0.0 \
  --hdbscan-min-cluster-size 20 --hdbscan-min-samples 20 \
  --stability-runs 10 \
  --run-id sweep-D-n50-mcs20-ms20-stab10

# 5-D UMAP — 10 seeds (did not improve)
python src/dinov3/cluster_cls.py --embeddings-run-id 20260713T131720Z \
  --pca-components 50 --umap-components 5 \
  --umap-neighbors 50 --umap-min-dist 0.0 \
  --hdbscan-min-cluster-size 20 --hdbscan-min-samples 20 \
  --stability-runs 10 \
  --run-id umap5-n50-mcs20-ms20-stab10
```

| Run | ARI | NMI |
|-----|-----|-----|
| A, 100 seeds | 0.45 ± 0.37 | 0.55 ± 0.29 |
| D, 10 seeds | 0.51 ± 0.41 | 0.58 ± 0.33 |
| 5-D, 10 seeds | 0.51 ± 0.41 | 0.58 ± 0.32 |

Do not retune UMAP for a higher ARI. Keep A’s seed-42 labels; use `--cluster-space pca --method kmeans --n-clusters 40` (`kmeans-k40-vitl`) when a seed-proof full-corpus partition is required.

#### Noise peel (Sweep A leftovers)

Refit PCA + 10-D UMAP on thumbnails with `cluster_id=-1` in the parent run. Knobs default to the parent manifest (A: `n_neighbors=15`, `min_dist=0`, `mcs=20`, `ms=20`, `eom`). Each round is a new folder; inspect grids before chaining `--rounds`.

```bash
python src/dinov3/cluster_cls_peel.py \
  --from-clusters-run-id sweep-A-n15-mcs20-ms20 --dry-run

python src/dinov3/cluster_cls_peel.py \
  --from-clusters-run-id sweep-A-n15-mcs20-ms20

python src/dinov3/cluster_cls_peel.py \
  --from-clusters-run-id sweep-A-n15-mcs20-ms20 --rounds 3
```

Writes `data/dinov3_cls_clusters/sweep-A-n15-mcs20-ms20-r1/` (same files as `cluster_cls.py`) and `data/dinov3_cls_clusters/sweep-A-n15-mcs20-ms20-peels/combined_assignments.csv` (`cluster_id` offset across rounds, plus `round` / `cluster_id_in_round`). Remaining noise stays `-1`. UMAP coordinates on peeled rows are from that round’s map, not A’s.

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

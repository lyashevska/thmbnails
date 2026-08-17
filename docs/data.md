# Data

This project uses several linked data artifacts across sampling, thumbnail retrieval, annotation, and DINOv3 feature extraction.

## Core tables and files

- `data/data2008-2024.csv` - source metadata table used for sampling
- `data/sampled_data.csv` - year-stratified sample before thumbnail retrieval (current: 12,500 rows, 2,500/year for 2020–2024)
- `data/sampled_with_thumbnails.csv` - sample augmented with thumbnail URLs, local paths, and `thumbnail_success`
- `data/thumbnails/` - downloaded thumbnail images (`{viewkey}.jpg`; current run: 10,957 files on disk)
- `data/scraper.log` - append-only scraper run log (checkpoints, failures, final success rate)

### Thumbnail retrieval summary (current run)

| Metric | Count |
|--------|-------|
| Sampled | 12,500 |
| Retrieved (HTTP success) | 10,957 (87.7%) |
| Not retrieved | 1,543 (12.3%) |
| Placeholder (< 4 KB) | 2,291 (18.3% of sample) |
| **Valid for analysis** (≥ 4 KB) | **8,666 (69.3%)** |

Per-year valid counts: 2020 → 1,639; 2021 → 1,655; 2022 → 1,660; 2023 → 1,734; 2024 → 1,978. See [results.md](results.md) for the full table.

## Annotation outputs

- `data/annotations_ollama/` - per-image JSON outputs from the Ollama VLM workflow
- `data/annotations_ollama.jsonl` - run-level log with success flags and truncated raw output

## DINOv3 outputs

Two tracks, each with embeddings then clusters. See [dinov3_runs.md](dinov3_runs.md) for active run IDs.

| Path | Contents |
|------|----------|
| `data/dinov3_cls_embeddings/<run_id>/` | CLS vectors: `vectors/<image_id>.npy`, `cls_embeddings.npy`, `image_ids.json`, `manifest.json` |
| `data/dinov3_cls_clusters/<run_id>/` | One label per thumbnail: `cluster_assignments.csv`, `cluster_summary.csv`, `umap.png`, `samples/` |
| `data/dinov3_patch_embeddings/<run_id>/` | Patch vectors: `vectors/<image_id>.npz`, `image_ids.json`, `manifest.json` |
| `data/dinov3_patch_clusters/<run_id>/` | One label per patch: `patch_assignments.csv`, `cluster_summary.csv`, `image_cluster_histogram.csv`, `patch_crops/` |

The preprocessing and validation scripts use the same valid-thumbnail criteria as the analysis pipeline: readable image, at least 4 KB in file size, and 640×360 source dimensions.

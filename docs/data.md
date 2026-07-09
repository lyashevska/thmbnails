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

DINOv3 feature extraction writes run-specific outputs to `data/dinov3_embeddings/<run_id>/`:

- `vectors/<image_id>.npy` - per-image CLS embeddings used for resume-friendly checkpointing
- `cls_embeddings.npy` - stacked matrix of embeddings for the full run
- `image_ids.json` - image order corresponding to the stacked matrix
- `manifest.json` - run metadata, model ID, and shape information

The preprocessing and validation scripts use the same valid-thumbnail criteria as the analysis pipeline: readable image, at least 4 KB in file size, and 640×360 source dimensions.

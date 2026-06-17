# Data

This project uses several linked data artifacts across sampling, thumbnail retrieval, annotation, and DINOv3 feature extraction.

## Core tables and files

- `data/data2008-2024.csv` - source metadata table used for sampling
- `data/sampled_data.csv` - year-stratified sample before thumbnail retrieval
- `data/sampled_with_thumbnails.csv` - sample augmented with thumbnail URLs and local paths
- `data/thumbnails/` - downloaded thumbnail images

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

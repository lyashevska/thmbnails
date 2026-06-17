Thumbnail study

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Current VLM Workflow (Ollama)

Primary script: src/vlm_annotate.py

The current workflow uses a local Ollama-served VLM to annotate thumbnails with prompt-guided JSON output.

### 1) Start Ollama and pull model

```bash
ollama serve
ollama pull huihui_ai/qwen3-vl-abliterated:4b-instruct
```

### 2) Run pilot batch

```bash
python src/vlm_annotate.py --dry-run --limit 10
python src/vlm_annotate.py --limit 3 --force
```

### 3) What the script does

- Reads rows from data/sampled_with_thumbnails.csv
- Uses thumbnail_path + title for each request
- Skips thumbnails smaller than 4 KB
- Sends prompt from prompt + image + title to Ollama
- Tries to parse JSON from the model response
- Saves parseable JSON outputs and logs failures with raw text

### 4) Outputs

- data/annotations_ollama/<image_id>.json
- data/annotations_ollama/<image_id>.raw.txt (only when parse fails)
- data/annotations_ollama.jsonl (run metadata, success flag, truncated raw output)

### 5) Notes on quality

- Current runner uses minimal postprocessing by design.
- Some outputs may deviate from the intended schema in prompt.
- For pilot runs, spot-check files manually before scaling.

## DINOv3 Visual Feature Extraction

### Overview

DINOv3 is a self-supervised vision transformer model used for extracting robust visual features from thumbnails. This pipeline supports visual clustering, content-based deduplication, and feature analysis without requiring labels.

### Why DINOv3?

- **Domain-sensitive**: Captures fine-grained distinctions in poses, body types, acts, angles, nudity levels, and quality tiers
- **Self-supervised**: No labeled data required; learns universal visual features
- **Flexible**: Use CLS token for global clustering or patch tokens for dense analysis
- **Efficient**: Fast inference even on consumer hardware for medium datasets (2k–10k images)

### Preprocessing Pipeline

The DINOv3 preprocessing pipeline handles image validation and normalization:

**Location**: [src/dinov3/preprocess.py](src/dinov3/preprocess.py)

**Pipeline steps**:
1. **Filter invalid / placeholder files** – Checks file size (minimum 4 KB by default) and dimensions (640×360 for thumbnails)
2. **Convert to RGB** – Handles various image formats
3. **Letterbox to square** – Preserves 16:9 composition without distortion; fills padding with black (0, 0, 0)
4. **Resize to model input** – 224px for CLS token, 518px for patch token extraction

**Valid thumbnail criteria** (from [src/dinov3/preprocess.py](src/dinov3/preprocess.py)):
- File exists and is readable
- Size ≥ 4096 bytes
- Dimensions = 640×360 px

Outputs: PIL Image objects ready for DINOv3 model inference.

### Model Selection & Hardware

ViT-L/16 distilled (300M params)

| Model | Parameters | VRAM | Use Case |
|-------|-----------|------|----------|
| ViT-S/S+ | 21–50M | 1–3 GB | Rapid prototyping, CPU-friendly |
| ViT-B/ConvNeXt Base | 86–89M | 3–6 GB | Laptop; good balance |
| **ViT-L/ConvNeXt Large** | **198–300M** | **8–12 GB** | Best quality/speed tradeoff |
| ViT-H+ | 840M | 20–30+ GB | High-end machines only |

**Estimated runtime** (ViT-L, GPU):
- 2,000 images: 3–8 minutes
- 10,000 images: 15–40 minutes

### Embedding Extraction

Install DINOv3 dependencies (requires a [Hugging Face token](https://huggingface.co/settings/tokens) with access to gated DINOv3 models):

```bash
pip install -r requirements.txt
huggingface-cli login
```

Preview letterbox preprocessing:

```bash
python src/inspect_dinov3_preprocess.py --limit 10 --seed 42
```

Extract CLS embeddings for valid thumbnails. The script is resumable and skips any already written `vectors/<image_id>.npy` files unless you pass `--force`:

```bash
# Dry run: show what would be processed
python src/dinov3/extract_embeddings.py --dry-run --limit 10

# Small real run
python src/dinov3/extract_embeddings.py --limit 5

# Full run
python src/dinov3/extract_embeddings.py

# Resume into an existing run directory
python src/dinov3/extract_embeddings.py --run-id 20260616T160746Z
```

Validate the results after extraction. This checks that the matrix and ID list match, that embeddings are finite, and prints vector norms plus random cosine similarities:

```bash
python src/dinov3/check_embeddings.py --run-id <run_id>
python src/dinov3/check_embeddings.py --run-dir data/dinov3_embeddings/<run_id>
```

A healthy validation run should print:
- `Embeddings shape: (N, 768)` for ViT-B/16 CLS vectors
- `Image IDs: N`
- finite vector norms with no NaN/Inf errors
- `OK: basic embedding checks passed.`

Outputs under `data/dinov3_embeddings/<run_id>/`:
- `vectors/<image_id>.npy` — per-image CLS vectors (resume checkpoints)
- `cls_embeddings.npy` — stacked matrix `(N, D)`
- `image_ids.json` — row order
- `manifest.json` — model and run metadata


# Methods

## Data sampling

To obtain a balanced temporal sample from the dataset, `src/sample_per_year.py` performs stratified sampling by year. The input dataset is loaded from `data/data2008-2024.csv` as a pandas DataFrame; years are derived from the `date` column (YYYY-MM-DD). Only the **latest five years present in the source file** are kept (currently 2020–2024).

The current run samples **2,500 videos per year** (`n = 2500` in `src/sample_per_year.py`), producing **12,500 rows** in `data/sampled_data.csv`. An earlier pilot used 500 per year (2,500 total).

## Thumbnail Acquisition

Thumbnails corresponding to each video in the sampled dataset were retrieved using a **hybrid strategy** combining two complementary approaches.

### Direct CDN Construction (Strategy 1)
The first approach attempted to construct the thumbnail URL directly from the video’s unique `viewkey` extracted from the `url` column. Using known patterns of Pornhub’s Content Delivery Network (CDN), a direct image URL was generated (e.g., `https://ci.phncdn.com/videos/{viewkey[:3]}/{viewkey[3:6]}/{viewkey[6:9]}/{viewkey}/thumbnail.jpg`). The image was then downloaded via the `requests` library and saved locally in the `data/thumbnails/` directory as `{viewkey}.jpg`. This method was computationally lightweight and fast, requiring only a single HTTP request per video. However, it proved highly fragile due to frequent changes in Pornhub’s CDN structure, resulting in low reliability across runs.

### Open Graph Meta Tag Extraction (Strategy 2) 
Due to the instability of the direct CDN method, a second, more robust approach was ultimately adopted as the **primary method**. For each video, the full video page was fetched using the `requests` library. The representative thumbnail URL was then extracted from the `<meta property="og:image">` tag using the BeautifulSoup4 HTML parser. The identified image was subsequently downloaded and stored in the `data/thumbnails/` directory as `{viewkey}.jpg`.  

A fixed delay of **5.0 seconds** is enforced between requests (`delay` in `src/scraper.py`) to reduce rate limiting and server load. An earlier pilot used 2.0 seconds.

For the final dataset, three additional columns were appended:  
- `thumbnail_url` (remote image location)  
- `thumbnail_path` (local relative path to the downloaded file)  
- `thumbnail_success` (boolean indicating successful acquisition)  

This explicit linkage enables straightforward correspondence between textual video metadata and its visual thumbnail. The enriched dataset is exported as `data/sampled_with_thumbnails.csv`. Run progress and per-video failures are appended to `data/scraper.log`.

Across the current **12,500-video sample** (2,500 per year, 2020–2024), the hybrid CDN → OG fallback scraper achieved an HTTP retrieval success rate of **87.7%** (10,957/12,500). See [results.md](results.md) for per-year breakdown and placeholder counts.

### Resume and Checkpointing

The scraper supports stopping and resuming without losing progress (`src/scraper.py`).

On startup:
- If `sampled_with_thumbnails.csv` exists, it is loaded as the working dataset and rows with `thumbnail_success=True` are skipped.
- Any new rows in `sampled_data.csv` that are not yet in the output file are merged in.
- **Disk reconciliation**: if `{viewkey}.jpg` already exists in `data/thumbnails/` but the CSV still marks the row as failed (e.g. after a partial rsync or crash), the row is marked successful without re-downloading.

During the run, progress is flushed to `sampled_with_thumbnails.csv` every 10 rows (`CHECKPOINT_EVERY`), so an interruption loses at most 10 rows of CSV state. Thumbnail files written to disk are kept regardless.

### Placeholder and invalid thumbnails

Some videos were disabled or still processing at scrape time. These either return **no image** (counted as not retrieved) or Pornhub’s generic placeholder (“This video is still converting”). Placeholders are small JPEGs, typically **1.9–3.7 KB**; downstream pipelines treat files **< 4 KB** as invalid (see `src/dinov3/preprocess.py` and `src/vlm_annotate.py`). Retrieved placeholders are excluded from embedding and VLM analysis even though `thumbnail_success=True` in the acquisition CSV.

## Visual Annotation with Ollama VLM (Current Workflow)

After thumbnail acquisition, we perform structured visual analysis using a local Ollama-served VLM with a prompt-driven batch annotator.

### Rationale
Manual annotation of 1700+ thumbnails is infeasible for a small team. A local VLM workflow provides rapid first-pass coding with reproducible prompt instructions while keeping the process auditable and resumable.

### Method
- Input: local JPG + associated video `title` from the CSV (title is included because many thumbnails contain overlaid text and category cues).
- Prompt: system prompt loaded from `prompt`, plus strict instruction to return only one JSON object.
- Generation: Ollama chat with low temperature (default `0.05`) and `format="json"`.
- Parsing: lightweight extraction of the outer JSON object from model text. If parsing fails, raw model output is saved for review.
- Filtering: current runner skips thumbnails smaller than 4 KB before inference.
- Storage: one pretty-printed JSON per successful image in `data/annotations_ollama/` and an append-only run log in `data/annotations_ollama.jsonl` containing metadata, success flags, and truncated raw output.

### Schema (current)
The intended schema is defined in the `prompt` file. Top-level fields are expected to include:
- `image_id`, `title`, `framing`, `overall_composition`
- `body_display`, `sexual_acts`, `text_direct_address` (each: `{signifiers: string[], signified: string}`)
- `analytical_observations`: `{body_display, power_dynamics, visual_conventions}`
- `quantitative_tags`: categorical fields + free `key_stereotypes` list


### Implementation
Primary current script: `src/vlm_annotate.py`

CLI supports pilot and resumable runs via `--limit`, `--force`, `--dry-run`, and configurable input/output paths (`--csv`, `--out-dir`, `--results`, `--prompt`, `--model`). Existing per-image JSON files are skipped unless `--force` is set.

Optional advanced path: `src/analyze_thumbnails.py` (Transformers/Hugging Face backend) remains available for stricter or larger-scale runs.

### Quality control & limitations
- The model is prompted to ground claims in visible pixels + title.
- Current script intentionally uses minimal postprocessing; some outputs may deviate from the target schema and should be spot-checked.
- All VLM outputs remain interpretive and should be treated as first-pass coding for later human validation.
- Racial/gender categories in `key_stereotypes` should be interpreted cautiously and only when visually cued or explicitly titled.
- Placeholder/corrupted images are filtered upstream (file size + existence), but edge cases can still occur.

### Re-running or changing models
Because outputs are cached per image_id, swapping models or prompt versions only requires re-processing the delta (or using `--force`).

Future work may include (a) human inter-annotator agreement study on a stratified subsample, (b) fine-tuning or prompt optimization on the hand-coded examples, (c) aggregation scripts producing year-wise stereotype prevalence tables.

## DINOv3 Visual Feature Extraction

The project also includes a DINOv3-based embedding pipeline for thumbnail-level visual feature extraction. This path is useful for clustering, similarity search, and other label-free analyses where a dense visual representation is preferred over schema-based annotation.

### Method
- Input: local thumbnail JPGs from `data/thumbnails/` or thumbnail paths recorded in the CSV.
- Filtering: the pipeline keeps only valid thumbnails, using the same minimum file-size check as the VLM workflow and requiring 640×360 source thumbnails.
- Preprocessing: thumbnails are converted to RGB, letterboxed to a square canvas, and resized to the model input size before inference.
- Extraction: `src/dinov3/extract_embeddings.py` loads a Hugging Face DINOv3 checkpoint and writes one CLS embedding per image.
- Validation: `src/dinov3/check_embeddings.py` verifies that the stacked embedding matrix, image-id list, and model metadata are consistent.

### Outputs
Embeddings are written under `data/dinov3_embeddings/<run_id>/` and include:
- `vectors/<image_id>.npy` for resume-friendly per-image checkpoints
- `cls_embeddings.npy` for the stacked embedding matrix
- `image_ids.json` for row order
- `manifest.json` for run metadata

### Interpretation
DINOv3 embeddings are not human-readable labels. They are continuous feature vectors that can be compared with cosine similarity or used as input to downstream clustering. Higher cosine similarity indicates more similar visual content.
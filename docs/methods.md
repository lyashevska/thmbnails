# Methods

## Data sampling
To obtain a balanced temporal sample from the dataset, a custom script was developed to perform a stratified sample by year. The input dataset was loaded as a pandas DataFrame from a CSV file containing video metadata with a date column in YYYY-MM-DD format.

## Thumbnail Acquisition

Thumbnails corresponding to each video in the sampled dataset were retrieved using a **hybrid strategy** combining two complementary approaches.

### Direct CDN Construction (Strategy 1)
The first approach attempted to construct the thumbnail URL directly from the video’s unique `viewkey` extracted from the `url` column. Using known patterns of Pornhub’s Content Delivery Network (CDN), a direct image URL was generated (e.g., `https://ci.phncdn.com/videos/{viewkey[:3]}/{viewkey[3:6]}/{viewkey[6:9]}/{viewkey}/thumbnail.jpg`). The image was then downloaded via the `requests` library and saved locally in the `data/thumbnails/` directory as `{viewkey}.jpg`. This method was computationally lightweight and fast, requiring only a single HTTP request per video. However, it proved highly fragile due to frequent changes in Pornhub’s CDN structure, resulting in low reliability across runs.

### Open Graph Meta Tag Extraction (Strategy 2) 
Due to the instability of the direct CDN method, a second, more robust approach was ultimately adopted as the **primary method**. For each video, the full video page was fetched using the `requests` library. The representative thumbnail URL was then extracted from the `<meta property="og:image">` tag using the BeautifulSoup4 HTML parser. The identified image was subsequently downloaded and stored in the `data/thumbnails/` directory as `{viewkey}.jpg`.  

A fixed delay of 2.0 seconds was enforced between requests in both strategies to avoid excessive server load and respect server constraints.  

For the final dataset, three additional columns were appended:  
- `thumbnail_url` (remote image location)  
- `thumbnail_path` (local relative path to the downloaded file)  
- `thumbnail_success` (boolean indicating successful acquisition)  

This explicit linkage enables straightforward correspondence between textual video metadata and its visual thumbnail. The enriched dataset was exported as `sampled_with_thumbnails.csv`. A success rate of **89.1%** was achieved using the hybrid CDN → OG fallback approach across 2500 sampled videos (500 per year, 2020–2024).

### Resume and Checkpointing

The scraper supports stopping and resuming without losing progress. On startup, if `sampled_with_thumbnails.csv` already exists, it is loaded as the working dataset and previously successful rows are skipped entirely — no repeat network requests are made. Additionally, if a thumbnail image file is found on disk but the CSV was not yet flushed (e.g. due to a crash), the row is recovered from disk rather than re-downloaded.

Progress is flushed to `sampled_with_thumbnails.csv` every 10 rows during the run, so an interruption (network failure, manual stop, system crash) loses at most 10 rows of work. The interval is controlled by the `CHECKPOINT_EVERY` constant in `src/scraper.py`.

**Note on Data Limitations**  
Some videos in the dataset were disabled or still processing at the time of scraping. These videos either contained **no thumbnail** or returned Pornhub’s generic placeholder image (“This video is still converting”). These issues will be addressed in a subsequent phase of data cleaning and validation.

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
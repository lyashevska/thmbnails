# Results

## Thumbnail Retrieval

A total of **12,500 videos** were sampled (**2,500 per year**, 2020–2024) and processed by the hybrid CDN → OG fallback scraper (`src/scraper.py`). The overall HTTP retrieval success rate was **87.7%** (10,957/12,500). A subset of retrieved files are **placeholder thumbnails** (file size < 4 KB), typically corresponding to disabled videos or content still processing at scrape time.

An earlier pilot on the same year range sampled 500 per year (2,500 total) and achieved **89.1%** retrieval (2,227/2,500). The larger run shows a similar but slightly lower retrieval rate, with more failures concentrated in 2023–2024.

### Valid thumbnails per year

A thumbnail was classified as **retrieved** if `{viewkey}.jpg` exists on disk (equivalent to `thumbnail_success=True` after disk reconciliation). A thumbnail was classified as **valid** if retrieved and file size ≥ **4 KB** (the threshold used by DINOv3 preprocessing and the VLM annotator). All placeholders in this run were below 3.7 KB, so the earlier 3.6 KB threshold yields the same counts.

| Year | Sampled | Retrieved | Not Retrieved (%) | Placeholder <4KB (%) | **Valid** | **Valid %** |
|------|---------|-----------|-------------------|----------------------|-----------|-------------|
| 2020 | 2,500 | 2,224 | 276 (11.0%) | 585 (23.4%) | **1,639** | **65.6%** |
| 2021 | 2,500 | 2,249 | 251 (10.0%) | 594 (23.8%) | **1,655** | **66.2%** |
| 2022 | 2,500 | 2,178 | 322 (12.9%) | 518 (20.7%) | **1,660** | **66.4%** |
| 2023 | 2,500 | 2,173 | 327 (13.1%) | 439 (17.6%) | **1,734** | **69.4%** |
| 2024 | 2,500 | 2,133 | 367 (14.7%) | 155 (6.2%) | **1,978** | **79.1%** |
| **Total** | **12,500** | **10,957** | **1,543 (12.3%)** | **2,291 (18.3%)** | **8,666** | **69.3%** |

Placeholder counts in the table are expressed as a percentage of **sampled** videos per year. Among **retrieved** files only, the placeholder share is roughly **21%** in 2020–2021, **16–24%** in 2022–2023, and **7%** in 2024.

### Observations

- The overall **valid thumbnail rate** was **69.3%** (8,666/12,500), comparable to the pilot’s **69.7%** (1,743/2,500).
- **Not retrieved** (12.3% overall) increased slightly versus the pilot (10.9%), especially in 2024 (14.7%).
- **Placeholder images** remain the largest source of invalid data in older years: about **24%** of the 2020 and 2021 samples. This likely reflects videos disabled or removed after the source metadata was collected.
- The placeholder rate declines from **23–24%** (2020–2021) to **6.2%** (2024); newer uploads are less often replaced by the generic “still converting” image.
- **2024** has the highest valid rate (**79.1%**) despite the highest not-retrieved rate (14.7%), because few 2024 retrievals are placeholders.
- **8,666 valid thumbnails** are available for downstream DINOv3 and VLM workflows; not-retrieved rows and placeholders are filtered out in preprocessing.

## Visual Analysis Pipeline (Current Ollama Workflow)

A dedicated VLM annotation stage is currently run through `src/vlm_annotate.py`.

- Prompt-driven JSON annotation using a local Ollama-served VLM.
- Resumable behavior via per-image output checks (`--force` to rerun).
- Auditable logs through per-image JSON, `.raw.txt` failure sidecars, and JSONL run metadata.
- Prompt includes the target schema and research framing from `prompt`.
- Hand-curated reference examples in `docs/reference_annotations/` are retained for calibration and later human validation.

Initial reference annotations (manually authored for prompt calibration and evaluation targets) cover:
- A commercial "slutty teen DP threesome" gonzo thumbnail (high genital salience, male-dominant, brand text).
- An "exotic Asia" orientalist solo (racialized fantasy, jewelry/body adornment codes, low genital focus).

Current output locations:
- `data/annotations_ollama/<image_id>.json` (parseable model JSON)
- `data/annotations_ollama/<image_id>.raw.txt` (raw text when JSON parse fails)
- `data/annotations_ollama.jsonl` (run log with metadata and success flags)

For current Ollama annotation runs, thumbnails smaller than 4 KB are skipped before inference.

This current workflow is optimized for fast pilot iteration and traceability.

## DINOv3 Embedding Checks

Embedding validation is performed with `src/dinov3/check_embeddings.py` after a run finishes. A healthy run should report:

- matching counts for the embedding matrix and `image_ids.json`
- finite vector norms with no NaN or Inf values
- a CLS embedding shape of `(N, 768)` for the ViT-B/16 checkpoint used in the current pipeline
- random cosine similarities that are finite and generally higher for visually closer thumbnails

This check is intended as a sanity test, not a formal quality benchmark. With only a few samples, the cosine similarities mainly confirm that the pipeline is producing plausible feature vectors.

## DINOv3 Usage Notes

For exploratory analysis, the embeddings can be used for:

- nearest-neighbor similarity inspection
- duplicate or near-duplicate detection
- clustering and UMAP-style visual exploration

The current pipeline keeps per-image vectors on disk so interrupted runs can be resumed without recomputing completed images.
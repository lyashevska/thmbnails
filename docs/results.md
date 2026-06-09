# Results

## Thumbnail Retrieval

A total of 2500 videos were sampled (500 per year, 2020–2024) and processed by the hybrid CDN → OG fallback scraper. The overall HTTP retrieval success rate was **89.1%** (2227/2500). However, a subset of retrieved images were found to be invalid placeholder thumbnails (file size < 3.6 KB), typically corresponding to disabled videos or content still processing at the time of scraping.

### Valid thumbnails per year

A thumbnail was classified as **valid** if it was successfully retrieved and its file size was ≥ 3.6 KB.

| Year | Sampled | Retrieved | Not Retrieved (%) | Placeholder <3.6KB (%) | **Valid** | **Valid %** |
|------|---------|-----------|-------------------|------------------------|-----------|-------------|
| 2020 | 500 | 454 | 46 (9.2%) | 132 (26.4%) | **322** | **64.4%** |
| 2021 | 500 | 466 | 34 (6.8%) | 131 (26.2%) | **335** | **67.0%** |
| 2022 | 500 | 438 | 62 (12.4%) | 97 (19.4%) | **341** | **68.2%** |
| 2023 | 500 | 440 | 60 (12.0%) | 93 (18.6%) | **347** | **69.4%** |
| 2024 | 500 | 429 | 71 (14.2%) | 31 (6.2%) | **398** | **79.6%** |
| **Total** | **2500** | **2227** | **273 (10.9%)** | **484 (19.4%)** | **1743** | **69.7%** |

### Observations

- The overall valid thumbnail rate was **69.7%** (1743/2500).
- **Placeholder images** (retrieved but too small to be real thumbnails) were the dominant source of invalid data in older years, accounting for 26% of the 2020 and 2021 samples. This likely reflects videos that were disabled or removed after the original dataset was collected.
- The placeholder rate declined consistently from 26.4% in 2020 to 6.2% in 2024, suggesting that more recent videos are less likely to have been taken down.
- **2024** had the highest valid thumbnail rate (79.6%) despite also having the highest not-retrieved rate (14.2%), because very few 2024 videos returned placeholder images.
- Invalid thumbnails (not retrieved + placeholder) will be excluded from subsequent visual analysis.

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

Note: the retrieval table above uses the earlier 3.6 KB validity threshold from the acquisition stage.

This current workflow is optimized for fast pilot iteration and traceability.
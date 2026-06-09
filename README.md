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

## Optional Advanced Path

Optional stricter backend: src/analyze_thumbnails.py (Transformers/Hugging Face).

Use this if you want tighter schema control and a heavier research pipeline setup.

## Reference Annotations

Hand-authored calibration examples are in docs/reference_annotations:

- docs/reference_annotations/ph5f89a9c6adc3d.json
- docs/reference_annotations/661b4074d08da.json

These are useful for prompt calibration and manual quality checks.
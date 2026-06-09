#!/usr/bin/env python3
"""
Batch thumbnail annotation via Ollama VLM.

This script:
- Loads your base system prompt from local "prompt" file.
- Reads title + thumbnail paths from a CSV.
- Calls Ollama VLM per thumbnail.
- Saves model JSON output per image and a JSONL run log.

Examples:
    python src/vlm_annotate.py --limit 5
    python src/vlm_annotate.py --csv data/sampled_with_thumbnails.csv --force
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    import ollama
except ImportError:
    print("Please run: pip install ollama")
    sys.exit(1)


MODEL = "huihui_ai/qwen3-vl-abliterated:4b-instruct"
PROMPT_FILE = Path("prompt")

USER_TEMPLATE = """Video title: {title}
Image filename: {image_id}

Analyze the attached thumbnail image according to the instructions. Output ONLY the JSON object."""


def load_prompt_from_file() -> str:
    """Load user prompt and add strict output rules."""
    if not PROMPT_FILE.exists():
        print(f"Warning: {PROMPT_FILE} not found. Using fallback prompt.")
        base = (
            "You are an expert cultural studies researcher analyzing stereotypes "
            "in pornographic visual culture."
        )
    else:
        base = PROMPT_FILE.read_text(encoding="utf-8").strip()
        print(f"Loaded base prompt from: {PROMPT_FILE}")

    return base + """

IMPORTANT OUTPUT REQUIREMENTS:
- Return ONLY one valid JSON object.
- No markdown fences, no extra commentary.
- Follow the required schema exactly.
"""


def is_valid_thumbnail(path: Path, min_bytes: int = 4096) -> bool:
    if not path.exists():
        return False
    try:
        return path.stat().st_size >= min_bytes
    except OSError:
        return False


def load_rows(csv_path: Path, thumb_dir_fallback: Path) -> List[Dict[str, str]]:
    df = pd.read_csv(csv_path)
    rows: List[Dict[str, str]] = []

    for _, row in df.iterrows():
        tpath = row.get("thumbnail_path")
        if pd.isna(tpath):
            continue

        p = Path(str(tpath))
        if not p.is_absolute():
            p = Path.cwd() / p

        if not is_valid_thumbnail(p):
            alt = thumb_dir_fallback / p.name
            if not is_valid_thumbnail(alt):
                continue
            p = alt

        title = str(row.get("title", "")) if not pd.isna(row.get("title")) else ""
        rows.append({"image_id": p.name, "title": title, "thumbnail_path": str(p)})

    return rows


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text or not text.strip():
        return None
    s = text.strip()

    if s.startswith("```"):
        parts = s.split("```", 2)
        if len(parts) >= 2:
            s = parts[1].strip()
            if s.lower().startswith("json"):
                s = s[4:].strip()

    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = s[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def request_annotation(
    model: str,
    system_prompt: str,
    image_path: Path,
    image_id: str,
    title: str,
    temperature: float,
) -> Tuple[Optional[Dict[str, Any]], str]:
    user_text = USER_TEMPLATE.format(title=title, image_id=image_id)
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_text,
                "images": [str(image_path)],
            },
        ],
        format="json",
        options={"temperature": temperature, "top_p": 0.9},
    )

    raw = response["message"]["content"].strip()
    return extract_json_from_text(raw), raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch Ollama VLM thumbnail annotator")
    parser.add_argument("--csv", default="data/sampled_with_thumbnails.csv")
    parser.add_argument("--thumb-dir", default="data/thumbnails")
    parser.add_argument("--prompt", default="prompt")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--out-dir", default="data/annotations_ollama")
    parser.add_argument("--results", default="data/annotations_ollama.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    global PROMPT_FILE
    PROMPT_FILE = Path(args.prompt)

    csv_path = Path(args.csv)
    thumb_dir = Path(args.thumb_dir)
    out_dir = Path(args.out_dir)
    results_path = Path(args.results)

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path, thumb_dir)
    if args.limit:
        rows = rows[: args.limit]

    to_process: List[Dict[str, str]] = []
    for row in rows:
        target = out_dir / f"{row['image_id']}.json"
        if args.force or not target.exists():
            to_process.append(row)

    print(f"Total valid rows: {len(rows)} | To process: {len(to_process)}")

    if args.dry_run:
        for row in to_process[:10]:
            print(f"- {row['image_id']}: {row['title'][:60]}")
        if len(to_process) > 10:
            print(f"... and {len(to_process) - 10} more")
        return

    if not to_process:
        print("Nothing to do. All selected images already have output files.")
        return

    system_prompt = load_prompt_from_file()
    success = 0
    failed = 0

    with open(results_path, "a", encoding="utf-8") as jl:
        for idx, row in enumerate(to_process, start=1):
            img_id = row["image_id"]
            title = row["title"]
            image_path = Path(row["thumbnail_path"])
            out_json = out_dir / f"{img_id}.json"
            raw_path = out_dir / f"{img_id}.raw.txt"

            if idx % 10 == 1 or idx == len(to_process):
                print(f"[{idx}/{len(to_process)}] Processing {img_id}")

            record: Dict[str, Any] = {
                "image_id": img_id,
                "title": title,
                "thumbnail_path": str(image_path),
                "timestamp": time.time(),
                "model": args.model,
                "success": False,
                "annotation": None,
                "raw_output": None,
            }

            try:
                parsed, raw = request_annotation(
                    model=args.model,
                    system_prompt=system_prompt,
                    image_path=image_path,
                    image_id=img_id,
                    title=title,
                    temperature=args.temp,
                )
                record["raw_output"] = raw[:2000] if raw else None
            except Exception as exc:
                parsed = None
                raw = f"REQUEST_ERROR: {exc}"
                record["raw_output"] = raw

            if parsed is None:
                raw_path.write_text(raw, encoding="utf-8")
                failed += 1
            else:
                with open(out_json, "w", encoding="utf-8") as f:
                    json.dump(parsed, f, indent=2, ensure_ascii=False)
                success += 1

            record["success"] = parsed is not None
            record["annotation"] = parsed

            jl.write(json.dumps(record, ensure_ascii=False) + "\n")
            jl.flush()

    print("\nRun complete")
    print(f"Success: {success}")
    print(f"Failed:  {failed}")
    print(f"JSON outputs: {out_dir.resolve()}")
    print(f"Run log: {results_path.resolve()}")


if __name__ == "__main__":
    main()

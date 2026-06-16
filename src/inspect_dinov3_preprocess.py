#!/usr/bin/env python3
"""
Visual inspection of DINOv3 letterbox preprocessing on random thumbnails.

Saves previews under data/dinov3_previews/<run_id>/:
  - <id>_letterbox_518.jpg  patch-resolution input
  - <id>_letterbox_224.jpg  CLS-resolution input
  - manifest.json           per-image metadata

Examples:
    python src/inspect_dinov3_preprocess.py
    python src/inspect_dinov3_preprocess.py --limit 10 --seed 42
    python src/inspect_dinov3_preprocess.py --include-placeholders
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image

# Allow running as: python src/inspect_dinov3_preprocess.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dinov3.preprocess import (  # noqa: E402
    DEFAULT_MIN_BYTES,
    DEFAULT_VALID_SIZE,
    is_valid_thumbnail,
    preprocess_for_dinov3,
)

THUMB_DIR = Path("data/thumbnails")
OUT_ROOT = Path("data/dinov3_previews")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Preview DINOv3 letterbox preprocessing on random thumbnails.")
    p.add_argument("--thumb-dir", type=Path, default=THUMB_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    p.add_argument("--limit", type=int, default=10, help="Number of random samples.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-bytes", type=int, default=DEFAULT_MIN_BYTES)
    p.add_argument(
        "--include-placeholders",
        action="store_true",
        help="Sample from all JPGs, not only valid 640x360 thumbnails.",
    )
    return p.parse_args()


def list_candidates(thumb_dir: Path, *, valid_only: bool, min_bytes: int) -> List[Path]:
    all_jpgs = sorted(thumb_dir.glob("*.jpg"))
    if not valid_only:
        return all_jpgs

    out: List[Path] = []
    for path in all_jpgs:
        ok, _ = is_valid_thumbnail(path, min_bytes=min_bytes, valid_size=DEFAULT_VALID_SIZE)
        if ok:
            out.append(path)
    return out


def run_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def process_one(path: Path, out_dir: Path, min_bytes: int) -> Dict[str, Any]:
    stem = path.stem
    ok, reason = is_valid_thumbnail(path, min_bytes=min_bytes)

    with Image.open(path) as raw:
        source_size = raw.size
        source_mode = raw.mode
    source_bytes = path.stat().st_size

    record: Dict[str, Any] = {
        "file": path.name,
        "valid": ok,
        "filter_reason": reason,
        "source_size": list(source_size),
        "source_mode": source_mode,
        "source_bytes": source_bytes,
    }

    if not ok:
        print(f"  SKIP {path.name}: {reason}")
        return record

    for target in (518, 224):
        result = preprocess_for_dinov3(path, target_size=target, min_bytes=min_bytes)
        out_path = out_dir / f"{stem}_letterbox_{target}.jpg"
        result.image.save(out_path, quality=92)
        record[f"letterbox_{target}"] = {
            "path": str(out_path),
            "square_before_resize": result.square_size,
            "pad_top_px": result.letterbox_pad_top,
            "output_size": list(result.image.size),
        }

    pad_top = record["letterbox_518"]["pad_top_px"]
    square = record["letterbox_518"]["square_before_resize"]
    print(
        f"  OK   {path.name}: {source_size[0]}x{source_size[1]} "
        f"-> letterbox {square}x{square} (pad_top={pad_top}) -> 518 / 224"
    )
    return record


def main() -> None:
    args = parse_args()

    if not args.thumb_dir.is_dir():
        print(f"Thumbnail directory not found: {args.thumb_dir}")
        sys.exit(1)

    valid_only = not args.include_placeholders
    candidates = list_candidates(args.thumb_dir, valid_only=valid_only, min_bytes=args.min_bytes)

    print("Step 1: Discover thumbnails")
    print(f"  dir={args.thumb_dir}")
    print(f"  valid_only={valid_only}  min_bytes={args.min_bytes}")
    print(f"  candidates={len(candidates)}")

    if not candidates:
        print("No candidates found.")
        sys.exit(1)

    print("\nStep 2: Random sample")
    random.seed(args.seed)
    n = min(args.limit, len(candidates))
    sample = random.sample(candidates, n)
    print(f"  seed={args.seed}  limit={args.limit}  selected={n}")
    for p in sample:
        print(f"    - {p.name}")

    run_id = run_id_now()
    out_dir = args.out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nStep 3: Letterbox preprocess and save previews")
    print(f"  out_dir={out_dir}")
    print("  pipeline: RGB -> letterbox -> resize (518, 224)")

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "seed": args.seed,
        "limit": args.limit,
        "valid_only": valid_only,
        "min_bytes": args.min_bytes,
        "valid_size": list(DEFAULT_VALID_SIZE),
        "strategy": "letterbox",
        "target_sizes": [518, 224],
        "images": [],
    }

    for path in sample:
        manifest["images"].append(process_one(path, out_dir, args.min_bytes))

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nStep 4: Write manifest")
    print(f"  {manifest_path}")
    print("\nDone. Open letterbox previews in the output directory.")


if __name__ == "__main__":
    main()
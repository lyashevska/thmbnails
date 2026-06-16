#!/usr/bin/env python3
"""
Batch-extract DINOv3 CLS embeddings for valid thumbnails.

Outputs under data/dinov3_embeddings/<run_id>/:
  vectors/<image_id>.npy   one CLS vector per thumbnail (resume-friendly)
  image_ids.json           ordered list of extracted IDs
  cls_embeddings.npy       stacked matrix (N, D), written at end
  manifest.json            run metadata

Examples:
    python src/dinov3/extract_embeddings.py --dry-run --limit 10
    python src/dinov3/extract_embeddings.py --limit 5
    python src/dinov3/extract_embeddings.py
    python src/dinov3/extract_embeddings.py --model facebook/dinov3-vitl16-pretrain-lvd1689m
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dinov3.config import (  # noqa: E402
    CSV_DEFAULT,
    DEFAULT_CLS_SIZE,
    DEFAULT_MODEL_ID,
    EMBEDDINGS_ROOT,
    THUMB_DIR_DEFAULT,
)
from src.dinov3.extract import extract_cls_from_path, load_dinov3  # noqa: E402
from src.dinov3.preprocess import DEFAULT_MIN_BYTES, is_valid_thumbnail  # noqa: E402

CHECKPOINT_EVERY = 25


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract DINOv3 CLS embeddings for thumbnails.")
    p.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    p.add_argument("--thumb-dir", type=Path, default=THUMB_DIR_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=EMBEDDINGS_ROOT)
    p.add_argument("--model", default=DEFAULT_MODEL_ID)
    p.add_argument("--cls-size", type=int, default=DEFAULT_CLS_SIZE)
    p.add_argument("--device", default=None, help="cuda, cpu, or auto (default)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min-bytes", type=int, default=DEFAULT_MIN_BYTES)
    p.add_argument("--force", action="store_true", help="Re-extract even if vector file exists.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run-id", default=None, help="Resume into an existing run directory.")
    return p.parse_args()


def run_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def load_rows(csv_path: Path, thumb_dir: Path, *, min_bytes: int) -> List[Dict[str, str]]:
    df = pd.read_csv(csv_path)
    rows: List[Dict[str, str]] = []

    for _, row in df.iterrows():
        tpath = row.get("thumbnail_path")
        if pd.isna(tpath):
            continue

        p = Path(str(tpath))
        if not p.is_absolute():
            p = Path.cwd() / p

        if not is_valid_thumbnail(p, min_bytes=min_bytes):
            alt = thumb_dir / p.name
            if not is_valid_thumbnail(alt, min_bytes=min_bytes):
                continue
            p = alt

        rows.append({"image_id": p.name, "thumbnail_path": str(p)})

    return rows


def load_completed_ids(vectors_dir: Path) -> List[str]:
    if not vectors_dir.is_dir():
        return []
    return sorted(p.stem for p in vectors_dir.glob("*.npy"))


def consolidate_vectors(vectors_dir: Path, image_ids: List[str]) -> np.ndarray:
    vectors = []
    for image_id in image_ids:
        path = vectors_dir / f"{image_id}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing vector for {image_id}: {path}")
        vectors.append(np.load(path))
    return np.stack(vectors, axis=0)


def save_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}")
        sys.exit(1)

    rows = load_rows(args.csv, args.thumb_dir, min_bytes=args.min_bytes)
    if args.limit:
        rows = rows[: args.limit]

    run_id = args.run_id or run_id_now()
    run_dir = args.out_dir / run_id
    vectors_dir = run_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    completed = set() if args.force else set(load_completed_ids(vectors_dir))
    to_process = [r for r in rows if r["image_id"] not in completed]

    print(f"Valid rows: {len(rows)} | Already done: {len(completed)} | To process: {len(to_process)}")
    print(f"Run dir: {run_dir}")

    if args.dry_run:
        for row in to_process[:10]:
            print(f"- {row['image_id']}")
        if len(to_process) > 10:
            print(f"... and {len(to_process) - 10} more")
        return

    if not to_process:
        print("Nothing to do.")
    else:
        try:
            bundle = load_dinov3(args.model, device=args.device, cls_size=args.cls_size)
        except OSError as exc:
            if "gated repo" in str(exc).lower():
                print(
                    "Cannot download the DINOv3 model. Accept the model license on Hugging Face, "
                    "then run: huggingface-cli login"
                )
            raise SystemExit(1) from exc
        print(f"Model: {bundle.model_id}  device: {bundle.device}  cls_size: {bundle.cls_size}")

        ok = 0
        failed = 0
        start = time.perf_counter()

        for i, row in enumerate(to_process, start=1):
            image_id = row["image_id"]
            path = Path(row["thumbnail_path"])
            out_path = vectors_dir / f"{image_id}.npy"
            try:
                embedding, _ = extract_cls_from_path(bundle, path, min_bytes=args.min_bytes)
                np.save(out_path, embedding)
                ok += 1
                print(f"[{i}/{len(to_process)}] OK {image_id}  dim={embedding.shape[0]}")
            except Exception as exc:
                failed += 1
                print(f"[{i}/{len(to_process)}] FAIL {image_id}: {exc}")

            if i % CHECKPOINT_EVERY == 0:
                ids = load_completed_ids(vectors_dir)
                save_manifest(
                    run_dir / "manifest.json",
                    _build_manifest(args, run_id, ids, ok, failed, partial=True),
                )
                print(f"  checkpoint ({len(ids)} vectors saved)")

        elapsed = time.perf_counter() - start
        print(f"Extraction finished: ok={ok} failed={failed} elapsed={elapsed:.1f}s")

    image_ids = load_completed_ids(vectors_dir)
    if not image_ids:
        print("No vectors extracted.")
        return

    stacked = consolidate_vectors(vectors_dir, image_ids)
    np.save(run_dir / "cls_embeddings.npy", stacked)
    (run_dir / "image_ids.json").write_text(json.dumps(image_ids, indent=2), encoding="utf-8")

    manifest = _build_manifest(args, run_id, image_ids, len(image_ids), 0, partial=False)
    manifest["embedding_shape"] = list(stacked.shape)
    manifest["embedding_dim"] = int(stacked.shape[1])
    save_manifest(run_dir / "manifest.json", manifest)

    print(f"Saved cls_embeddings.npy {stacked.shape} and {len(image_ids)} IDs")


def _build_manifest(
    args: argparse.Namespace,
    run_id: str,
    image_ids: List[str],
    ok: int,
    failed: int,
    *,
    partial: bool,
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "partial": partial,
        "model_id": args.model,
        "cls_size": args.cls_size,
        "min_bytes": args.min_bytes,
        "csv": str(args.csv),
        "thumb_dir": str(args.thumb_dir),
        "extracted_count": len(image_ids),
        "ok": ok,
        "failed": failed,
        "image_ids_sample": image_ids[:5],
    }


if __name__ == "__main__":
    main()
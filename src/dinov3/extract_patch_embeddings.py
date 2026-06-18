#!/usr/bin/env python3
"""
Extract and cache DINOv3 patch embeddings (224px by default).

Outputs under data/dinov3_patch_embeddings/<run_id>/:
  vectors/<image_id>.npz
  image_ids.json
  manifest.json

Examples:
    python src/dinov3/extract_patch_embeddings.py --dry-run --limit 10
    python src/dinov3/extract_patch_embeddings.py --limit 50
    python src/dinov3/extract_patch_embeddings.py --embeddings-run-id 20260617T091002Z
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dinov3.cluster import load_embedding_run, resolve_embeddings_run  # noqa: E402
from src.dinov3.config import (  # noqa: E402
    CSV_DEFAULT,
    DEFAULT_MODEL_ID,
    DEFAULT_PATCH_SIZE,
    PATCH_EMBEDDINGS_ROOT,
    THUMB_DIR_DEFAULT,
)
from src.dinov3.extract import extract_patches_from_path, load_dinov3  # noqa: E402
from src.dinov3.extract_embeddings import load_rows  # noqa: E402
from src.dinov3.patch_motifs import run_id_now, save_patch_vector  # noqa: E402
from src.dinov3.preprocess import DEFAULT_MIN_BYTES  # noqa: E402

CHECKPOINT_EVERY = 25


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract DINOv3 patch embeddings.")
    p.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    p.add_argument("--thumb-dir", type=Path, default=THUMB_DIR_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=PATCH_EMBEDDINGS_ROOT)
    p.add_argument("--embeddings-run-id", default=None, help="Reuse image_ids from a CLS embedding run.")
    p.add_argument("--model", default=DEFAULT_MODEL_ID)
    p.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min-bytes", type=int, default=DEFAULT_MIN_BYTES)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def image_ids_from_embeddings_run(run_id: str) -> List[str]:
    run_dir = resolve_embeddings_run(run_id=run_id)
    _, image_ids, _ = load_embedding_run(run_dir)
    return image_ids


def main() -> None:
    args = parse_args()

    if args.embeddings_run_id:
        image_ids = image_ids_from_embeddings_run(args.embeddings_run_id)
        rows = [{"image_id": i, "thumbnail_path": str(args.thumb_dir / i)} for i in image_ids]
    else:
        rows = load_rows(args.csv, args.thumb_dir, min_bytes=args.min_bytes)

    if args.limit:
        rows = rows[: args.limit]

    run_id = args.run_id or run_id_now()
    run_dir = args.out_dir / run_id
    vectors_dir = run_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)

    completed = set() if args.force else {p.name[:-4] for p in vectors_dir.glob("*.npz")}
    to_process = [r for r in rows if r["image_id"] not in completed]

    print(f"Images: {len(rows)} | Done: {len(completed)} | To process: {len(to_process)}")
    print(f"Patch size: {args.patch_size}px  run_dir: {run_dir}")

    if args.dry_run:
        for row in to_process[:10]:
            print(f"- {row['image_id']}")
        return

    if not to_process:
        print("Nothing to do.")
    else:
        bundle = load_dinov3(args.model, device=args.device, cls_size=args.patch_size)
        ok = failed = 0
        start = time.perf_counter()
        thumb_paths: Dict[str, str] = {}

        for i, row in enumerate(to_process, start=1):
            image_id = row["image_id"]
            path = Path(row["thumbnail_path"])
            out_path = vectors_dir / f"{image_id}.npz"
            try:
                result = extract_patches_from_path(
                    bundle,
                    path,
                    patch_size=args.patch_size,
                    min_bytes=args.min_bytes,
                )
                save_patch_vector(out_path, result, image_id)
                thumb_paths[image_id] = str(path)
                ok += 1
                print(
                    f"[{i}/{len(to_process)}] OK {image_id} "
                    f"patches={result.patches.shape[0]} grid={result.grid_shape}"
                )
            except Exception as exc:
                failed += 1
                print(f"[{i}/{len(to_process)}] FAIL {image_id}: {exc}")

            if i % CHECKPOINT_EVERY == 0:
                _write_manifest(run_dir, args, run_id, thumb_paths, ok, failed, partial=True)

        print(f"Done: ok={ok} failed={failed} elapsed={time.perf_counter() - start:.1f}s")

    image_ids = sorted(p.name[:-4] for p in vectors_dir.glob("*.npz"))
    (run_dir / "image_ids.json").write_text(json.dumps(image_ids, indent=2), encoding="utf-8")
    thumb_paths = _load_thumb_paths(run_dir, rows)
    for r in rows:
        if r["image_id"] in image_ids:
            thumb_paths.setdefault(r["image_id"], r["thumbnail_path"])
    _write_manifest(run_dir, args, run_id, thumb_paths, len(image_ids), 0, partial=False)


def _load_thumb_paths(run_dir: Path, rows: List[Dict[str, str]]) -> Dict[str, str]:
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing = data.get("thumbnail_paths", {})
        if existing:
            return existing
    return {r["image_id"]: r["thumbnail_path"] for r in rows}


def _write_manifest(
    run_dir: Path,
    args: argparse.Namespace,
    run_id: str,
    thumb_paths: Dict[str, str],
    ok: int,
    failed: int,
    *,
    partial: bool,
) -> None:
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "embedding_type": "patch",
        "partial": partial,
        "model_id": args.model,
        "patch_size": args.patch_size,
        "min_bytes": args.min_bytes,
        "extracted_images": len(thumb_paths),
        "ok": ok,
        "failed": failed,
        "thumbnail_paths": thumb_paths,
        "compare_with": "CLS embeddings in data/dinov3_embeddings/; CLS clusters in data/dinov3_clusters/",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
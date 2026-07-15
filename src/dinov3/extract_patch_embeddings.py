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
    python src/dinov3/extract_patch_embeddings.py --embeddings-run-id 20260713T131720Z
    python src/dinov3/extract_patch_embeddings.py --run-id <run_id>  # resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dinov3.cluster import load_embedding_run, resolve_embeddings_run  # noqa: E402
from src.dinov3.config import (  # noqa: E402
    CSV_DEFAULT,
    DEFAULT_MODEL_ID,
    DEFAULT_PATCH_SIZE,
    PATCH_EMBEDDINGS_ROOT,
    THUMB_DIR_DEFAULT,
    expected_cls_dim,
)
from src.dinov3.extract import extract_patches_from_path, load_dinov3  # noqa: E402
from src.dinov3.extract_embeddings import load_rows  # noqa: E402
from src.dinov3.patch_motifs import run_id_now, save_patch_vector  # noqa: E402
from src.dinov3.preprocess import DEFAULT_MIN_BYTES  # noqa: E402
from src.dinov3.timing import format_duration  # noqa: E402

CHECKPOINT_EVERY = 25


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract DINOv3 patch embeddings.")
    p.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    p.add_argument("--thumb-dir", type=Path, default=THUMB_DIR_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=PATCH_EMBEDDINGS_ROOT)
    p.add_argument(
        "--embeddings-run-id",
        default=None,
        help="Reuse image_ids from a CLS embedding run (recommended for ViT-L corpus).",
    )
    p.add_argument("--model", default=DEFAULT_MODEL_ID)
    p.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min-bytes", type=int, default=DEFAULT_MIN_BYTES)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--run-id", default=None, help="Resume into an existing run directory.")
    return p.parse_args()


def image_ids_from_embeddings_run(run_id: str) -> List[str]:
    run_dir = resolve_embeddings_run(run_id=run_id)
    _, image_ids, _ = load_embedding_run(run_dir)
    return image_ids


def load_completed_ids(vectors_dir: Path) -> List[str]:
    if not vectors_dir.is_dir():
        return []
    return sorted(p.stem for p in vectors_dir.glob("*.npz"))


def load_prior_counts(run_dir: Path) -> tuple[int, int]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return 0, 0
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return int(data.get("ok", 0)), int(data.get("failed", 0))


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

    completed = set() if args.force else set(load_completed_ids(vectors_dir))
    to_process = [r for r in rows if r["image_id"] not in completed]
    prior_ok, prior_failed = (0, 0) if args.force else load_prior_counts(run_dir)

    print(f"Images: {len(rows)} | Done: {len(completed)} | To process: {len(to_process)}")
    print(f"Patch size: {args.patch_size}px  run_dir: {run_dir}")
    if args.embeddings_run_id:
        print(f"CLS corpus: {args.embeddings_run_id}")

    if args.dry_run:
        for row in to_process[:10]:
            print(f"- {row['image_id']}")
        if len(to_process) > 10:
            print(f"... and {len(to_process) - 10} more")
        return

    model_load_seconds: Optional[float] = None
    inference_seconds: Optional[float] = None
    device_name: Optional[str] = None
    ok = prior_ok
    failed = prior_failed
    n_to_process = len(to_process)

    if not to_process:
        print("Nothing to do.")
    else:
        load_start = time.perf_counter()
        try:
            bundle = load_dinov3(args.model, device=args.device, cls_size=args.patch_size)
        except OSError as exc:
            if "gated repo" in str(exc).lower():
                print(
                    "Cannot download the DINOv3 model. Accept the model license on Hugging Face, "
                    "then run: huggingface-cli login"
                )
            raise SystemExit(1) from exc
        model_load_seconds = time.perf_counter() - load_start
        device_name = str(bundle.device)
        print(
            f"Model: {bundle.model_id}  device: {bundle.device}  patch_size: {args.patch_size}px  "
            f"load={model_load_seconds:.1f}s"
        )

        infer_start = time.perf_counter()
        run_ok = 0
        run_failed = 0

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
                ok += 1
                run_ok += 1
                print(
                    f"[{i}/{len(to_process)}] OK {image_id} "
                    f"patches={result.patches.shape[0]} grid={result.grid_shape}"
                )
            except Exception as exc:
                failed += 1
                run_failed += 1
                print(f"[{i}/{len(to_process)}] FAIL {image_id}: {exc}")

            if i % CHECKPOINT_EVERY == 0:
                image_ids = load_completed_ids(vectors_dir)
                manifest = _build_manifest(
                    args,
                    run_id,
                    image_ids,
                    ok,
                    failed,
                    partial=True,
                )
                manifest["model_timing"] = _model_timing_dict(
                    model_load_seconds=model_load_seconds,
                    inference_seconds=time.perf_counter() - infer_start,
                    device=device_name,
                    images_in_run=n_to_process,
                    ok=run_ok,
                )
                _save_manifest(run_dir, manifest)
                print(f"  checkpoint ({len(image_ids)} vectors saved)")

        inference_seconds = time.perf_counter() - infer_start
        print(
            f"Extraction finished: ok={run_ok} failed={run_failed} "
            f"inference={format_duration(inference_seconds)}"
        )

    image_ids = load_completed_ids(vectors_dir)
    if not image_ids:
        print("No patch vectors extracted.")
        return

    (run_dir / "image_ids.json").write_text(json.dumps(image_ids, indent=2), encoding="utf-8")

    manifest = _build_manifest(args, run_id, image_ids, ok, failed, partial=False)
    if model_load_seconds is not None or inference_seconds is not None:
        manifest["model_timing"] = _model_timing_dict(
            model_load_seconds=model_load_seconds,
            inference_seconds=inference_seconds,
            device=device_name,
            images_in_run=n_to_process,
            ok=ok - prior_ok if n_to_process else len(image_ids),
        )
    _save_manifest(run_dir, manifest)

    print(f"Saved {len(image_ids)} patch vector files under {vectors_dir}")
    if inference_seconds is not None:
        timing = manifest["model_timing"]
        print(
            f"Model timing: load={timing['model_load_seconds']}s  "
            f"inference={timing['inference_human']}  "
            f"{timing['seconds_per_image']}s/image"
        )


def _model_timing_dict(
    *,
    model_load_seconds: Optional[float],
    inference_seconds: Optional[float],
    device: Optional[str],
    images_in_run: int,
    ok: int,
) -> Dict[str, Any]:
    timing: Dict[str, Any] = {}
    if device:
        timing["device"] = device
    if model_load_seconds is not None:
        timing["model_load_seconds"] = round(model_load_seconds, 2)
    if inference_seconds is not None:
        timing["inference_seconds"] = round(inference_seconds, 2)
        timing["inference_human"] = format_duration(inference_seconds)
        if ok > 0:
            timing["seconds_per_image"] = round(inference_seconds / ok, 3)
    timing["images_processed_this_run"] = images_in_run
    timing["images_ok_this_run"] = ok
    return timing


def _build_manifest(
    args: argparse.Namespace,
    run_id: str,
    image_ids: List[str],
    ok: int,
    failed: int,
    *,
    partial: bool,
) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "embedding_type": "patch",
        "partial": partial,
        "model_id": args.model,
        "patch_size": args.patch_size,
        "embedding_dim": expected_cls_dim(args.model),
        "min_bytes": args.min_bytes,
        "csv": str(args.csv),
        "thumb_dir": str(args.thumb_dir),
        "extracted_count": len(image_ids),
        "ok": ok,
        "failed": failed,
        "image_ids_sample": image_ids[:5],
        "compare_with": "CLS embeddings in data/dinov3_embeddings/; CLS clusters in data/dinov3_clusters/",
    }
    if args.embeddings_run_id:
        manifest["embeddings_run_id"] = args.embeddings_run_id
    return manifest


def _save_manifest(run_dir: Path, manifest: Dict[str, Any]) -> None:
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
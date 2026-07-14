#!/usr/bin/env python3
"""
Sanity-check a DINOv3 patch embedding run.

Examples:
    python src/dinov3/check_patch_embeddings.py --run-id 20260714T120000Z
    python src/dinov3/check_patch_embeddings.py --run-dir data/dinov3_patch_embeddings/<run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dinov3.config import PATCH_EMBEDDINGS_ROOT, expected_cls_dim  # noqa: E402
from src.dinov3.patch_motifs import load_patch_vector  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a DINOv3 patch embedding run.")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--sample-images", type=int, default=5, help="Random images to inspect.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def main() -> None:
    args = parse_args()
    if args.run_dir:
        run_dir = args.run_dir
    elif args.run_id:
        run_dir = PATCH_EMBEDDINGS_ROOT / args.run_id
    else:
        print("Provide --run-id or --run-dir")
        sys.exit(1)

    vectors_dir = run_dir / "vectors"
    ids_path = run_dir / "image_ids.json"
    manifest_path = run_dir / "manifest.json"

    if not vectors_dir.is_dir():
        print(f"Missing vectors directory: {vectors_dir}")
        sys.exit(1)

    manifest: dict = {}
    print(f"Run: {run_dir}")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_id = manifest.get("model_id", "unknown")
        print(
            f"Model: {model_id}  patch_size: {manifest.get('patch_size')}  "
            f"embeddings_run_id: {manifest.get('embeddings_run_id', 'n/a')}"
        )
        if manifest.get("partial"):
            print("WARNING: manifest marked partial=True (run may still be in progress)")

    if ids_path.exists():
        image_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    else:
        image_ids = sorted(p.stem for p in vectors_dir.glob("*.npz"))

    npz_files = sorted(vectors_dir.glob("*.npz"))
    print(f"Vector files: {len(npz_files)}")
    print(f"Image IDs: {len(image_ids)}")

    if len(npz_files) != len(image_ids):
        print("WARNING: vector file count does not match image_ids length")

    expected_dim = manifest.get("embedding_dim") or expected_cls_dim(str(manifest.get("model_id", "")))
    total_patches = 0
    patch_counts: list[int] = []
    missing_ids: list[str] = []

    for image_id in image_ids:
        path = vectors_dir / f"{image_id}.npz"
        if not path.exists():
            missing_ids.append(image_id)
            continue
        patches, rows, cols, _ = load_patch_vector(path)
        if expected_dim and patches.shape[1] != int(expected_dim):
            print(
                f"ERROR: {image_id} patch dim {patches.shape[1]} != expected {expected_dim}"
            )
            sys.exit(1)
        if len(rows) != len(patches) or len(cols) != len(patches):
            print(f"ERROR: {image_id} row/col length mismatch")
            sys.exit(1)
        if not np.isfinite(patches).all():
            print(f"ERROR: {image_id} contains NaN or Inf")
            sys.exit(1)
        n = len(patches)
        total_patches += n
        patch_counts.append(n)

    if missing_ids:
        print(f"ERROR: missing {len(missing_ids)} vector files (e.g. {missing_ids[:3]})")
        sys.exit(1)

    counts = np.array(patch_counts, dtype=np.int64)
    print(f"Total patches: {total_patches:,}")
    print(
        f"Patches per image: min={counts.min()} max={counts.max()} "
        f"mean={counts.mean():.1f} median={int(np.median(counts))}"
    )

    rng = np.random.default_rng(args.seed)
    n_sample = min(args.sample_images, len(image_ids))
    if n_sample > 0:
        print(f"\nSample images ({n_sample}):")
        for image_id in rng.choice(image_ids, size=n_sample, replace=False):
            path = vectors_dir / f"{image_id}.npz"
            patches, rows, cols, _ = load_patch_vector(path)
            norms = np.linalg.norm(patches, axis=1)
            print(
                f"  {image_id}: patches={len(patches)} "
                f"norms min={norms.min():.3f} max={norms.max():.3f}"
            )
            if len(patches) >= 2:
                i, j = rng.choice(len(patches), size=2, replace=False)
                sim = cosine_similarity(patches[i], patches[j])
                print(
                    f"    random pair ({rows[i]},{cols[i]}) vs ({rows[j]},{cols[j]}): "
                    f"cosine={sim:.4f}"
                )

    if manifest.get("model_timing"):
        timing = manifest["model_timing"]
        print(
            f"\nModel timing: load={timing.get('model_load_seconds')}s  "
            f"inference={timing.get('inference_human')}  "
            f"{timing.get('seconds_per_image')}s/image  device={timing.get('device')}"
        )

    print("\nOK: basic patch embedding checks passed.")


if __name__ == "__main__":
    main()
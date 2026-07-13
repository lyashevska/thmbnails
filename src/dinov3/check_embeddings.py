#!/usr/bin/env python3
"""
Sanity-check a DINOv3 embedding run.

Examples:
    python src/dinov3/check_embeddings.py --run-id 20260616T150000Z
    python src/dinov3/check_embeddings.py --run-dir data/dinov3_embeddings/<run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dinov3.config import EMBEDDINGS_ROOT, expected_cls_dim  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a DINOv3 embedding run.")
    p.add_argument("--run-dir", type=Path, default=None)
    p.add_argument("--run-id", default=None)
    p.add_argument("--pairs", type=int, default=5, help="Random pairs for cosine similarity.")
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
        run_dir = EMBEDDINGS_ROOT / args.run_id
    else:
        print("Provide --run-id or --run-dir")
        sys.exit(1)

    emb_path = run_dir / "cls_embeddings.npy"
    ids_path = run_dir / "image_ids.json"
    manifest_path = run_dir / "manifest.json"

    for path in (emb_path, ids_path):
        if not path.exists():
            print(f"Missing required file: {path}")
            sys.exit(1)

    embeddings = np.load(emb_path)
    image_ids = json.loads(ids_path.read_text(encoding="utf-8"))

    manifest: dict = {}
    print(f"Run: {run_dir}")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model_id = manifest.get("model_id", "unknown")
        print(f"Model: {model_id}  cls_size: {manifest.get('cls_size')}")
        expected_dim = manifest.get("embedding_dim") or expected_cls_dim(str(model_id))
        if expected_dim and embeddings.shape[1] != int(expected_dim):
            print(
                f"ERROR: embedding dim {embeddings.shape[1]} != expected {expected_dim} for {model_id}"
            )
            sys.exit(1)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Image IDs: {len(image_ids)}")

    if embeddings.shape[0] != len(image_ids):
        print("ERROR: row count does not match image_ids length")
        sys.exit(1)

    if not np.isfinite(embeddings).all():
        print("ERROR: embeddings contain NaN or Inf")
        sys.exit(1)

    norms = np.linalg.norm(embeddings, axis=1)
    print(f"Vector norms: min={norms.min():.4f} max={norms.max():.4f} mean={norms.mean():.4f}")

    rng = np.random.default_rng(args.seed)
    n = embeddings.shape[0]
    if n >= 2 and args.pairs > 0:
        print(f"\nRandom cosine similarities ({args.pairs} pairs):")
        for _ in range(min(args.pairs, n * (n - 1) // 2)):
            i, j = rng.choice(n, size=2, replace=False)
            sim = cosine_similarity(embeddings[i], embeddings[j])
            print(f"  {image_ids[i]} vs {image_ids[j]}: {sim:.4f}")

    print("\nOK: basic embedding checks passed.")


if __name__ == "__main__":
    main()
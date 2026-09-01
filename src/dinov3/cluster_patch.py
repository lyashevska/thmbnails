#!/usr/bin/env python3
"""
Cluster DINOv3 patch embeddings (one label per 16×16 token).

Parallel to cluster_cls.py (one label per thumbnail):
  - CLS   -> data/dinov3_cls_clusters/
  - Patch -> data/dinov3_patch_clusters/

Examples:
    python src/dinov3/extract_patch.py --embeddings-run-id 20260713T131720Z --limit 50
    python src/dinov3/cluster_patch.py --patch-run-id <patch_run_id>
    python src/dinov3/cluster_patch.py --patch-run-id <patch_run_id> --hdbscan-min-cluster-size 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dinov3.config import (  # noqa: E402
    CLS_CLUSTERING_TYPE,
    CLS_CLUSTERS_ROOT,
    CSV_DEFAULT,
    DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
    DEFAULT_HDBSCAN_MIN_SAMPLES,
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_PATCH_SIZE,
    DEFAULT_SAMPLES_PER_CLUSTER,
    DEFAULT_UMAP_MIN_DIST,
    DEFAULT_UMAP_NEIGHBORS,
    DEFAULT_VIT_PATCH_SIZE,
    PATCH_CLUSTERING_TYPE,
    PATCH_CLUSTERS_ROOT,
    PATCH_EMBEDDINGS_ROOT,
    THUMB_DIR_DEFAULT,
)
from src.dinov3.cluster import (  # noqa: E402
    build_image_cluster_histogram,
    build_patch_assignments,
    cluster_summary,
    load_patch_corpus,
    run_cluster_pipeline,
    run_id_now,
    save_patch_cluster_montages,
    save_umap_plot,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cluster DINOv3 patch embeddings (224px).")
    p.add_argument("--patch-run-id", required=True, help="Patch embedding run in data/dinov3_patch_embeddings/")
    p.add_argument("--out-dir", type=Path, default=PATCH_CLUSTERS_ROOT)
    p.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    p.add_argument("--thumb-dir", type=Path, default=THUMB_DIR_DEFAULT)
    p.add_argument("--pca-components", type=int, default=DEFAULT_PCA_COMPONENTS)
    p.add_argument("--umap-neighbors", type=int, default=DEFAULT_UMAP_NEIGHBORS)
    p.add_argument("--umap-min-dist", type=float, default=DEFAULT_UMAP_MIN_DIST)
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=30)
    p.add_argument("--hdbscan-min-samples", type=int, default=3)
    p.add_argument(
        "--hdbscan-selection-method",
        choices=("eom", "leaf"),
        default="eom",
        help="HDBSCAN cluster_selection_method (eom = fewer clusters, leaf = more micro-clusters).",
    )
    p.add_argument("--samples-per-cluster", type=int, default=DEFAULT_SAMPLES_PER_CLUSTER)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-id", default=None)
    return p.parse_args()


def resolve_patch_run(run_id: str) -> Path:
    run_dir = PATCH_EMBEDDINGS_ROOT / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Patch embedding run not found: {run_dir}")
    return run_dir


def main() -> None:
    args = parse_args()

    print("Step 1: Load patch corpus")
    patch_run_dir = resolve_patch_run(args.patch_run_id)
    corpus = load_patch_corpus(patch_run_dir)
    print(f"  patch_run={patch_run_dir.name}")
    print(f"  patches={corpus.patches.shape}  images={len(set(corpus.image_ids))}")

    print("\nStep 2: PCA → UMAP → HDBSCAN on patches")
    result = run_cluster_pipeline(
        corpus.patches,
        cluster_space="pca",
        pca_components=args.pca_components,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        hdbscan_selection_method=args.hdbscan_selection_method,
        seed=args.seed,
    )
    n_clusters = len(set(result.labels)) - (1 if -1 in result.labels else 0)
    n_noise = int((result.labels == -1).sum())
    print(f"  pca_components={result.pca_components}  explained_var={result.explained_variance_ratio:.3f}")
    print(f"  clusters={n_clusters}  noise={n_noise} ({100 * n_noise / len(result.labels):.1f}%)")

    print("\nStep 3: Build tables")
    assignments = build_patch_assignments(corpus, result)
    summary = cluster_summary(assignments)
    histogram = build_image_cluster_histogram(assignments)

    run_id = args.run_id or run_id_now()
    out_dir = args.out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    assignments.to_csv(out_dir / "patch_assignments.csv", index=False)
    summary.to_csv(out_dir / "cluster_summary.csv", index=False)
    histogram.to_csv(out_dir / "image_cluster_histogram.csv", index=False)
    print(f"  {out_dir}/patch_assignments.csv")
    print(summary.head(15).to_string(index=False))

    print("\nStep 4: Patch-level UMAP plot")
    save_umap_plot(
        assignments,
        out_dir / "patch_umap.png",
        title="Patch clusters (UMAP)",
        point_size=6,
    )

    print("\nStep 5: Patch crops (local 16x16 units)")
    sample_info = save_patch_cluster_montages(
        assignments,
        out_dir / "patch_crops",
        thumb_dir=args.thumb_dir,
        samples_per_cluster=args.samples_per_cluster,
        vit_patch_size=DEFAULT_VIT_PATCH_SIZE,
    )
    print(f"  {out_dir}/patch_crops/ ({len(sample_info)} cluster folders)")

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "clustering_type": PATCH_CLUSTERING_TYPE,
        "comparison": {
            "cls_clustering_type": CLS_CLUSTERING_TYPE,
            "cls_clusters_dir": str(CLS_CLUSTERS_ROOT),
            "note": "CLS clusters group whole thumbnails; patch clusters group local visual units.",
        },
        "patch_embeddings_run_id": patch_run_dir.name,
        "patch_size": DEFAULT_PATCH_SIZE,
        "pca_components": result.pca_components,
        "explained_variance_ratio": result.explained_variance_ratio,
        "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
        "hdbscan_min_samples": args.hdbscan_min_samples,
        "hdbscan_selection_method": args.hdbscan_selection_method,
        "seed": args.seed,
        "n_patches": int(len(assignments)),
        "n_images": int(assignments["image_id"].nunique()),
        "n_clusters": n_clusters,
        "n_noise_patches": n_noise,
        "patch_crop_info": sample_info,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nStep 6: Manifest written")
    print(f"  {out_dir}/manifest.json")
    print(f"\nDone. Cluster tables and patch crops are in {out_dir}/")


if __name__ == "__main__":
    main()

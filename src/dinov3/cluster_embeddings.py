#!/usr/bin/env python3
"""
Cluster DINOv3 CLS embeddings with PCA → UMAP → HDBSCAN.

Reads a completed embedding run from data/dinov3_embeddings/<run_id>/ and writes:
  cluster_assignments.csv
  cluster_summary.csv
  umap.png
  samples/cluster_<id>/   copied thumbnails + _grid.jpg
  manifest.json

Examples:
    python src/dinov3/cluster_embeddings.py --embeddings-run-id 20260617T091002Z
    python src/dinov3/cluster_embeddings.py
    python src/dinov3/cluster_embeddings.py --hdbscan-min-cluster-size 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dinov3.cluster import (  # noqa: E402
    build_assignments_frame,
    build_metadata_frame,
    cluster_summary,
    load_embedding_run,
    resolve_embeddings_run,
    run_cluster_pipeline,
    run_id_now,
    save_cluster_sample_grids,
    save_umap_plot,
)
from src.dinov3.config import (  # noqa: E402
    CLUSTERS_ROOT,
    CSV_DEFAULT,
    DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
    DEFAULT_HDBSCAN_MIN_SAMPLES,
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_SAMPLES_PER_CLUSTER,
    DEFAULT_UMAP_MIN_DIST,
    DEFAULT_UMAP_NEIGHBORS,
    THUMB_DIR_DEFAULT,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cluster DINOv3 CLS embeddings.")
    p.add_argument("--embeddings-run-id", default=None, help="Embedding run folder name.")
    p.add_argument("--out-dir", type=Path, default=CLUSTERS_ROOT)
    p.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    p.add_argument("--thumb-dir", type=Path, default=THUMB_DIR_DEFAULT)
    p.add_argument("--pca-components", type=int, default=DEFAULT_PCA_COMPONENTS)
    p.add_argument("--umap-neighbors", type=int, default=DEFAULT_UMAP_NEIGHBORS)
    p.add_argument("--umap-min-dist", type=float, default=DEFAULT_UMAP_MIN_DIST)
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE)
    p.add_argument("--hdbscan-min-samples", type=int, default=DEFAULT_HDBSCAN_MIN_SAMPLES)
    p.add_argument("--samples-per-cluster", type=int, default=DEFAULT_SAMPLES_PER_CLUSTER)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-id", default=None, help="Output cluster run id (default: timestamp).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("Step 1: Load embeddings")
    emb_run_dir = resolve_embeddings_run(run_id=args.embeddings_run_id)
    embeddings, image_ids, emb_manifest = load_embedding_run(emb_run_dir)
    print(f"  embeddings_run={emb_run_dir.name}")
    print(f"  shape={embeddings.shape}  model={emb_manifest.get('model_id', 'unknown')}")

    print("\nStep 2: PCA → UMAP → HDBSCAN")
    result = run_cluster_pipeline(
        embeddings,
        pca_components=args.pca_components,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        seed=args.seed,
    )
    n_clusters = len(set(result.labels)) - (1 if -1 in result.labels else 0)
    n_noise = int((result.labels == -1).sum())
    print(f"  pca_components={result.pca_components}  explained_var={result.explained_variance_ratio:.3f}")
    print(f"  clusters={n_clusters}  noise={n_noise} ({100 * n_noise / len(result.labels):.1f}%)")

    print("\nStep 3: Join metadata and save tables")
    metadata = build_metadata_frame(image_ids, csv_path=args.csv)
    assignments = build_assignments_frame(image_ids, result, metadata)
    summary = cluster_summary(assignments)

    run_id = args.run_id or run_id_now()
    out_dir = args.out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    assignments_path = out_dir / "cluster_assignments.csv"
    summary_path = out_dir / "cluster_summary.csv"
    assignments.to_csv(assignments_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"  {assignments_path}")
    print(f"  {summary_path}")
    print(summary.to_string(index=False))

    print("\nStep 4: UMAP plot")
    umap_path = out_dir / "umap.png"
    save_umap_plot(assignments, umap_path)
    print(f"  {umap_path}")

    print("\nStep 5: Cluster sample thumbnails")
    samples_dir = out_dir / "samples"
    sample_info = save_cluster_sample_grids(
        assignments,
        samples_dir,
        thumb_dir=args.thumb_dir,
        samples_per_cluster=args.samples_per_cluster,
    )
    print(f"  {samples_dir} ({len(sample_info)} cluster folders)")

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "embeddings_run_id": emb_run_dir.name,
        "embeddings_model": emb_manifest.get("model_id"),
        "embedding_shape": list(embeddings.shape),
        "pca_components": result.pca_components,
        "explained_variance_ratio": result.explained_variance_ratio,
        "umap_neighbors": args.umap_neighbors,
        "umap_min_dist": args.umap_min_dist,
        "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
        "hdbscan_min_samples": args.hdbscan_min_samples,
        "seed": args.seed,
        "n_images": len(image_ids),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "cluster_sample_info": sample_info,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nStep 6: Write manifest")
    print(f"  {manifest_path}")
    print("\nDone. Open umap.png and samples/cluster_*/_grid.jpg to review clusters.")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
CLS thumbnail clustering (one label per image).

Default HDBSCAN path: PCA → 10-D UMAP (cluster space) → HDBSCAN, with a
separate 2D UMAP for umap.png. Pass --pca-components 0 / --skip-pca to run
UMAP on raw CLS. Pass --cluster-space pca to reproduce older runs that
clustered in PCA. For patch-token clusters, use cluster_patch.py.

Reads a completed embedding run from data/dinov3_cls_embeddings/<run_id>/ and writes:
  cluster_assignments.csv
  cluster_summary.csv
  cluster_space.npy        the array HDBSCAN/k-means used
  umap.png
  samples/cluster_<id>/   copied thumbnails + _grid.jpg
  manifest.json
  stability.json          only with --stability-runs

Examples:
    python src/dinov3/cluster_cls.py --embeddings-run-id 20260713T131720Z
    python src/dinov3/cluster_cls.py --skip-pca --umap-neighbors 15 --umap-min-dist 0.0
    python src/dinov3/cluster_cls.py --cluster-space pca --method kmeans --n-clusters 40
    python src/dinov3/cluster_cls.py --stability-runs 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dinov3.cluster import (  # noqa: E402
    CLUSTER_METHODS,
    CLUSTER_SPACES,
    build_assignments_frame,
    build_metadata_frame,
    cluster_in_space,
    cluster_summary,
    load_embedding_run,
    resolve_embeddings_run,
    run_cluster_pipeline,
    run_id_now,
    save_cluster_sample_grids,
    save_umap_plot,
    umap_reduce,
)
from src.dinov3.cluster_metrics import pairwise_stability, quality_row  # noqa: E402
from src.dinov3.config import (  # noqa: E402
    CLS_CLUSTERING_TYPE,
    CLS_CLUSTERS_ROOT,
    CSV_DEFAULT,
    DEFAULT_CLUSTER_SPACE,
    DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
    DEFAULT_HDBSCAN_MIN_SAMPLES,
    DEFAULT_HDBSCAN_SELECTION_METHOD,
    DEFAULT_KMEANS_CLUSTERS,
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_SAMPLES_PER_CLUSTER,
    DEFAULT_UMAP_CLUSTER_COMPONENTS,
    DEFAULT_UMAP_MIN_DIST,
    DEFAULT_UMAP_NEIGHBORS,
    PATCH_CLUSTERING_TYPE,
    PATCH_CLUSTERS_ROOT,
    THUMB_DIR_DEFAULT,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cluster DINOv3 CLS embeddings.")
    p.add_argument("--embeddings-run-id", default=None, help="Embedding run folder name.")
    p.add_argument("--out-dir", type=Path, default=CLS_CLUSTERS_ROOT)
    p.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    p.add_argument("--thumb-dir", type=Path, default=THUMB_DIR_DEFAULT)
    p.add_argument(
        "--method",
        choices=CLUSTER_METHODS,
        default="hdbscan",
        help="Clustering algorithm.",
    )
    p.add_argument(
        "--cluster-space",
        choices=CLUSTER_SPACES,
        default=DEFAULT_CLUSTER_SPACE,
        help="HDBSCAN/k-means geometry: umap (default, n-D UMAP) or pca (older runs).",
    )
    p.add_argument(
        "--pca-components",
        type=int,
        default=DEFAULT_PCA_COMPONENTS,
        help="PCA width. 0 skips PCA and feeds raw CLS to UMAP (umap space only).",
    )
    p.add_argument(
        "--skip-pca",
        action="store_true",
        help="Same as --pca-components 0.",
    )
    p.add_argument("--umap-components", type=int, default=DEFAULT_UMAP_CLUSTER_COMPONENTS)
    p.add_argument("--umap-neighbors", type=int, default=DEFAULT_UMAP_NEIGHBORS)
    p.add_argument("--umap-min-dist", type=float, default=DEFAULT_UMAP_MIN_DIST)
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE)
    p.add_argument("--hdbscan-min-samples", type=int, default=DEFAULT_HDBSCAN_MIN_SAMPLES)
    p.add_argument(
        "--hdbscan-selection-method",
        choices=("leaf", "eom"),
        default=DEFAULT_HDBSCAN_SELECTION_METHOD,
        help="HDBSCAN cluster_selection_method (eom = fewer clusters, leaf = more).",
    )
    p.add_argument(
        "--n-clusters",
        type=int,
        default=DEFAULT_KMEANS_CLUSTERS,
        help="K for kmeans/agglomerative (ignored for hdbscan).",
    )
    p.add_argument("--samples-per-cluster", type=int, default=DEFAULT_SAMPLES_PER_CLUSTER)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--stability-runs",
        type=int,
        default=0,
        help="After the main run, repeat UMAP+cluster with this many seeds; write ARI/NMI.",
    )
    p.add_argument("--run-id", default=None, help="Output cluster run id (default: timestamp).")
    return p.parse_args()


def _run_stability(
    *,
    pca_embeddings: np.ndarray,
    args: argparse.Namespace,
    reference_labels: np.ndarray,
) -> Dict[str, Any]:
    """Vary only the UMAP seed; the UMAP source stays fixed. First partition is the main run."""
    n_runs = args.stability_runs
    src = "raw CLS" if args.pca_components <= 0 else "PCA"
    print(f"\nStability: {n_runs} UMAP seeds ({src} frozen)")
    label_runs: List[np.ndarray] = [np.asarray(reference_labels)]
    for i in range(1, n_runs):
        seed = args.seed + i
        if args.cluster_space == "umap":
            space = umap_reduce(
                pca_embeddings,
                n_components=args.umap_components,
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
                seed=seed,
            )
        else:
            space = pca_embeddings
        labels, _ = cluster_in_space(
            space,
            method=args.method,
            hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
            hdbscan_min_samples=args.hdbscan_min_samples,
            hdbscan_selection_method=args.hdbscan_selection_method,
            n_clusters=args.n_clusters,
            seed=seed,
        )
        label_runs.append(labels)
        if (i + 1) % 10 == 0 or i + 1 == n_runs:
            print(f"  {i + 1}/{n_runs}")
    stats = pairwise_stability(label_runs)
    print(
        f"  ARI={stats['ari_mean']:.3f} (SD {stats['ari_sd']:.3f})  "
        f"NMI={stats['nmi_mean']:.3f} (SD {stats['nmi_sd']:.3f})"
    )
    if stats["ari_assigned_mean"] is not None:
        print(
            f"  assigned-only ARI={stats['ari_assigned_mean']:.3f} "
            f"NMI={stats['nmi_assigned_mean']:.3f}"
        )
    return stats


def main() -> None:
    args = parse_args()
    if args.skip_pca:
        args.pca_components = 0

    print("Step 1: Load embeddings")
    emb_run_dir = resolve_embeddings_run(run_id=args.embeddings_run_id)
    embeddings, image_ids, emb_manifest = load_embedding_run(emb_run_dir)
    print(f"  embeddings_run={emb_run_dir.name}")
    print(f"  shape={embeddings.shape}  model={emb_manifest.get('model_id', 'unknown')}")

    space_label = (
        f"UMAP-{args.umap_components}D"
        if args.cluster_space == "umap"
        else "PCA"
    )
    source_label = "raw CLS" if args.pca_components <= 0 else "PCA"
    print(f"\nStep 2: {source_label} → {space_label} → {args.method.upper()}")
    result = run_cluster_pipeline(
        embeddings,
        method=args.method,
        cluster_space=args.cluster_space,
        pca_components=args.pca_components,
        umap_cluster_components=args.umap_components,
        umap_neighbors=args.umap_neighbors,
        umap_min_dist=args.umap_min_dist,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_min_samples=args.hdbscan_min_samples,
        hdbscan_selection_method=args.hdbscan_selection_method,
        n_clusters=args.n_clusters,
        seed=args.seed,
    )
    quality = quality_row(
        result.cluster_embeddings,
        result.labels,
        min_cluster_size=args.hdbscan_min_cluster_size if args.method == "hdbscan" else None,
    )
    if result.pca_components == 0:
        print("  pca skipped (UMAP on raw CLS)")
    else:
        print(
            f"  pca_components={result.pca_components}  "
            f"explained_var={result.explained_variance_ratio:.3f}"
        )
    print(
        f"  clusters={quality['n_clusters']}  noise={quality['n_noise']} "
        f"({100 * quality['noise_fraction']:.1f}%)"
    )
    print(f"  dbcv={quality['dbcv']}  silhouette={quality['silhouette']}")

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
    np.save(out_dir / "cluster_space.npy", result.cluster_embeddings)
    print(f"  {assignments_path}")
    print(f"  {summary_path}")
    print(f"  {out_dir / 'cluster_space.npy'}  {result.cluster_embeddings.shape}")
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

    stability: Dict[str, Any] | None = None
    if args.stability_runs and args.stability_runs > 1:
        if args.cluster_space == "pca":
            print(
                "\nNote: --cluster-space pca is nearly deterministic; "
                "seeded ARI/NMI will sit near 1. Use umap space for a real stability test."
            )
        stability = _run_stability(
            pca_embeddings=result.pca_embeddings,
            args=args,
            reference_labels=result.labels,
        )
        (out_dir / "stability.json").write_text(json.dumps(stability, indent=2), encoding="utf-8")
        print(f"  {out_dir / 'stability.json'}")

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "clustering_type": CLS_CLUSTERING_TYPE,
        "comparison": {
            "patch_clustering_type": PATCH_CLUSTERING_TYPE,
            "patch_clusters_dir": str(PATCH_CLUSTERS_ROOT),
            "note": "CLS clusters group whole thumbnails; patch clusters group local visual units.",
        },
        "embeddings_run_id": emb_run_dir.name,
        "embeddings_model": emb_manifest.get("model_id"),
        "embedding_shape": list(embeddings.shape),
        "method": args.method,
        "cluster_space": result.cluster_space,
        "cluster_space_shape": list(result.cluster_embeddings.shape),
        "pca_components": result.pca_components,
        "skip_pca": result.pca_components == 0,
        "explained_variance_ratio": result.explained_variance_ratio,
        "umap_cluster_components": result.umap_cluster_components,
        "umap_neighbors": args.umap_neighbors,
        "umap_min_dist": args.umap_min_dist,
        "hdbscan_min_cluster_size": args.hdbscan_min_cluster_size,
        "hdbscan_min_samples": args.hdbscan_min_samples,
        "hdbscan_selection_method": args.hdbscan_selection_method,
        "n_clusters_target": args.n_clusters if args.method != "hdbscan" else None,
        "seed": args.seed,
        "n_images": len(image_ids),
        "n_clusters": quality["n_clusters"],
        "n_noise": quality["n_noise"],
        "noise_fraction": quality["noise_fraction"],
        "dbcv": quality["dbcv"],
        "silhouette": quality["silhouette"],
        "size_median": quality["size_median"],
        "cluster_sample_info": sample_info,
        "stability": stability,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nStep 6: Write manifest")
    print(f"  {manifest_path}")
    print("\nDone. Open umap.png and samples/cluster_*/_grid.jpg to review clusters.")


if __name__ == "__main__":
    main()

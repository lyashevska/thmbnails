#!/usr/bin/env python3
"""
Stage 1 residual CLS: remove the parent-cluster look, then cluster what is left.

Peel re-clusters leftover *images*. This subtracts leftover *appearance* in
raw 1024-d CLS (not UMAP) and keeps every thumbnail. Assigned points subtract
their parent mean; ungrouped points subtract the nearest parent centroid
(cosine). Residuals are L2-normalised, then the parent UMAP+HDBSCAN knobs
are refit.

Examples:
    python src/dinov3/cluster_cls_residual.py \\
      --from-clusters-run-id nopca-n15-mcs20-ms20 --dry-run

    python src/dinov3/cluster_cls_residual.py \\
      --from-clusters-run-id nopca-n15-mcs20-ms20

    python src/dinov3/cluster_cls_residual.py \\
      --from-clusters-run-id nopca-n15-mcs20-ms20 --deflate pca --pca-deflate 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dinov3.cluster import (  # noqa: E402
    CLUSTER_SPACES,
    build_assignments_frame,
    build_metadata_frame,
    cluster_summary,
    load_cluster_run,
    load_embedding_run,
    residual_from_global_pca,
    residual_from_parent_means,
    resolve_clusters_run,
    resolve_embeddings_run,
    run_cluster_pipeline,
    save_cluster_sample_grids,
    save_umap_plot,
)
from src.dinov3.cluster_metrics import quality_row  # noqa: E402
from src.dinov3.config import (  # noqa: E402
    CLS_CLUSTERING_TYPE,
    CLS_CLUSTERS_ROOT,
    CSV_DEFAULT,
    DEFAULT_CLUSTER_SPACE,
    DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
    DEFAULT_HDBSCAN_MIN_SAMPLES,
    DEFAULT_HDBSCAN_SELECTION_METHOD,
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_SAMPLES_PER_CLUSTER,
    DEFAULT_UMAP_CLUSTER_COMPONENTS,
    DEFAULT_UMAP_MIN_DIST,
    DEFAULT_UMAP_NEIGHBORS,
    PATCH_CLUSTERING_TYPE,
    PATCH_CLUSTERS_ROOT,
    THUMB_DIR_DEFAULT,
)

DEFLATE_MODES = ("cluster-mean", "pca")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cluster residual CLS after removing the parent-cluster look."
    )
    p.add_argument(
        "--from-clusters-run-id",
        required=True,
        help="Parent cluster folder under data/dinov3_cls_clusters/.",
    )
    p.add_argument(
        "--embeddings-run-id",
        default=None,
        help="CLS embedding run. Default: parent manifest embeddings_run_id.",
    )
    p.add_argument("--out-dir", type=Path, default=CLS_CLUSTERS_ROOT)
    p.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    p.add_argument("--thumb-dir", type=Path, default=THUMB_DIR_DEFAULT)
    p.add_argument(
        "--deflate",
        choices=DEFLATE_MODES,
        default="cluster-mean",
        help="cluster-mean: subtract parent centroid (default). pca: drop global PCs.",
    )
    p.add_argument(
        "--pca-deflate",
        type=int,
        default=3,
        help="How many global PCs to remove when --deflate pca (default 3).",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Output folder (default: <parent>-residual or <parent>-residual-pca).",
    )
    p.add_argument("--cluster-space", choices=CLUSTER_SPACES, default=None)
    p.add_argument("--pca-components", type=int, default=None)
    p.add_argument("--umap-components", type=int, default=None)
    p.add_argument("--umap-neighbors", type=int, default=None)
    p.add_argument("--umap-min-dist", type=float, default=None)
    p.add_argument("--hdbscan-min-cluster-size", type=int, default=None)
    p.add_argument("--hdbscan-min-samples", type=int, default=None)
    p.add_argument("--hdbscan-selection-method", choices=("leaf", "eom"), default=None)
    p.add_argument("--samples-per-cluster", type=int, default=DEFAULT_SAMPLES_PER_CLUSTER)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _pick(cli: Any, parent: Dict[str, Any], key: str, default: Any) -> Any:
    if cli is not None:
        return cli
    if key in parent and parent[key] is not None:
        return parent[key]
    return default


def knobs_from_parent(args: argparse.Namespace, parent_man: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cluster_space": _pick(args.cluster_space, parent_man, "cluster_space", DEFAULT_CLUSTER_SPACE),
        "pca_components": int(
            _pick(args.pca_components, parent_man, "pca_components", DEFAULT_PCA_COMPONENTS)
        ),
        "umap_components": int(
            _pick(
                args.umap_components,
                parent_man,
                "umap_cluster_components",
                DEFAULT_UMAP_CLUSTER_COMPONENTS,
            )
        ),
        "umap_neighbors": int(
            _pick(args.umap_neighbors, parent_man, "umap_neighbors", DEFAULT_UMAP_NEIGHBORS)
        ),
        "umap_min_dist": float(
            _pick(args.umap_min_dist, parent_man, "umap_min_dist", DEFAULT_UMAP_MIN_DIST)
        ),
        "hdbscan_min_cluster_size": int(
            _pick(
                args.hdbscan_min_cluster_size,
                parent_man,
                "hdbscan_min_cluster_size",
                DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
            )
        ),
        "hdbscan_min_samples": int(
            _pick(
                args.hdbscan_min_samples,
                parent_man,
                "hdbscan_min_samples",
                DEFAULT_HDBSCAN_MIN_SAMPLES,
            )
        ),
        "hdbscan_selection_method": str(
            _pick(
                args.hdbscan_selection_method,
                parent_man,
                "hdbscan_selection_method",
                DEFAULT_HDBSCAN_SELECTION_METHOD,
            )
        ),
        "seed": int(_pick(args.seed, parent_man, "seed", 42)),
    }


def align_parent_labels(
    image_ids: List[str],
    assignments: pd.DataFrame,
) -> np.ndarray:
    """Parent cluster_id in embedding-row order."""
    dedup = assignments.drop_duplicates("image_id")
    by_id = dedup.set_index(dedup["image_id"].astype(str))["cluster_id"]
    missing = [i for i in image_ids if i not in by_id.index]
    if missing:
        raise KeyError(f"{len(missing)} embedding ids not in parent assignments (e.g. {missing[:3]})")
    return by_id.loc[list(image_ids)].to_numpy(dtype=np.int64)


def residual_mix_table(assignments: pd.DataFrame) -> pd.DataFrame:
    """How mixed each residual cluster is across parent labels."""
    rows: List[Dict[str, Any]] = []
    for cid, group in assignments.groupby("cluster_id"):
        cid_int = int(cid)
        n = int(len(group))
        counts = group["parent_cluster_id"].value_counts()
        majority_id = int(counts.index[0])
        rows.append(
            {
                "residual_cluster_id": cid_int,
                "n": n,
                "n_parent_ids": int(group["parent_cluster_id"].nunique()),
                "majority_parent_id": majority_id,
                "majority_parent_frac": round(float(counts.iloc[0]) / n, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("residual_cluster_id").reset_index(drop=True)


def default_run_id(parent_name: str, deflate: str) -> str:
    if deflate == "pca":
        return f"{parent_name}-residual-pca"
    return f"{parent_name}-residual"


def main() -> None:
    args = parse_args()

    parent_dir = resolve_clusters_run(args.from_clusters_run_id)
    parent_assign, parent_man = load_cluster_run(parent_dir)
    knobs = knobs_from_parent(args, parent_man)
    emb_id = args.embeddings_run_id or parent_man.get("embeddings_run_id")
    if not emb_id:
        print("Pass --embeddings-run-id (parent manifest has none).")
        sys.exit(1)

    run_id = args.run_id or default_run_id(parent_dir.name, args.deflate)
    out_dir = args.out_dir / run_id

    print("Step 1: Load parent labels and CLS embeddings")
    print(f"  parent={parent_dir.name}")
    print(f"  deflate={args.deflate}")
    print(
        f"  knobs: space={knobs['cluster_space']} umap={knobs['umap_components']}D "
        f"n_neighbors={knobs['umap_neighbors']} min_dist={knobs['umap_min_dist']} "
        f"mcs={knobs['hdbscan_min_cluster_size']} ms={knobs['hdbscan_min_samples']} "
        f"{knobs['hdbscan_selection_method']}"
    )

    emb_run_dir = resolve_embeddings_run(run_id=str(emb_id))
    embeddings, image_ids, emb_manifest = load_embedding_run(emb_run_dir)
    parent_labels = align_parent_labels(image_ids, parent_assign)
    n_assigned = int((parent_labels >= 0).sum())
    n_noise = int((parent_labels < 0).sum())
    n_parent_clusters = int(len(set(parent_labels.tolist()) - {-1}))
    print(f"  embeddings_run={emb_run_dir.name}  shape={embeddings.shape}")
    print(f"  parent assigned={n_assigned}  noise={n_noise}  parent_clusters={n_parent_clusters}")

    if args.dry_run:
        print(f"Dry run: would write {out_dir}")
        if args.deflate == "cluster-mean":
            print("  residual = x - parent mean (noise: nearest centroid, cosine)")
        else:
            print(f"  residual = x minus first {args.pca_deflate} global PCs")
        print("  then L2-normalise and refit parent UMAP+HDBSCAN on all images")
        return

    print("\nStep 2: Residual in raw 1024-d CLS")
    pca_explained = None
    pca_deflate_n = None
    deflate_ids: np.ndarray
    if args.deflate == "cluster-mean":
        residual = residual_from_parent_means(embeddings, parent_labels)
        residual_vectors = residual.residuals
        deflate_ids = residual.deflate_cluster_id
        print(f"  subtracted {len(residual.mean_cluster_ids)} parent means")
    else:
        residual_vectors, pca_deflate_n, pca_explained = residual_from_global_pca(
            embeddings,
            n_components=args.pca_deflate,
            seed=knobs["seed"],
        )
        deflate_ids = np.full(len(parent_labels), -1, dtype=np.int64)
        print(
            f"  removed {pca_deflate_n} global PCs  "
            f"explained_var={pca_explained:.3f}"
        )

    src = "raw residual CLS" if knobs["pca_components"] <= 0 else "PCA of residuals"
    print(f"\nStep 3: {src} → UMAP → HDBSCAN")
    result = run_cluster_pipeline(
        residual_vectors,
        method="hdbscan",
        cluster_space=knobs["cluster_space"],
        pca_components=knobs["pca_components"],
        umap_cluster_components=knobs["umap_components"],
        umap_neighbors=knobs["umap_neighbors"],
        umap_min_dist=knobs["umap_min_dist"],
        hdbscan_min_cluster_size=knobs["hdbscan_min_cluster_size"],
        hdbscan_min_samples=knobs["hdbscan_min_samples"],
        hdbscan_selection_method=knobs["hdbscan_selection_method"],
        seed=knobs["seed"],
    )
    quality = quality_row(
        result.cluster_embeddings,
        result.labels,
        min_cluster_size=knobs["hdbscan_min_cluster_size"],
    )
    print(
        f"  clusters={quality['n_clusters']}  noise={quality['n_noise']} "
        f"({100 * quality['noise_fraction']:.1f}%)  "
        f"dbcv={quality['dbcv']}  silhouette={quality['silhouette']}"
    )

    print("\nStep 4: Join parent labels and save")
    metadata = build_metadata_frame(image_ids, csv_path=args.csv)
    assignments = build_assignments_frame(image_ids, result, metadata)
    assignments.insert(
        assignments.columns.get_loc("cluster_id"),
        "parent_cluster_id",
        parent_labels,
    )
    assignments.insert(
        assignments.columns.get_loc("cluster_id"),
        "deflate_cluster_id",
        deflate_ids,
    )
    summary = cluster_summary(assignments)
    mix = residual_mix_table(assignments)

    out_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(out_dir / "cluster_assignments.csv", index=False)
    summary.to_csv(out_dir / "cluster_summary.csv", index=False)
    mix.to_csv(out_dir / "parent_residual_mix.csv", index=False)
    np.save(out_dir / "cluster_space.npy", result.cluster_embeddings)
    print(f"  {out_dir / 'cluster_assignments.csv'}")
    print(summary.to_string(index=False))
    assigned_mix = mix[mix["residual_cluster_id"] >= 0]
    if not assigned_mix.empty:
        print(
            "  residual mix (assigned): "
            f"median majority_parent_frac="
            f"{assigned_mix['majority_parent_frac'].median():.2f}  "
            f"median n_parent_ids={assigned_mix['n_parent_ids'].median():.0f}"
        )

    print("\nStep 5: UMAP plot and sample grids")
    save_umap_plot(
        assignments,
        out_dir / "umap.png",
        title=f"Residual CLS ({args.deflate}) UMAP",
    )
    sample_info = save_cluster_sample_grids(
        assignments,
        out_dir / "samples",
        thumb_dir=args.thumb_dir,
        samples_per_cluster=args.samples_per_cluster,
    )
    print(f"  {out_dir / 'samples'} ({len(sample_info)} cluster folders)")

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "clustering_type": CLS_CLUSTERING_TYPE,
        "kind": "cls_residual",
        "deflate": args.deflate,
        "pca_deflate": pca_deflate_n,
        "pca_deflate_explained_variance_ratio": pca_explained,
        "parent_clusters_run_id": parent_dir.name,
        "n_parent_assigned": n_assigned,
        "n_parent_noise": n_noise,
        "n_parent_clusters": n_parent_clusters,
        "comparison": {
            "patch_clustering_type": PATCH_CLUSTERING_TYPE,
            "patch_clusters_dir": str(PATCH_CLUSTERS_ROOT),
            "note": (
                "Residual clustering subtracts a whole-image look in raw CLS, "
                "then groups all thumbnails. cluster_id is the residual label; "
                "parent_cluster_id is the parent cut. Review residual sample grids."
            ),
        },
        "embeddings_run_id": emb_run_dir.name,
        "embeddings_model": emb_manifest.get("model_id"),
        "embedding_shape": list(embeddings.shape),
        "method": "hdbscan",
        "cluster_space": result.cluster_space,
        "cluster_space_shape": list(result.cluster_embeddings.shape),
        "pca_components": result.pca_components,
        "explained_variance_ratio": result.explained_variance_ratio,
        "umap_cluster_components": result.umap_cluster_components,
        "umap_neighbors": knobs["umap_neighbors"],
        "umap_min_dist": knobs["umap_min_dist"],
        "hdbscan_min_cluster_size": knobs["hdbscan_min_cluster_size"],
        "hdbscan_min_samples": knobs["hdbscan_min_samples"],
        "hdbscan_selection_method": knobs["hdbscan_selection_method"],
        "seed": knobs["seed"],
        "n_images": len(image_ids),
        "n_clusters": quality["n_clusters"],
        "n_noise": quality["n_noise"],
        "noise_fraction": quality["noise_fraction"],
        "dbcv": quality["dbcv"],
        "silhouette": quality["silhouette"],
        "size_median": quality["size_median"],
        "cluster_sample_info": sample_info,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  {out_dir / 'manifest.json'}")
    print(
        "\nDone. Inspect samples/cluster_*/_grid.jpg: a useful residual group "
        "mixes parent clusters and shares something else (pose, framing, object)."
    )


if __name__ == "__main__":
    main()

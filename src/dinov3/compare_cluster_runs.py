#!/usr/bin/env python3
"""Compare two CLS cluster runs on native and shared-space metrics.

Native DBCV/silhouette live in each run's own HDBSCAN space (usually 10-D UMAP)
and are not interchangeable. Shared-space scores put both labelings in the same
geometry: raw CLS (cosine silhouette) and PCA-50 (euclidean silhouette).

Examples:
    python src/dinov3/compare_cluster_runs.py \\
      --run-a sweep-A-n15-mcs20-ms20 \\
      --run-b nopca-n15-mcs20-ms20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dinov3.cluster import load_cluster_run, load_embedding_run, pca_reduce, resolve_clusters_run, resolve_embeddings_run  # noqa: E402
from src.dinov3.cluster_metrics import (  # noqa: E402
    n_clusters,
    n_noise,
    noise_fraction,
    pairwise_stability,
    quality_row,
    silhouette_assigned,
    size_stats,
)
from src.dinov3.config import CLS_CLUSTERS_ROOT, DEFAULT_PCA_COMPONENTS  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two CLS HDBSCAN runs.")
    p.add_argument("--run-a", required=True, help="First cluster folder (e.g. sweep-A / PCA-UMAP).")
    p.add_argument("--run-b", required=True, help="Second cluster folder (e.g. no-PCA UMAP).")
    p.add_argument("--label-a", default=None)
    p.add_argument("--label-b", default=None)
    p.add_argument("--embeddings-run-id", default=None)
    p.add_argument("--pca-components", type=int, default=DEFAULT_PCA_COMPONENTS)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def _labels(assignments: pd.DataFrame) -> np.ndarray:
    return assignments["cluster_id"].to_numpy(dtype=np.int32)


def _mean_prob(assignments: pd.DataFrame) -> float | None:
    if "cluster_probability" not in assignments.columns:
        return None
    assigned = assignments[assignments["cluster_id"] >= 0]["cluster_probability"]
    if assigned.empty:
        return None
    return float(assigned.mean())


def _native(run_dir: Path, assignments: pd.DataFrame, manifest: Dict[str, Any]) -> Dict[str, Any]:
    labels = _labels(assignments)
    space_path = run_dir / "cluster_space.npy"
    native: Dict[str, Any]
    if space_path.exists():
        space = np.load(space_path)
        native = quality_row(space, labels, min_cluster_size=manifest.get("hdbscan_min_cluster_size"))
        native["cluster_space_shape"] = list(space.shape)
    else:
        sizes = size_stats(labels)
        native = {
            "n_images": int(len(labels)),
            "n_clusters": n_clusters(labels),
            "n_noise": n_noise(labels),
            "noise_fraction": noise_fraction(labels),
            "dbcv": manifest.get("dbcv"),
            "silhouette": manifest.get("silhouette"),
            "size_min": sizes["size_min"],
            "size_median": sizes["size_median"],
            "size_p90": sizes["size_p90"],
            "size_max": sizes["size_max"],
            "cluster_space_shape": manifest.get("cluster_space_shape"),
        }
    native["mean_cluster_probability"] = _mean_prob(assignments)
    native["pca_components"] = manifest.get("pca_components")
    native["skip_pca"] = bool(manifest.get("skip_pca", manifest.get("pca_components") == 0))
    native["umap_neighbors"] = manifest.get("umap_neighbors")
    native["umap_min_dist"] = manifest.get("umap_min_dist")
    native["hdbscan_min_cluster_size"] = manifest.get("hdbscan_min_cluster_size")
    native["hdbscan_min_samples"] = manifest.get("hdbscan_min_samples")
    return native


def _noise_crosstab(la: np.ndarray, lb: np.ndarray) -> Dict[str, Any]:
    a_n = la < 0
    b_n = lb < 0
    n = len(la)
    return {
        "both_noise": int((a_n & b_n).sum()),
        "a_noise_b_assigned": int((a_n & ~b_n).sum()),
        "a_assigned_b_noise": int((~a_n & b_n).sum()),
        "both_assigned": int((~a_n & ~b_n).sum()),
        "n": n,
        "frac_both_noise": float((a_n & b_n).mean()),
        "frac_both_assigned": float((~a_n & ~b_n).mean()),
    }


def main() -> None:
    args = parse_args()
    dir_a = resolve_clusters_run(args.run_a)
    dir_b = resolve_clusters_run(args.run_b)
    assign_a, man_a = load_cluster_run(dir_a)
    assign_b, man_b = load_cluster_run(dir_b)
    label_a = args.label_a or dir_a.name
    label_b = args.label_b or dir_b.name

    emb_id = args.embeddings_run_id or man_a.get("embeddings_run_id") or man_b.get("embeddings_run_id")
    if not emb_id:
        print("Pass --embeddings-run-id (neither manifest has embeddings_run_id).")
        sys.exit(1)
    emb_dir = resolve_embeddings_run(run_id=str(emb_id))
    embeddings, image_ids, _ = load_embedding_run(emb_dir)
    order = {i: k for k, i in enumerate(image_ids)}
    for frame, name in ((assign_a, label_a), (assign_b, label_b)):
        missing = [i for i in frame["image_id"].astype(str) if i not in order]
        if missing:
            raise KeyError(f"{len(missing)} ids from {name} not in embeddings (e.g. {missing[:3]})")

    native_a = _native(dir_a, assign_a, man_a)
    native_b = _native(dir_b, assign_b, man_b)

    merged = assign_a[["image_id", "cluster_id"]].merge(
        assign_b[["image_id", "cluster_id"]],
        on="image_id",
        suffixes=("_a", "_b"),
        how="inner",
    )
    if len(merged) != len(assign_a) or len(merged) != len(assign_b):
        raise ValueError(
            f"image_id mismatch: A={len(assign_a)} B={len(assign_b)} inner={len(merged)}"
        )
    la = merged["cluster_id_a"].to_numpy(dtype=np.int32)
    lb = merged["cluster_id_b"].to_numpy(dtype=np.int32)
    idx = [order[str(i)] for i in merged["image_id"].astype(str)]
    X = embeddings[idx]
    pca_50, n_pca, explained = pca_reduce(X, pca_components=args.pca_components, seed=42)

    shared = {
        "raw_cls_cosine_silhouette": {
            label_a: silhouette_assigned(X, la, metric="cosine"),
            label_b: silhouette_assigned(X, lb, metric="cosine"),
        },
        "pca50_euclidean_silhouette": {
            label_a: silhouette_assigned(pca_50, la, metric="euclidean"),
            label_b: silhouette_assigned(pca_50, lb, metric="euclidean"),
        },
        "pca_components": n_pca,
        "pca_explained_variance_ratio": explained,
    }

    agreement = pairwise_stability([la, lb])
    noise_overlap = _noise_crosstab(la, lb)

    report: Dict[str, Any] = {
        "run_a": dir_a.name,
        "run_b": dir_b.name,
        "label_a": label_a,
        "label_b": label_b,
        "embeddings_run_id": emb_dir.name,
        "n_images": int(len(la)),
        "native": {label_a: native_a, label_b: native_b},
        "shared_space": shared,
        "agreement": {
            "ari": agreement["ari_mean"],
            "nmi": agreement["nmi_mean"],
            "ari_assigned": agreement["ari_assigned_mean"],
            "nmi_assigned": agreement["nmi_assigned_mean"],
        },
        "noise_overlap": noise_overlap,
        "note": (
            "Native DBCV/silhouette are in each run's own cluster space. "
            "Shared-space silhouette is the fairer quality comparison. "
            "ARI/NMI measure agreement, not which cut is better."
        ),
    }

    print(f"{label_a} vs {label_b}  n={len(la)}")
    print("\nNative (own UMAP/HDBSCAN space)")
    cols = [
        "n_clusters",
        "noise_fraction",
        "dbcv",
        "silhouette",
        "size_median",
        "size_max",
        "mean_cluster_probability",
        "pca_components",
        "umap_neighbors",
        "umap_min_dist",
        "hdbscan_min_cluster_size",
        "hdbscan_min_samples",
    ]
    table = pd.DataFrame({label_a: native_a, label_b: native_b}).T
    print(table[cols].to_string())
    print("\nShared-space silhouette (higher = tighter groups in that geometry)")
    print(
        f"  raw CLS cosine:  {label_a}={shared['raw_cls_cosine_silhouette'][label_a]}  "
        f"{label_b}={shared['raw_cls_cosine_silhouette'][label_b]}"
    )
    print(
        f"  PCA-{n_pca} euclidean: {label_a}={shared['pca50_euclidean_silhouette'][label_a]}  "
        f"{label_b}={shared['pca50_euclidean_silhouette'][label_b]}"
    )
    print("\nAgreement")
    print(
        f"  ARI={agreement['ari_mean']:.3f}  NMI={agreement['nmi_mean']:.3f}  "
        f"assigned-only ARI={agreement['ari_assigned_mean']}  "
        f"NMI={agreement['nmi_assigned_mean']}"
    )
    print(
        f"  both assigned={noise_overlap['both_assigned']}  "
        f"both noise={noise_overlap['both_noise']}  "
        f"A-only noise={noise_overlap['a_noise_b_assigned']}  "
        f"B-only noise={noise_overlap['a_assigned_b_noise']}"
    )

    out_dir = args.out_dir or (CLS_CLUSTERS_ROOT / f"compare-{dir_a.name}-vs-{dir_b.name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "comparison.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    table[cols].to_csv(out_dir / "native_metrics.csv")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()

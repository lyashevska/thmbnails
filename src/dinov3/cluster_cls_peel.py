#!/usr/bin/env python3
"""
Re-cluster HDBSCAN noise from a parent CLS run (refit PCA + n-D UMAP on leftovers,
or raw CLS → UMAP if the parent used --skip-pca).

Default recipe for Sweep A: inherit knobs from the parent manifest
(10-D UMAP, n_neighbors=15, min_dist=0, eom, min_cluster_size=20, min_samples=20).
Each round writes a new cluster folder. Remaining noise can be peeled again with
--rounds N.

Examples:
    python src/dinov3/cluster_cls_peel.py \\
      --from-clusters-run-id sweep-A-n15-mcs20-ms20

    python src/dinov3/cluster_cls_peel.py \\
      --from-clusters-run-id sweep-A-n15-mcs20-ms20 --rounds 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dinov3.cluster import (  # noqa: E402
    CLUSTER_SPACES,
    build_assignments_frame,
    build_metadata_frame,
    cluster_summary,
    combine_peel_assignments,
    load_cluster_run,
    load_embedding_run,
    noise_image_ids,
    resolve_clusters_run,
    resolve_embeddings_run,
    run_cluster_pipeline,
    save_cluster_sample_grids,
    save_umap_plot,
    subset_embeddings,
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Peel HDBSCAN noise from a CLS cluster run (refit UMAP on leftovers)."
    )
    p.add_argument(
        "--from-clusters-run-id",
        required=True,
        help="Parent cluster folder under data/dinov3_cls_clusters/ (e.g. sweep-A-n15-mcs20-ms20).",
    )
    p.add_argument(
        "--embeddings-run-id",
        default=None,
        help="CLS embedding run. Default: parent manifest embeddings_run_id.",
    )
    p.add_argument("--out-dir", type=Path, default=CLS_CLUSTERS_ROOT)
    p.add_argument("--csv", type=Path, default=CSV_DEFAULT)
    p.add_argument("--thumb-dir", type=Path, default=THUMB_DIR_DEFAULT)
    p.add_argument("--rounds", type=int, default=1, help="How many successive peels (default 1).")
    p.add_argument(
        "--run-id",
        default=None,
        help="First peel folder name (default: <parent>-r1). Later rounds append -r2, …",
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


def peel_run_id(args: argparse.Namespace, parent_name: str, round_n: int) -> str:
    if round_n == 1 and args.run_id:
        return args.run_id
    if args.run_id and round_n > 1:
        return f"{args.run_id}-r{round_n}"
    return f"{parent_name}-r{round_n}"


def write_cluster_run(
    *,
    args: argparse.Namespace,
    knobs: Dict[str, Any],
    run_id: str,
    parent_name: str,
    round_n: int,
    embeddings_run: str,
    embeddings_model: Any,
    embeddings: np.ndarray,
    image_ids: List[str],
    result: Any,
    quality: Dict[str, Any],
) -> Path:
    metadata = build_metadata_frame(image_ids, csv_path=args.csv)
    assignments = build_assignments_frame(image_ids, result, metadata)
    summary = cluster_summary(assignments)

    out_dir = args.out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(out_dir / "cluster_assignments.csv", index=False)
    summary.to_csv(out_dir / "cluster_summary.csv", index=False)
    np.save(out_dir / "cluster_space.npy", result.cluster_embeddings)

    save_umap_plot(assignments, out_dir / "umap.png", title=f"Noise peel round {round_n} (UMAP)")
    sample_info = save_cluster_sample_grids(
        assignments,
        out_dir / "samples",
        thumb_dir=args.thumb_dir,
        samples_per_cluster=args.samples_per_cluster,
    )

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "clustering_type": CLS_CLUSTERING_TYPE,
        "kind": "cls_noise_peel",
        "parent_clusters_run_id": parent_name,
        "round": round_n,
        "n_input": len(image_ids),
        "n_new_clusters": quality["n_clusters"],
        "n_noise": quality["n_noise"],
        "comparison": {
            "patch_clustering_type": PATCH_CLUSTERING_TYPE,
            "patch_clusters_dir": str(PATCH_CLUSTERS_ROOT),
            "note": "Peel refits PCA+UMAP on leftover thumbnails only; not the parent map.",
        },
        "embeddings_run_id": embeddings_run,
        "embeddings_model": embeddings_model,
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
        "noise_fraction": quality["noise_fraction"],
        "dbcv": quality["dbcv"],
        "silhouette": quality["silhouette"],
        "size_median": quality["size_median"],
        "cluster_sample_info": sample_info,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"  wrote {out_dir}")
    print(summary.to_string(index=False))
    return out_dir


def main() -> None:
    args = parse_args()
    if args.rounds < 1:
        print("--rounds must be >= 1")
        sys.exit(1)

    parent_dir = resolve_clusters_run(args.from_clusters_run_id)
    parent_assign, parent_man = load_cluster_run(parent_dir)
    knobs = knobs_from_parent(args, parent_man)
    emb_id = args.embeddings_run_id or parent_man.get("embeddings_run_id")
    if not emb_id:
        print("Pass --embeddings-run-id (parent manifest has none).")
        sys.exit(1)

    print("Step 1: Load parent noise and embeddings")
    print(f"  parent={parent_dir.name}")
    noise_ids = noise_image_ids(parent_assign)
    print(f"  parent images={len(parent_assign)}  noise={len(noise_ids)}")
    print(
        f"  knobs: space={knobs['cluster_space']} umap={knobs['umap_components']}D "
        f"n_neighbors={knobs['umap_neighbors']} min_dist={knobs['umap_min_dist']} "
        f"mcs={knobs['hdbscan_min_cluster_size']} ms={knobs['hdbscan_min_samples']} "
        f"{knobs['hdbscan_selection_method']}"
    )

    emb_run_dir = resolve_embeddings_run(run_id=str(emb_id))
    embeddings, image_ids, emb_manifest = load_embedding_run(emb_run_dir)
    print(f"  embeddings_run={emb_run_dir.name}  shape={embeddings.shape}")

    if args.dry_run:
        print(f"Dry run: would peel {len(noise_ids)} images, rounds={args.rounds}")
        for i, image_id in enumerate(noise_ids[:10]):
            print(f"  - {image_id}")
        if len(noise_ids) > 10:
            print(f"  ... and {len(noise_ids) - 10} more")
        return

    if len(noise_ids) < knobs["hdbscan_min_cluster_size"] * 2:
        print(
            f"Not enough noise ({len(noise_ids)}) for min_cluster_size="
            f"{knobs['hdbscan_min_cluster_size']}."
        )
        sys.exit(1)

    current_ids = noise_ids
    peels: List[Tuple[int, str, pd.DataFrame]] = []
    last_parent_name = parent_dir.name

    for round_n in range(1, args.rounds + 1):
        src = "raw CLS" if knobs["pca_components"] <= 0 else "PCA"
        print(f"\nStep 2.{round_n}: Refit {src} → UMAP → HDBSCAN on {len(current_ids)} leftovers")
        subset, subset_ids = subset_embeddings(embeddings, image_ids, current_ids)
        result = run_cluster_pipeline(
            subset,
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
        if quality["n_clusters"] == 0:
            print("  no new clusters; stopping.")
            break

        run_id = peel_run_id(args, parent_dir.name, round_n)
        metadata = build_metadata_frame(subset_ids, csv_path=args.csv)
        round_assign = build_assignments_frame(subset_ids, result, metadata)
        write_cluster_run(
            args=args,
            knobs=knobs,
            run_id=run_id,
            parent_name=last_parent_name,
            round_n=round_n,
            embeddings_run=emb_run_dir.name,
            embeddings_model=emb_manifest.get("model_id"),
            embeddings=subset,
            image_ids=subset_ids,
            result=result,
            quality=quality,
        )
        peels.append((round_n, run_id, round_assign))
        last_parent_name = run_id
        current_ids = noise_image_ids(round_assign)
        if len(current_ids) < knobs["hdbscan_min_cluster_size"] * 2:
            print(f"  remaining noise {len(current_ids)} is too small for another round.")
            break

    if not peels:
        return

    print("\nStep 3: Combined assignments (offset cluster ids)")
    combined = combine_peel_assignments(
        parent_assign,
        parent_run_id=parent_dir.name,
        peels=peels,
    )
    combo_dir = args.out_dir / f"{parent_dir.name}-peels"
    combo_dir.mkdir(parents=True, exist_ok=True)
    combined_path = combo_dir / "combined_assignments.csv"
    combined.to_csv(combined_path, index=False)
    combo_summary = cluster_summary(combined)
    combo_summary.to_csv(combo_dir / "cluster_summary.csv", index=False)
    n_assigned = int((combined["cluster_id"] >= 0).sum())
    n_noise = int((combined["cluster_id"] == -1).sum())
    n_by_round = {
        str(int(k)): int(v) for k, v in combined.groupby("round").size().items()
    }
    combo_manifest = {
        "kind": "cls_noise_peel_combined",
        "parent_clusters_run_id": parent_dir.name,
        "peel_run_ids": [p[1] for p in peels],
        "n_rounds": len(peels),
        "n_images": int(len(combined)),
        "n_assigned": n_assigned,
        "n_noise": n_noise,
        "n_clusters": int(combined.loc[combined["cluster_id"] >= 0, "cluster_id"].nunique()),
        "n_by_round": n_by_round,
        "note": (
            "cluster_id is unique across rounds (offset). round=0 is the parent cut; "
            "round>=1 is a peel. Remaining noise has cluster_id=-1 and round=-1. "
            "umap_x/umap_y for peeled rows are from that round's UMAP, not the parent map."
        ),
    }
    (combo_dir / "manifest.json").write_text(json.dumps(combo_manifest, indent=2), encoding="utf-8")
    print(f"  {combined_path}")
    print(f"  assigned={n_assigned}  still_noise={n_noise}")
    print("\nDone. Inspect each peel's samples/cluster_*/_grid.jpg before accepting new groups.")


if __name__ == "__main__":
    main()

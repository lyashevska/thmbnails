#!/usr/bin/env python3
"""
Grid search for CLS HDBSCAN in 10-D UMAP (no plots).

PCA is fit once (or skipped with --skip-pca). Each unique (n_neighbors,
min_dist) UMAP is fit once, then HDBSCAN varies. Cells are ranked by min-max
DBCV 50% / silhouette 30% / (1 - noise) 20%. Inspect sample grids for the
shortlist; do not treat the composite as a final "best" cluster.

Examples:
    python src/dinov3/cluster_cls_sweep.py --embeddings-run-id 20260713T131720Z --dry-run
    python src/dinov3/cluster_cls_sweep.py --embeddings-run-id 20260713T131720Z
    python src/dinov3/cluster_cls_sweep.py --embeddings-run-id 20260713T131720Z --skip-pca
    python src/dinov3/cluster_cls_sweep.py --embeddings-run-id 20260713T131720Z --limit 400
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.dinov3.cluster import (  # noqa: E402
    cluster_in_space,
    load_embedding_run,
    pca_reduce,
    resolve_embeddings_run,
    run_id_now,
    umap_reduce,
)
from src.dinov3.cluster_metrics import add_composite, quality_row  # noqa: E402
from src.dinov3.config import (  # noqa: E402
    CLS_CLUSTERS_ROOT,
    DEFAULT_HDBSCAN_SELECTION_METHOD,
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_UMAP_CLUSTER_COMPONENTS,
    SWEEP_DBCV_WEIGHT,
    SWEEP_NOISE_WEIGHT,
    SWEEP_SILHOUETTE_WEIGHT,
)

UMAP_NEIGHBORS_GRID = (15, 30, 50)
UMAP_MIN_DIST_GRID = (0.0, 0.1, 0.25)
MIN_CLUSTER_SIZE_GRID = (10, 20, 30, 50)
MIN_SAMPLES_GRID = (5, 10, "mcs")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep UMAP + HDBSCAN knobs on CLS embeddings.")
    p.add_argument("--embeddings-run-id", default=None)
    p.add_argument("--out-dir", type=Path, default=CLS_CLUSTERS_ROOT / "sweeps")
    p.add_argument(
        "--pca-components",
        type=int,
        default=DEFAULT_PCA_COMPONENTS,
        help="PCA width. 0 skips PCA and feeds raw CLS to UMAP.",
    )
    p.add_argument(
        "--skip-pca",
        action="store_true",
        help="Same as --pca-components 0.",
    )
    p.add_argument("--umap-components", type=int, default=DEFAULT_UMAP_CLUSTER_COMPONENTS)
    p.add_argument(
        "--hdbscan-selection-method",
        choices=("eom", "leaf"),
        default=DEFAULT_HDBSCAN_SELECTION_METHOD,
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="Optional row cap for a smoke test.")
    p.add_argument("--run-id", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--top-k", type=int, default=5, help="How many composite leaders to print.")
    return p.parse_args()


def hdbscan_grid() -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    seen: set[Tuple[int, int]] = set()
    for mcs, ms in product(MIN_CLUSTER_SIZE_GRID, MIN_SAMPLES_GRID):
        min_samples = mcs if ms == "mcs" else int(ms)
        key = (mcs, min_samples)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)
    return pairs


def cell_count() -> int:
    return len(UMAP_NEIGHBORS_GRID) * len(UMAP_MIN_DIST_GRID) * len(hdbscan_grid())


def main() -> None:
    args = parse_args()
    if args.skip_pca:
        args.pca_components = 0
    n_cells = cell_count()
    source = "raw CLS" if args.pca_components <= 0 else f"PCA-{args.pca_components}"
    print(
        f"Grid: {n_cells} cells  source={source}  umap_components={args.umap_components}  "
        f"method=hdbscan/{args.hdbscan_selection_method}"
    )
    print(f"  n_neighbors={list(UMAP_NEIGHBORS_GRID)}  min_dist={list(UMAP_MIN_DIST_GRID)}")
    print(f"  min_cluster_size={list(MIN_CLUSTER_SIZE_GRID)}  min_samples={MIN_SAMPLES_GRID}")

    if args.dry_run:
        return

    emb_run_dir = resolve_embeddings_run(run_id=args.embeddings_run_id)
    embeddings, image_ids, emb_manifest = load_embedding_run(emb_run_dir)
    if args.limit:
        embeddings = embeddings[: args.limit]
        image_ids = image_ids[: args.limit]
    if args.pca_components <= 0:
        print("\nStep 1: Load embeddings (no PCA)")
        umap_source = embeddings
        n_pca = 0
        explained = 1.0
        print(f"  embeddings_run={emb_run_dir.name}  n={len(image_ids)}  pca=skipped")
    else:
        print("\nStep 1: Load embeddings and fit PCA once")
        umap_source, n_pca, explained = pca_reduce(
            embeddings, pca_components=args.pca_components, seed=args.seed
        )
        print(
            f"  embeddings_run={emb_run_dir.name}  n={len(image_ids)}  "
            f"pca={n_pca}  var={explained:.3f}"
        )

    run_id = args.run_id or run_id_now()
    out_dir = args.out_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    hdb_pairs = hdbscan_grid()
    cell_i = 0

    print("\nStep 2: UMAP (cached) → HDBSCAN")
    for n_neighbors, min_dist in product(UMAP_NEIGHBORS_GRID, UMAP_MIN_DIST_GRID):
        print(f"  UMAP n_neighbors={n_neighbors} min_dist={min_dist} ...")
        cluster_embeddings = umap_reduce(
            umap_source,
            n_components=args.umap_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            seed=args.seed,
        )

        for min_cluster_size, min_samples in hdb_pairs:
            cell_i += 1
            labels, _ = cluster_in_space(
                cluster_embeddings,
                method="hdbscan",
                hdbscan_min_cluster_size=min_cluster_size,
                hdbscan_min_samples=min_samples,
                hdbscan_selection_method=args.hdbscan_selection_method,
                seed=args.seed,
            )
            row = quality_row(
                cluster_embeddings,
                labels,
                min_cluster_size=min_cluster_size,
            )
            row.update(
                {
                    "umap_neighbors": n_neighbors,
                    "umap_min_dist": min_dist,
                    "umap_components": int(cluster_embeddings.shape[1]),
                    "hdbscan_min_cluster_size": min_cluster_size,
                    "hdbscan_min_samples": min_samples,
                    "hdbscan_selection_method": args.hdbscan_selection_method,
                }
            )
            rows.append(row)
            print(
                f"    [{cell_i}/{n_cells}] mcs={min_cluster_size} ms={min_samples} "
                f"k={row['n_clusters']} noise={row['noise_fraction']:.2f} "
                f"dbcv={row['dbcv']} sil={row['silhouette']}"
            )

    print("\nStep 3: Rank by 50/30/20 composite")
    sweep = add_composite(
        pd.DataFrame(rows),
        dbcv_weight=SWEEP_DBCV_WEIGHT,
        silhouette_weight=SWEEP_SILHOUETTE_WEIGHT,
        noise_weight=SWEEP_NOISE_WEIGHT,
    )
    sweep_path = out_dir / "sweep.csv"
    sweep.to_csv(sweep_path, index=False)

    top = sweep.head(args.top_k)
    cols = [
        "composite",
        "dbcv",
        "silhouette",
        "noise_fraction",
        "n_clusters",
        "umap_neighbors",
        "umap_min_dist",
        "hdbscan_min_cluster_size",
        "hdbscan_min_samples",
        "size_median",
    ]
    print("Composite leaders (often 2-cluster splits; not the operating cut):")
    print(top[cols].to_string(index=False))

    usable = sweep[(sweep["n_clusters"] >= 20) & (sweep["n_clusters"] <= 80)]
    print(f"\nUsable band (n_clusters 20–80): {len(usable)} cells")
    if usable.empty:
        print("  none")
        usable_top = usable
    else:
        usable_top = usable.head(args.top_k)
        print(usable_top[cols].to_string(index=False))

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "kind": "cls_hdbscan_sweep",
        "embeddings_run_id": emb_run_dir.name,
        "embeddings_model": emb_manifest.get("model_id"),
        "n_images": len(image_ids),
        "pca_components": n_pca,
        "skip_pca": n_pca == 0,
        "explained_variance_ratio": explained,
        "umap_components": args.umap_components,
        "cluster_space": "umap",
        "hdbscan_selection_method": args.hdbscan_selection_method,
        "seed": args.seed,
        "n_cells": n_cells,
        "weights": {
            "dbcv": SWEEP_DBCV_WEIGHT,
            "silhouette": SWEEP_SILHOUETTE_WEIGHT,
            "noise_coverage": SWEEP_NOISE_WEIGHT,
        },
        "grid": {
            "umap_neighbors": list(UMAP_NEIGHBORS_GRID),
            "umap_min_dist": list(UMAP_MIN_DIST_GRID),
            "min_cluster_size": list(MIN_CLUSTER_SIZE_GRID),
            "min_samples": [x if x != "mcs" else "min_cluster_size" for x in MIN_SAMPLES_GRID],
        },
        "top": top[cols].to_dict(orient="records"),
        "usable_top": usable_top[cols].to_dict(orient="records") if len(usable_top) else [],
        "note": (
            "Composite ranks the grid only. Inspect umap.png and sample grids for the "
            "shortlist. Prefer tight cores with leftover mass if a later noise peel is planned. "
            "Do not mix these labels with PCA-space runs such as hdbscan-eom-vitl."
        ),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote {sweep_path}")
    print(f"Wrote {out_dir / 'manifest.json'}")
    print("Next: cluster_cls.py with the shortlist knobs, then inspect samples/ and umap.png.")


if __name__ == "__main__":
    main()

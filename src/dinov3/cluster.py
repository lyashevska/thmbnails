"""
PCA → UMAP → HDBSCAN clustering for DINOv3 CLS embeddings.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from .config import (
    CLUSTERS_ROOT,
    CSV_DEFAULT,
    DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
    DEFAULT_HDBSCAN_MIN_SAMPLES,
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_SAMPLES_PER_CLUSTER,
    DEFAULT_UMAP_MIN_DIST,
    DEFAULT_UMAP_NEIGHBORS,
    EMBEDDINGS_ROOT,
    THUMB_DIR_DEFAULT,
)

try:
    import hdbscan
    import matplotlib.pyplot as plt
    import umap
    from sklearn.decomposition import PCA
except ImportError as exc:
    hdbscan = None  # type: ignore
    plt = None  # type: ignore
    umap = None  # type: ignore
    PCA = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class ClusterPipelineResult:
    labels: np.ndarray
    probabilities: np.ndarray
    pca_components: int
    pca_embeddings: np.ndarray
    umap_2d: np.ndarray
    explained_variance_ratio: float


def _require_cluster_deps() -> None:
    if _IMPORT_ERROR is not None:
        raise ImportError(
            "Clustering requires scikit-learn, umap-learn, hdbscan, and matplotlib. "
            "Install with: pip install scikit-learn umap-learn hdbscan matplotlib"
        ) from _IMPORT_ERROR


def resolve_embeddings_run(
    *,
    run_id: Optional[str] = None,
    embeddings_root: Path = EMBEDDINGS_ROOT,
) -> Path:
    if run_id:
        run_dir = embeddings_root / run_id
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Embeddings run not found: {run_dir}")
        return run_dir

    candidates = [p for p in embeddings_root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No embedding runs under {embeddings_root}")

    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_embedding_run(run_dir: Path) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
    emb_path = run_dir / "cls_embeddings.npy"
    ids_path = run_dir / "image_ids.json"
    manifest_path = run_dir / "manifest.json"

    if not emb_path.exists() or not ids_path.exists():
        raise FileNotFoundError(f"Missing cls_embeddings.npy or image_ids.json in {run_dir}")

    embeddings = np.load(emb_path)
    image_ids = json.loads(ids_path.read_text(encoding="utf-8"))
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if embeddings.shape[0] != len(image_ids):
        raise ValueError(
            f"Embedding rows ({embeddings.shape[0]}) != image_ids ({len(image_ids)}) in {run_dir}"
        )

    return embeddings, image_ids, manifest


CLUSTER_METHODS = ("hdbscan", "kmeans", "agglomerative")


def run_cluster_pipeline(
    embeddings: np.ndarray,
    *,
    method: str = "hdbscan",
    pca_components: int = DEFAULT_PCA_COMPONENTS,
    umap_neighbors: int = DEFAULT_UMAP_NEIGHBORS,
    umap_min_dist: float = DEFAULT_UMAP_MIN_DIST,
    hdbscan_min_cluster_size: int = DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
    hdbscan_min_samples: int = DEFAULT_HDBSCAN_MIN_SAMPLES,
    hdbscan_selection_method: str = "leaf",
    n_clusters: int | None = None,
    seed: int = 42,
    compute_umap: bool = True,
) -> ClusterPipelineResult:
    _require_cluster_deps()

    if method not in CLUSTER_METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {CLUSTER_METHODS}")

    n_samples, n_features = embeddings.shape
    n_components = min(pca_components, n_samples, n_features)

    pca = PCA(n_components=n_components, random_state=seed)
    pca_embeddings = pca.fit_transform(embeddings)
    explained = float(np.sum(pca.explained_variance_ratio_))

    if compute_umap:
        reducer = umap.UMAP(
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            n_components=2,
            metric="cosine",
            random_state=seed,
        )
        umap_2d = reducer.fit_transform(pca_embeddings)
    else:
        umap_2d = np.zeros((n_samples, 2), dtype=np.float64)

    if method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            metric="euclidean",
            cluster_selection_method=hdbscan_selection_method,
        )
        labels = clusterer.fit_predict(pca_embeddings)
        probabilities = clusterer.probabilities_
        if probabilities is None:
            probabilities = np.zeros(len(labels), dtype=np.float64)
    else:
        if n_clusters is None or n_clusters < 2:
            raise ValueError(f"{method} requires --n-clusters >= 2")
        if method == "kmeans":
            from sklearn.cluster import KMeans

            model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        else:
            from sklearn.cluster import AgglomerativeClustering

            model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels = model.fit_predict(pca_embeddings)
        probabilities = np.ones(len(labels), dtype=np.float64)

    return ClusterPipelineResult(
        labels=labels,
        probabilities=probabilities,
        pca_components=n_components,
        pca_embeddings=pca_embeddings,
        umap_2d=umap_2d,
        explained_variance_ratio=explained,
    )


def build_metadata_frame(
    image_ids: List[str],
    *,
    csv_path: Path = CSV_DEFAULT,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "thumbnail_path" not in df.columns:
        raise ValueError(f"CSV missing thumbnail_path column: {csv_path}")

    df = df.copy()

    def _image_id_from_path(value: object) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        text = str(value).strip()
        if not text or text.lower() == "nan":
            return ""
        return Path(text).name

    df["image_id"] = df["thumbnail_path"].map(_image_id_from_path)
    df = df[df["image_id"] != ""]
    meta = df.set_index("image_id", drop=False)

    rows = []
    for image_id in image_ids:
        if image_id in meta.index:
            row = meta.loc[image_id]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            rows.append(
                {
                    "image_id": image_id,
                    "title": row.get("title", ""),
                    "year": row.get("year", ""),
                    "categories": row.get("categories", ""),
                    "thumbnail_path": row.get("thumbnail_path", ""),
                }
            )
        else:
            rows.append(
                {
                    "image_id": image_id,
                    "title": "",
                    "year": "",
                    "categories": "",
                    "thumbnail_path": "",
                }
            )

    return pd.DataFrame(rows)


def build_assignments_frame(
    image_ids: List[str],
    result: ClusterPipelineResult,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    out = metadata.copy()
    out["cluster_id"] = result.labels
    out["cluster_probability"] = result.probabilities
    out["umap_x"] = result.umap_2d[:, 0]
    out["umap_y"] = result.umap_2d[:, 1]
    return out


def cluster_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    counts = assignments.groupby("cluster_id").size().reset_index(name="count").sort_values("cluster_id")
    total = len(assignments)
    counts["pct"] = (counts["count"] / total * 100).round(1)
    return counts


def save_umap_plot(
    assignments: pd.DataFrame,
    out_path: Path,
    *,
    title: str = "DINOv3 thumbnail clusters (UMAP)",
) -> None:
    _require_cluster_deps()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    clusters = sorted(assignments["cluster_id"].unique())
    cmap = plt.get_cmap("tab20", max(len(clusters), 1))

    for idx, cluster_id in enumerate(clusters):
        subset = assignments[assignments["cluster_id"] == cluster_id]
        label = "noise" if cluster_id == -1 else f"cluster {cluster_id}"
        color = "#aaaaaa" if cluster_id == -1 else cmap(idx % 20)
        ax.scatter(
            subset["umap_x"],
            subset["umap_y"],
            s=14 if cluster_id == -1 else 18,
            alpha=0.55 if cluster_id == -1 else 0.8,
            label=f"{label} (n={len(subset)})",
            c=[color],
            edgecolors="none",
        )

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize=8, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _resolve_thumb_path(image_id: str, thumb_path: str, thumb_dir: Path) -> Optional[Path]:
    if thumb_path:
        p = Path(str(thumb_path))
        if not p.is_absolute():
            p = Path.cwd() / p
        if p.exists():
            return p
    alt = thumb_dir / image_id
    return alt if alt.exists() else None


def save_cluster_sample_grids(
    assignments: pd.DataFrame,
    out_dir: Path,
    *,
    thumb_dir: Path = THUMB_DIR_DEFAULT,
    samples_per_cluster: int = DEFAULT_SAMPLES_PER_CLUSTER,
    cols: int = 4,
    thumb_height: int = 90,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Any] = {}

    for cluster_id in sorted(assignments["cluster_id"].unique()):
        subset = assignments[assignments["cluster_id"] == cluster_id]
        label = "noise" if cluster_id == -1 else str(int(cluster_id))
        cluster_dir = out_dir / f"cluster_{label}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        # Higher-probability points first; stable tie-break by image_id.
        ordered = subset.sort_values(
            ["cluster_probability", "image_id"],
            ascending=[False, True],
        ).head(samples_per_cluster)

        copied = []
        panels: List[Image.Image] = []
        for _, row in ordered.iterrows():
            image_id = row["image_id"]
            src = _resolve_thumb_path(image_id, str(row.get("thumbnail_path", "")), thumb_dir)
            if src is None:
                continue
            dst = cluster_dir / image_id
            shutil.copy2(src, dst)
            copied.append(image_id)

            with Image.open(src) as im:
                im = im.convert("RGB")
                w, h = im.size
                new_w = max(1, int(round(w * thumb_height / h)))
                panels.append(im.resize((new_w, thumb_height), Image.BICUBIC))

        if panels:
            rows_n = (len(panels) + cols - 1) // cols
            row_widths = []
            for r in range(rows_n):
                row_panels = panels[r * cols : (r + 1) * cols]
                row_widths.append(sum(p.width for p in row_panels))
            grid_w = max(row_widths) if row_widths else 0
            grid_h = rows_n * thumb_height
            grid = Image.new("RGB", (grid_w, grid_h), (0, 0, 0))
            y = 0
            for r in range(rows_n):
                row_panels = panels[r * cols : (r + 1) * cols]
                x = 0
                for panel in row_panels:
                    grid.paste(panel, (x, y))
                    x += panel.width
                y += thumb_height
            grid.save(cluster_dir / "_grid.jpg", quality=90)

        saved[label] = {"count": int(len(subset)), "samples_copied": copied}

    return saved


def run_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
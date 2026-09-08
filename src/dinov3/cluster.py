"""
DINOv3 clustering library (CLS thumbnails and patch tokens).

CLIs: cluster_cls.py, cluster_patch.py.
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
    CLS_CLUSTERS_ROOT,
    CLS_EMBEDDINGS_ROOT,
    CSV_DEFAULT,
    DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
    DEFAULT_HDBSCAN_MIN_SAMPLES,
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_PATCH_SIZE,
    DEFAULT_SAMPLES_PER_CLUSTER,
    DEFAULT_UMAP_CLUSTER_COMPONENTS,
    DEFAULT_UMAP_MIN_DIST,
    DEFAULT_UMAP_NEIGHBORS,
    DEFAULT_VIT_PATCH_SIZE,
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


CLUSTER_SPACES = ("pca", "umap")


@dataclass
class ClusterPipelineResult:
    labels: np.ndarray
    probabilities: np.ndarray
    pca_components: int
    pca_embeddings: np.ndarray
    umap_2d: np.ndarray
    explained_variance_ratio: float
    cluster_space: str
    cluster_embeddings: np.ndarray
    umap_cluster_components: int | None


def _require_cluster_deps() -> None:
    if _IMPORT_ERROR is not None:
        raise ImportError(
            "Clustering requires scikit-learn, umap-learn, hdbscan, and matplotlib. "
            "Install with: pip install scikit-learn umap-learn hdbscan matplotlib"
        ) from _IMPORT_ERROR


def resolve_embeddings_run(
    *,
    run_id: Optional[str] = None,
    embeddings_root: Path = CLS_EMBEDDINGS_ROOT,
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


def resolve_clusters_run(
    run_id: str,
    *,
    clusters_root: Path = CLS_CLUSTERS_ROOT,
) -> Path:
    run_dir = clusters_root / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Cluster run not found: {run_dir}")
    return run_dir


def load_cluster_run(run_dir: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    assignments_path = run_dir / "cluster_assignments.csv"
    if not assignments_path.exists():
        raise FileNotFoundError(f"Missing cluster_assignments.csv in {run_dir}")
    assignments = pd.read_csv(assignments_path)
    if "image_id" not in assignments.columns or "cluster_id" not in assignments.columns:
        raise ValueError(f"{assignments_path} needs image_id and cluster_id columns")
    manifest: Dict[str, Any] = {}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return assignments, manifest


def noise_image_ids(assignments: pd.DataFrame) -> List[str]:
    noise = assignments[assignments["cluster_id"] == -1]
    return [str(i) for i in noise["image_id"].tolist()]


def subset_embeddings(
    embeddings: np.ndarray,
    image_ids: List[str],
    keep_ids: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Keep rows whose image_id is in keep_ids, in corpus order."""
    index = {image_id: i for i, image_id in enumerate(image_ids)}
    missing = [k for k in keep_ids if k not in index]
    if missing:
        raise KeyError(f"{len(missing)} noise ids not in the embedding run (e.g. {missing[:3]})")
    keep_set = set(keep_ids)
    order = [i for i, image_id in enumerate(image_ids) if image_id in keep_set]
    subset_ids = [image_ids[i] for i in order]
    return embeddings[order], subset_ids


def next_cluster_id_offset(assignments: pd.DataFrame) -> int:
    assigned = assignments.loc[assignments["cluster_id"] >= 0, "cluster_id"]
    if assigned.empty:
        return 0
    return int(assigned.max()) + 1


def combine_peel_assignments(
    parent: pd.DataFrame,
    *,
    parent_run_id: str,
    peels: List[Tuple[int, str, pd.DataFrame]],
) -> pd.DataFrame:
    """Merge peel rounds onto the parent table. Cluster ids are offset so rounds do not collide.

    ``peels`` is a list of (round_number, run_id, assignments_for_that_round_input).
    UMAP coordinates for peeled points come from that round (not comparable to round 0).
    """
    out = parent.copy()
    if "round" not in out.columns:
        out["round"] = np.where(out["cluster_id"] >= 0, 0, -1)
    else:
        out.loc[out["cluster_id"] < 0, "round"] = -1
    if "cluster_id_in_round" not in out.columns:
        out["cluster_id_in_round"] = out["cluster_id"]
    if "source_run" not in out.columns:
        out["source_run"] = np.where(out["cluster_id"] >= 0, parent_run_id, "")

    by_id = {str(i): idx for idx, i in enumerate(out["image_id"].astype(str))}
    for round_n, run_id, peel_df in peels:
        offset = next_cluster_id_offset(out)
        for _, row in peel_df.iterrows():
            image_id = str(row["image_id"])
            idx = by_id.get(image_id)
            if idx is None:
                continue
            label = int(row["cluster_id"])
            if label < 0:
                continue
            out.iat[idx, out.columns.get_loc("cluster_id")] = offset + label
            out.iat[idx, out.columns.get_loc("round")] = round_n
            out.iat[idx, out.columns.get_loc("cluster_id_in_round")] = label
            out.iat[idx, out.columns.get_loc("source_run")] = run_id
            if "cluster_probability" in out.columns and "cluster_probability" in peel_df.columns:
                out.iat[idx, out.columns.get_loc("cluster_probability")] = row["cluster_probability"]
            if "umap_x" in out.columns and "umap_x" in peel_df.columns:
                out.iat[idx, out.columns.get_loc("umap_x")] = row["umap_x"]
                out.iat[idx, out.columns.get_loc("umap_y")] = row["umap_y"]
    return out


CLUSTER_METHODS = ("hdbscan", "kmeans", "agglomerative")


def pca_reduce(
    embeddings: np.ndarray,
    *,
    pca_components: int = DEFAULT_PCA_COMPONENTS,
    seed: int = 42,
) -> Tuple[np.ndarray, int, float]:
    """Return (pca_embeddings, n_components, explained_variance_ratio)."""
    _require_cluster_deps()
    if pca_components < 1:
        raise ValueError("pca_reduce requires pca_components >= 1; use 0 on the pipeline to skip PCA")
    n_samples, n_features = embeddings.shape
    n_components = min(pca_components, n_samples, n_features)
    pca = PCA(n_components=n_components, random_state=seed)
    pca_embeddings = pca.fit_transform(embeddings)
    explained = float(np.sum(pca.explained_variance_ratio_))
    return pca_embeddings, n_components, explained


def umap_reduce(
    features: np.ndarray,
    *,
    n_components: int,
    n_neighbors: int = DEFAULT_UMAP_NEIGHBORS,
    min_dist: float = DEFAULT_UMAP_MIN_DIST,
    seed: int = 42,
) -> np.ndarray:
    """UMAP on PCA coordinates or raw embeddings (cosine)."""
    _require_cluster_deps()
    n_neighbors = min(n_neighbors, max(2, features.shape[0] - 1))
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=min(n_components, features.shape[0] - 1),
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(features)


def cluster_in_space(
    cluster_embeddings: np.ndarray,
    *,
    method: str = "hdbscan",
    hdbscan_min_cluster_size: int = DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
    hdbscan_min_samples: int = DEFAULT_HDBSCAN_MIN_SAMPLES,
    hdbscan_selection_method: str = "eom",
    n_clusters: int | None = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit labels in an already-reduced space. Returns (labels, probabilities)."""
    _require_cluster_deps()
    if method not in CLUSTER_METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {CLUSTER_METHODS}")

    if method == "hdbscan":
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=hdbscan_min_cluster_size,
            min_samples=hdbscan_min_samples,
            metric="euclidean",
            cluster_selection_method=hdbscan_selection_method,
        )
        labels = clusterer.fit_predict(cluster_embeddings)
        probabilities = clusterer.probabilities_
        if probabilities is None:
            probabilities = np.zeros(len(labels), dtype=np.float64)
        return labels, probabilities

    if n_clusters is None or n_clusters < 2:
        raise ValueError(f"{method} requires --n-clusters >= 2")
    if method == "kmeans":
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    else:
        from sklearn.cluster import AgglomerativeClustering

        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(cluster_embeddings)
    probabilities = np.ones(len(labels), dtype=np.float64)
    return labels, probabilities


def run_cluster_pipeline(
    embeddings: np.ndarray,
    *,
    method: str = "hdbscan",
    cluster_space: str = "pca",
    pca_components: int = DEFAULT_PCA_COMPONENTS,
    umap_cluster_components: int = DEFAULT_UMAP_CLUSTER_COMPONENTS,
    umap_neighbors: int = DEFAULT_UMAP_NEIGHBORS,
    umap_min_dist: float = DEFAULT_UMAP_MIN_DIST,
    hdbscan_min_cluster_size: int = DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
    hdbscan_min_samples: int = DEFAULT_HDBSCAN_MIN_SAMPLES,
    hdbscan_selection_method: str = "eom",
    n_clusters: int | None = None,
    seed: int = 42,
    compute_umap: bool = True,
    pca_embeddings: np.ndarray | None = None,
    explained_variance_ratio: float | None = None,
    fitted_pca_components: int | None = None,
) -> ClusterPipelineResult:
    """Optionally PCA, then cluster in that space or in n-D UMAP. 2D UMAP is for plots only.

    ``cluster_space='pca'`` is the older CLS/patch path (HDBSCAN on PCA).
    ``cluster_space='umap'`` is the CLS path (HDBSCAN on n-D UMAP).
    ``pca_components <= 0`` skips PCA and feeds raw embeddings to UMAP
    (``cluster_space='umap'`` only). Pass precomputed ``pca_embeddings`` to skip
    refitting PCA (sweeps).
    """
    _require_cluster_deps()

    if method not in CLUSTER_METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {CLUSTER_METHODS}")
    if cluster_space not in CLUSTER_SPACES:
        raise ValueError(f"Unknown cluster_space {cluster_space!r}; choose from {CLUSTER_SPACES}")

    skip_pca = pca_components <= 0
    if skip_pca and cluster_space == "pca":
        raise ValueError("cluster_space='pca' requires pca_components >= 1")

    n_samples = embeddings.shape[0]
    if skip_pca:
        if pca_embeddings is None:
            source = embeddings
        else:
            if pca_embeddings.shape[0] != n_samples:
                raise ValueError("pca_embeddings row count does not match embeddings")
            source = pca_embeddings
        n_pca = 0
        explained = 1.0
    elif pca_embeddings is None:
        source, n_pca, explained = pca_reduce(
            embeddings, pca_components=pca_components, seed=seed
        )
    else:
        if pca_embeddings.shape[0] != n_samples:
            raise ValueError("pca_embeddings row count does not match embeddings")
        source = pca_embeddings
        n_pca = int(fitted_pca_components or pca_embeddings.shape[1])
        explained = float(explained_variance_ratio if explained_variance_ratio is not None else 0.0)

    umap_2d: np.ndarray
    cluster_embeddings: np.ndarray
    umap_dims: int | None = None

    if cluster_space == "umap":
        n_umap = min(umap_cluster_components, source.shape[1], max(2, n_samples - 1))
        cluster_embeddings = umap_reduce(
            source,
            n_components=n_umap,
            n_neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            seed=seed,
        )
        umap_dims = int(cluster_embeddings.shape[1])
        if compute_umap:
            if umap_dims == 2:
                umap_2d = cluster_embeddings
            else:
                umap_2d = umap_reduce(
                    source,
                    n_components=2,
                    n_neighbors=umap_neighbors,
                    min_dist=umap_min_dist,
                    seed=seed,
                )
        else:
            umap_2d = np.zeros((n_samples, 2), dtype=np.float64)
    else:
        cluster_embeddings = source
        if compute_umap:
            umap_2d = umap_reduce(
                source,
                n_components=2,
                n_neighbors=umap_neighbors,
                min_dist=umap_min_dist,
                seed=seed,
            )
        else:
            umap_2d = np.zeros((n_samples, 2), dtype=np.float64)

    labels, probabilities = cluster_in_space(
        cluster_embeddings,
        method=method,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_samples=hdbscan_min_samples,
        hdbscan_selection_method=hdbscan_selection_method,
        n_clusters=n_clusters,
        seed=seed,
    )

    return ClusterPipelineResult(
        labels=labels,
        probabilities=probabilities,
        pca_components=n_pca,
        pca_embeddings=source,
        umap_2d=umap_2d,
        explained_variance_ratio=explained,
        cluster_space=cluster_space,
        cluster_embeddings=cluster_embeddings,
        umap_cluster_components=umap_dims,
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
    title: str = "DINOv3 clusters (UMAP)",
    point_size: int = 18,
    max_legend: int = 20,
) -> None:
    _require_cluster_deps()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    clusters = sorted(assignments["cluster_id"].unique())
    cmap = plt.get_cmap("tab20", max(len(clusters), 1))
    show_legend = len(clusters) <= max_legend

    for idx, cluster_id in enumerate(clusters):
        subset = assignments[assignments["cluster_id"] == cluster_id]
        label = "noise" if cluster_id == -1 else f"cluster {cluster_id}"
        color = "#aaaaaa" if cluster_id == -1 else cmap(idx % 20)
        ax.scatter(
            subset["umap_x"],
            subset["umap_y"],
            s=point_size - 4 if cluster_id == -1 else point_size,
            alpha=0.55 if cluster_id == -1 else 0.8,
            label=f"{label} (n={len(subset)})" if show_legend else None,
            c=[color],
            edgecolors="none",
        )

    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    if show_legend:
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


@dataclass
class PatchCorpus:
    patches: np.ndarray
    image_ids: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    thumbnail_paths: np.ndarray


def load_patch_corpus(patch_run_dir: Path) -> PatchCorpus:
    from .extract import load_patch_vector

    vectors_dir = patch_run_dir / "vectors"
    files = sorted(vectors_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No patch vectors in {vectors_dir}")

    patch_chunks: List[np.ndarray] = []
    image_ids: List[str] = []
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    thumb_paths: List[str] = []

    manifest_path = patch_run_dir / "manifest.json"
    thumb_lookup: Dict[str, str] = {}
    thumb_dir: Optional[Path] = THUMB_DIR_DEFAULT
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        thumb_lookup = manifest.get("thumbnail_paths", {})
        if manifest.get("thumb_dir"):
            thumb_dir = Path(manifest["thumb_dir"])

    for path in files:
        patches, r, c, image_id = load_patch_vector(path)
        if not image_id.endswith(".jpg") and path.name.endswith(".npz"):
            image_id = path.name[:-4]
        patch_chunks.append(patches)
        image_ids.extend([image_id] * len(patches))
        rows.append(r)
        cols.append(c)
        if image_id in thumb_lookup and thumb_lookup[image_id]:
            thumb_path = thumb_lookup[image_id]
        elif thumb_dir is not None:
            thumb_path = str(thumb_dir / image_id)
        else:
            thumb_path = ""
        thumb_paths.extend([thumb_path] * len(patches))

    return PatchCorpus(
        patches=np.vstack(patch_chunks),
        image_ids=np.array(image_ids),
        rows=np.concatenate(rows),
        cols=np.concatenate(cols),
        thumbnail_paths=np.array(thumb_paths),
    )


def build_patch_assignments(corpus: PatchCorpus, result: ClusterPipelineResult) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": corpus.image_ids,
            "patch_row": corpus.rows,
            "patch_col": corpus.cols,
            "cluster_id": result.labels,
            "cluster_probability": result.probabilities,
            "umap_x": result.umap_2d[:, 0],
            "umap_y": result.umap_2d[:, 1],
            "thumbnail_path": corpus.thumbnail_paths,
        }
    )


def build_image_cluster_histogram(assignments: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        assignments.groupby(["image_id", "cluster_id"])
        .size()
        .reset_index(name="patch_count")
    )
    totals = grouped.groupby("image_id")["patch_count"].transform("sum")
    grouped["patch_fraction"] = (grouped["patch_count"] / totals).round(4)
    return grouped.sort_values(["image_id", "patch_fraction"], ascending=[True, False])


def save_patch_cluster_montages(
    assignments: pd.DataFrame,
    out_dir: Path,
    *,
    thumb_dir: Path = THUMB_DIR_DEFAULT,
    samples_per_cluster: int = DEFAULT_SAMPLES_PER_CLUSTER,
    vit_patch_size: int = DEFAULT_VIT_PATCH_SIZE,
    cols: int = 6,
    upscale: int = 8,
) -> Dict[str, Any]:
    from .preprocess import preprocess_for_dinov3

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Any] = {}
    preprocessed_cache: Dict[str, Image.Image] = {}

    for cluster_id in sorted(assignments["cluster_id"].unique()):
        label = "noise" if cluster_id == -1 else str(int(cluster_id))
        cluster_dir = out_dir / f"cluster_{label}"
        cluster_dir.mkdir(parents=True, exist_ok=True)

        subset = assignments[assignments["cluster_id"] == cluster_id]
        ordered = subset.sort_values("cluster_probability", ascending=False).head(samples_per_cluster)

        panels: List[Image.Image] = []
        copied = 0
        for _, row in ordered.iterrows():
            image_id = row["image_id"]
            thumb = _resolve_thumb_path(image_id, str(row.get("thumbnail_path", "")), thumb_dir)
            if thumb is None:
                continue

            if image_id not in preprocessed_cache:
                pre = preprocess_for_dinov3(thumb, target_size=DEFAULT_PATCH_SIZE)
                preprocessed_cache[image_id] = pre.image

            image = preprocessed_cache[image_id]
            ps = vit_patch_size
            r = int(row["patch_row"])
            c = int(row["patch_col"])
            crop = image.crop((c * ps, r * ps, (c + 1) * ps, (r + 1) * ps))
            crop = crop.resize((ps * upscale, ps * upscale), Image.NEAREST)
            out_name = f"{image_id.rsplit('.', 1)[0]}_r{r}_c{c}.jpg"
            crop.save(cluster_dir / out_name, quality=90)
            panels.append(crop)
            copied += 1

        if panels:
            rows_n = (len(panels) + cols - 1) // cols
            tile = ps * upscale
            grid = Image.new("RGB", (cols * tile, rows_n * tile), (0, 0, 0))
            for i, panel in enumerate(panels):
                x = (i % cols) * tile
                y = (i // cols) * tile
                grid.paste(panel, (x, y))
            grid.save(cluster_dir / "_grid.jpg", quality=90)

        saved[label] = {"patch_count": int(len(subset)), "samples_saved": copied}

    return saved
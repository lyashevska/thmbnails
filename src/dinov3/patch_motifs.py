"""
Patch-level motif discovery: cluster local visual units across the corpus.
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

from .cluster import _require_cluster_deps, build_metadata_frame, run_cluster_pipeline
from .config import (
    CSV_DEFAULT,
    DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
    DEFAULT_HDBSCAN_MIN_SAMPLES,
    DEFAULT_PCA_COMPONENTS,
    DEFAULT_PATCH_SIZE,
    DEFAULT_VIT_PATCH_SIZE,
    DEFAULT_SAMPLES_PER_CLUSTER,
    DEFAULT_UMAP_MIN_DIST,
    DEFAULT_UMAP_NEIGHBORS,
    PATCH_EMBEDDINGS_ROOT,
    THUMB_DIR_DEFAULT,
)

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore


@dataclass
class PatchCorpus:
    patches: np.ndarray
    image_ids: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    thumbnail_paths: np.ndarray


def run_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_patch_vector(path: Path, result: "PatchExtractResult", image_id: str) -> None:
    from .extract import PatchExtractResult  # noqa: F401

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        patches=result.patches,
        rows=result.rows,
        cols=result.cols,
        grid_shape=np.array(result.grid_shape, dtype=np.int32),
        patch_size=np.int32(result.patch_size),
        image_id=np.array(image_id),
    )


def load_patch_vector(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    data = np.load(path)
    image_id = str(data["image_id"].item()) if data["image_id"].shape == () else str(data["image_id"][0])
    return data["patches"], data["rows"], data["cols"], image_id


def load_patch_corpus(patch_run_dir: Path) -> PatchCorpus:
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
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        thumb_lookup = manifest.get("thumbnail_paths", {})

    for path in files:
        patches, r, c, image_id = load_patch_vector(path)
        if image_id.endswith(".jpg"):
            pass
        elif path.name.endswith(".npz"):
            image_id = path.name[:-4]
        patch_chunks.append(patches)
        image_ids.extend([image_id] * len(patches))
        rows.append(r)
        cols.append(c)
        thumb_paths.extend([thumb_lookup.get(image_id, "")] * len(patches))

    return PatchCorpus(
        patches=np.vstack(patch_chunks),
        image_ids=np.array(image_ids),
        rows=np.concatenate(rows),
        cols=np.concatenate(cols),
        thumbnail_paths=np.array(thumb_paths),
    )


def run_patch_motif_pipeline(
    corpus: PatchCorpus,
    *,
    pca_components: int = DEFAULT_PCA_COMPONENTS,
    umap_neighbors: int = DEFAULT_UMAP_NEIGHBORS,
    umap_min_dist: float = DEFAULT_UMAP_MIN_DIST,
    hdbscan_min_cluster_size: int = DEFAULT_HDBSCAN_MIN_CLUSTER_SIZE,
    hdbscan_min_samples: int = DEFAULT_HDBSCAN_MIN_SAMPLES,
    seed: int = 42,
):
    return run_cluster_pipeline(
        corpus.patches,
        pca_components=pca_components,
        umap_neighbors=umap_neighbors,
        umap_min_dist=umap_min_dist,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_samples=hdbscan_min_samples,
        seed=seed,
    )


def build_patch_assignments(corpus: PatchCorpus, result) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": corpus.image_ids,
            "patch_row": corpus.rows,
            "patch_col": corpus.cols,
            "motif_id": result.labels,
            "motif_probability": result.probabilities,
            "umap_x": result.umap_2d[:, 0],
            "umap_y": result.umap_2d[:, 1],
            "thumbnail_path": corpus.thumbnail_paths,
        }
    )


def motif_summary(assignments: pd.DataFrame) -> pd.DataFrame:
    counts = assignments.groupby("motif_id").size().reset_index(name="count").sort_values("motif_id")
    total = len(assignments)
    counts["pct"] = (counts["count"] / total * 100).round(2)
    return counts


def build_image_motif_histogram(assignments: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        assignments.groupby(["image_id", "motif_id"])
        .size()
        .reset_index(name="patch_count")
    )
    totals = grouped.groupby("image_id")["patch_count"].transform("sum")
    grouped["patch_fraction"] = (grouped["patch_count"] / totals).round(4)
    return grouped.sort_values(["image_id", "patch_fraction"], ascending=[True, False])


def build_dominant_motif_per_image(assignments: pd.DataFrame) -> pd.DataFrame:
    hist = build_image_motif_histogram(assignments)
    idx = hist.groupby("image_id")["patch_count"].idxmax()
    dominant = hist.loc[idx].copy()
    dominant = dominant.rename(columns={"motif_id": "dominant_motif_id"})
    return dominant.reset_index(drop=True)


def save_patch_umap_plot(assignments: pd.DataFrame, out_path: Path) -> None:
    _require_cluster_deps()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    motifs = sorted(assignments["motif_id"].unique())
    cmap = plt.get_cmap("tab20", max(len(motifs), 1))

    for idx, motif_id in enumerate(motifs):
        subset = assignments[assignments["motif_id"] == motif_id]
        label = "noise" if motif_id == -1 else f"motif {motif_id}"
        color = "#aaaaaa" if motif_id == -1 else cmap(idx % 20)
        ax.scatter(
            subset["umap_x"],
            subset["umap_y"],
            s=4 if motif_id == -1 else 6,
            alpha=0.25 if motif_id == -1 else 0.55,
            label=f"{label} (n={len(subset)})",
            c=[color],
            edgecolors="none",
        )

    ax.set_title("Patch motif clusters (UMAP)")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize=8, markerscale=2)
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


def save_motif_patch_montages(
    assignments: pd.DataFrame,
    out_dir: Path,
    *,
    patch_run_dir: Path,
    thumb_dir: Path = THUMB_DIR_DEFAULT,
    samples_per_motif: int = DEFAULT_SAMPLES_PER_CLUSTER,
    vit_patch_size: int = DEFAULT_VIT_PATCH_SIZE,
    cols: int = 6,
    upscale: int = 8,
) -> Dict[str, Any]:
    from .preprocess import preprocess_for_dinov3

    out_dir.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, Any] = {}
    preprocessed_cache: Dict[str, Image.Image] = {}

    for motif_id in sorted(assignments["motif_id"].unique()):
        label = "noise" if motif_id == -1 else str(int(motif_id))
        motif_dir = out_dir / f"motif_{label}"
        motif_dir.mkdir(parents=True, exist_ok=True)

        subset = assignments[assignments["motif_id"] == motif_id]
        ordered = subset.sort_values("motif_probability", ascending=False).head(samples_per_motif)

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
            crop.save(motif_dir / out_name, quality=90)
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
            grid.save(motif_dir / "_grid.jpg", quality=90)

        saved[label] = {"patch_count": int(len(subset)), "samples_saved": copied}

    return saved


def join_metadata_to_dominant_motifs(
    dominant: pd.DataFrame,
    *,
    csv_path: Path = CSV_DEFAULT,
) -> pd.DataFrame:
    image_ids = dominant["image_id"].tolist()
    metadata = build_metadata_frame(image_ids, csv_path=csv_path)
    return dominant.merge(metadata, on="image_id", how="left")
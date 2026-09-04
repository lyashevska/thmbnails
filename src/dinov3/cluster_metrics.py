"""Unsupervised scores for CLS HDBSCAN: DBCV, silhouette, noise, ARI/NMI."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .config import SWEEP_DBCV_WEIGHT, SWEEP_NOISE_WEIGHT, SWEEP_SILHOUETTE_WEIGHT


def n_clusters(labels: np.ndarray) -> int:
    return int(len(set(labels.tolist()) - {-1}))


def n_noise(labels: np.ndarray) -> int:
    return int((labels == -1).sum())


def noise_fraction(labels: np.ndarray) -> float:
    if len(labels) == 0:
        return float("nan")
    return float(n_noise(labels) / len(labels))


def size_stats(labels: np.ndarray) -> dict[str, float]:
    assigned = labels[labels >= 0]
    if assigned.size == 0:
        return {
            "size_min": float("nan"),
            "size_median": float("nan"),
            "size_p90": float("nan"),
            "size_max": float("nan"),
            "frac_clusters_at_min_size": float("nan"),
        }
    counts = pd.Series(assigned).value_counts()
    min_size = int(counts.min())
    at_min = float((counts == min_size).mean()) if min_size else float("nan")
    return {
        "size_min": float(counts.min()),
        "size_median": float(counts.median()),
        "size_p90": float(counts.quantile(0.9)),
        "size_max": float(counts.max()),
        "frac_clusters_at_min_size": at_min,
    }


def dbcv_score(cluster_embeddings: np.ndarray, labels: np.ndarray) -> float | None:
    """Density-based cluster validity on the space used for HDBSCAN. None if undefined."""
    if n_clusters(labels) < 2:
        return None
    try:
        from hdbscan.validity import validity_index

        score = validity_index(
            np.asarray(cluster_embeddings, dtype=np.float64),
            np.asarray(labels, dtype=np.int32),
            metric="euclidean",
        )
    except Exception:
        return None
    if score is None or not np.isfinite(score):
        return None
    return float(score)


def silhouette_assigned(
    cluster_embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    metric: str = "euclidean",
) -> float | None:
    """Silhouette on non-noise points. None if undefined."""
    mask = labels >= 0
    if int(mask.sum()) < 3 or n_clusters(labels) < 2:
        return None
    try:
        from sklearn.metrics import silhouette_score

        score = silhouette_score(
            cluster_embeddings[mask],
            labels[mask],
            metric=metric,
        )
    except Exception:
        return None
    if score is None or not np.isfinite(score):
        return None
    return float(score)


def quality_row(
    cluster_embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    min_cluster_size: int | None = None,
) -> dict[str, Any]:
    sizes = size_stats(labels)
    counts = None
    frac_at_mcs = sizes["frac_clusters_at_min_size"]
    if min_cluster_size is not None:
        assigned = labels[labels >= 0]
        if assigned.size:
            counts = pd.Series(assigned).value_counts()
            frac_at_mcs = float((counts == min_cluster_size).mean())
        else:
            frac_at_mcs = float("nan")
    return {
        "n_images": int(len(labels)),
        "n_clusters": n_clusters(labels),
        "n_noise": n_noise(labels),
        "noise_fraction": noise_fraction(labels),
        "dbcv": dbcv_score(cluster_embeddings, labels),
        "silhouette": silhouette_assigned(cluster_embeddings, labels),
        "size_min": sizes["size_min"],
        "size_median": sizes["size_median"],
        "size_p90": sizes["size_p90"],
        "size_max": sizes["size_max"],
        "frac_clusters_at_min_cluster_size": frac_at_mcs,
    }


def _minmax(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index, dtype=float)
    lo, hi = float(valid.min()), float(valid.max())
    if hi == lo:
        out = pd.Series(np.nan, index=series.index, dtype=float)
        out.loc[valid.index] = 1.0
        return out
    return (series - lo) / (hi - lo)


def add_composite(
    frame: pd.DataFrame,
    *,
    dbcv_weight: float = SWEEP_DBCV_WEIGHT,
    silhouette_weight: float = SWEEP_SILHOUETTE_WEIGHT,
    noise_weight: float = SWEEP_NOISE_WEIGHT,
) -> pd.DataFrame:
    """Min-max normalise DBCV, silhouette, and (1 - noise) over the grid; weighted sum."""
    out = frame.copy()
    dbcv_n = _minmax(out["dbcv"].astype(float))
    sil_n = _minmax(out["silhouette"].astype(float))
    coverage_n = _minmax((1.0 - out["noise_fraction"]).astype(float))
    out["dbcv_norm"] = dbcv_n
    out["silhouette_norm"] = sil_n
    out["coverage_norm"] = coverage_n
    out["composite"] = (
        dbcv_weight * dbcv_n.fillna(0.0)
        + silhouette_weight * sil_n.fillna(0.0)
        + noise_weight * coverage_n.fillna(0.0)
    )
    return out.sort_values("composite", ascending=False).reset_index(drop=True)


def pairwise_stability(label_runs: Sequence[np.ndarray]) -> dict[str, Any]:
    """Mean±SD ARI and NMI over unique pairs of partitions.

    Primary scores treat noise (-1) as a label. ``*_assigned`` scores keep only
    points that were clustered (not noise) in both runs.
    """
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    if len(label_runs) < 2:
        raise ValueError("Need at least two label vectors for stability")

    aris: list[float] = []
    nmis: list[float] = []
    aris_assigned: list[float] = []
    nmis_assigned: list[float] = []

    for i in range(len(label_runs)):
        for j in range(i + 1, len(label_runs)):
            a = np.asarray(label_runs[i])
            b = np.asarray(label_runs[j])
            aris.append(float(adjusted_rand_score(a, b)))
            nmis.append(float(normalized_mutual_info_score(a, b)))
            both = (a >= 0) & (b >= 0)
            if int(both.sum()) < 2:
                continue
            if len(set(a[both].tolist())) < 2 or len(set(b[both].tolist())) < 2:
                continue
            aris_assigned.append(float(adjusted_rand_score(a[both], b[both])))
            nmis_assigned.append(float(normalized_mutual_info_score(a[both], b[both])))

    def _mean_sd(values: Iterable[float]) -> tuple[float | None, float | None]:
        arr = np.array(list(values), dtype=np.float64)
        if arr.size == 0:
            return None, None
        return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)

    ari_mean, ari_sd = _mean_sd(aris)
    nmi_mean, nmi_sd = _mean_sd(nmis)
    ari_a_mean, ari_a_sd = _mean_sd(aris_assigned)
    nmi_a_mean, nmi_a_sd = _mean_sd(nmis_assigned)
    return {
        "n_runs": len(label_runs),
        "n_pairs": len(aris),
        "n_pairs_assigned": len(aris_assigned),
        "ari_mean": ari_mean,
        "ari_sd": ari_sd,
        "nmi_mean": nmi_mean,
        "nmi_sd": nmi_sd,
        "ari_assigned_mean": ari_a_mean,
        "ari_assigned_sd": ari_a_sd,
        "nmi_assigned_mean": nmi_a_mean,
        "nmi_assigned_sd": nmi_a_sd,
    }

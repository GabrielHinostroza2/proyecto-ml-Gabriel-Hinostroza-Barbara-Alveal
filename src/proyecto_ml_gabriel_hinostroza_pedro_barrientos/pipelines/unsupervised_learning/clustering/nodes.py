from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def run_kmeans(
    data: pd.DataFrame,
    k: int,
    random_state: int = 42,
) -> Tuple[pd.Series, Dict]:
    """Ejecuta K-Means y retorna etiquetas + métricas."""
    model = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = model.fit_predict(data)

    metrics = _compute_cluster_metrics(
        data=data,
        labels=labels,
        inertia=model.inertia_,
        algo=f"kmeans_{k}",
    )
    return pd.Series(labels, name=f"kmeans_{k}_labels"), metrics


def run_dbscan(
    data: pd.DataFrame,
    eps: float,
    min_samples: int,
) -> Tuple[pd.Series, Dict]:
    """Ejecuta DBSCAN y retorna etiquetas + métricas."""
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data)

    metrics = _compute_cluster_metrics(
        data=data,
        labels=labels,
        inertia=None,
        algo=f"dbscan_eps{eps}",
    )
    return pd.Series(labels, name="dbscan_labels"), metrics


def run_gmm(
    data: pd.DataFrame,
    n_components: int,
    random_state: int = 42,
) -> Tuple[pd.Series, Dict]:
    """Ejecuta Gaussian Mixture Models y retorna etiquetas + métricas."""
    model = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = model.fit_predict(data)

    metrics = _compute_cluster_metrics(
        data=data,
        labels=labels,
        inertia=None,
        algo=f"gmm_{n_components}",
    )
    return pd.Series(labels, name=f"gmm_{n_components}_labels"), metrics


def _compute_cluster_metrics(
    data: pd.DataFrame,
    labels: np.ndarray,
    inertia: float | None,
    algo: str,
) -> Dict:
    """Calcula métricas clásicas de clustering."""
    metrics: Dict[str, float | int | str] = {"algorithm": algo}

    # Para DBSCAN, -1 = ruido → lo excluimos de métricas
    valid_mask = labels != -1

    if valid_mask.sum() > 1 and len(np.unique(labels[valid_mask])) > 1:
        metrics["silhouette"] = float(
            silhouette_score(data[valid_mask], labels[valid_mask])
        )
        metrics["davies_bouldin"] = float(
            davies_bouldin_score(data[valid_mask], labels[valid_mask])
        )
        metrics["calinski_harabasz"] = float(
            calinski_harabasz_score(data[valid_mask], labels[valid_mask])
        )
    else:
        metrics["silhouette"] = np.nan
        metrics["davies_bouldin"] = np.nan
        metrics["calinski_harabasz"] = np.nan

    if inertia is not None:
        metrics["inertia"] = float(inertia)

    metrics["n_clusters"] = int(len(np.unique(labels)))
    metrics["n_noise"] = int((labels == -1).sum())
    return metrics


def aggregate_clustering_results(
    kmeans_labels: pd.Series,
    dbscan_labels: pd.Series,
    gmm_labels: pd.Series,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Combina features originales con etiquetas de todos los algoritmos."""
    df = features.copy()
    df[kmeans_labels.name] = kmeans_labels.values
    df[dbscan_labels.name] = dbscan_labels.values
    df[dbscan_labels.name] = dbscan_labels.values
    df[gmm_labels.name] = gmm_labels.values
    return df


def aggregate_metrics(*metrics_list: Dict) -> pd.DataFrame:
    """Combina dicts de métricas en un solo DataFrame."""
    return pd.DataFrame(list(metrics_list))

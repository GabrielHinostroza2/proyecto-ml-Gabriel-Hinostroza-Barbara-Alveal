from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    run_kmeans,
    run_dbscan,
    run_gmm,
    aggregate_clustering_results,
    aggregate_metrics,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Pipeline de clustering con KMeans, DBSCAN y GMM."""
    return pipeline(
        [
            node(
                func=run_kmeans,
                inputs=dict(
                    data="clustering_features",
                    k="params:clustering.kmeans_k",
                    random_state="params:random_state",
                ),
                outputs=["kmeans_labels", "kmeans_metrics"],
                name="run_kmeans_node",
            ),
            node(
                func=run_dbscan,
                inputs=dict(
                    data="clustering_features",
                    eps="params:clustering.dbscan_eps",
                    min_samples="params:clustering.dbscan_min_samples",
                ),
                outputs=["dbscan_labels", "dbscan_metrics"],
                name="run_dbscan_node",
            ),
            node(
                func=run_gmm,
                inputs=dict(
                    data="clustering_features",
                    n_components="params:clustering.gmm_components",
                    random_state="params:random_state",
                ),
                outputs=["gmm_labels", "gmm_metrics"],
                name="run_gmm_node",
            ),
            node(
                func=aggregate_clustering_results,
                inputs=[
                    "kmeans_labels",
                    "dbscan_labels",
                    "gmm_labels",
                    "clustering_features",
                ],
                outputs="clustering_results",
                name="aggregate_clustering_results_node",
            ),
            node(
                func=aggregate_metrics,
                inputs=["kmeans_metrics", "dbscan_metrics", "gmm_metrics"],
                outputs="clustering_metrics",
                name="aggregate_metrics_node",
            ),
        ]
    )

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import run_pca, run_tsne


def create_pipeline(**kwargs) -> Pipeline:
    """Pipeline de reducción de dimensionalidad (PCA + t-SNE)."""
    return pipeline(
        [
            node(
                func=run_pca,
                inputs=dict(
                    data="clustering_features",
                    n_components="params:dimensionality_reduction.pca_n_components",
                ),
                outputs=["pca_projection", "pca_explained_variance"],
                name="run_pca_node",
            ),
            node(
                func=run_tsne,
                inputs=dict(
                    data="clustering_features",
                    n_components="params:dimensionality_reduction.tsne_n_components",
                    perplexity="params:dimensionality_reduction.tsne_perplexity",
                    random_state="params:random_state",
                ),
                outputs="tsne_projection",
                name="run_tsne_node",
            ),
        ]
    )

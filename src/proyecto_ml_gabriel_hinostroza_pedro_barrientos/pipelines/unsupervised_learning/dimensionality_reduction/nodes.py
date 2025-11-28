import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def run_pca(data: pd.DataFrame, n_components: int = 2):
    """Ejecuta PCA y retorna proyección + varianza explicada."""
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data)

    projection = pd.DataFrame(
        transformed, columns=[f"PC{i+1}" for i in range(n_components)]
    )

    explained = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(n_components)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )

    return projection, explained


def run_tsne(
    data: pd.DataFrame,
    n_components: int,
    perplexity: float,
    random_state: int = 42,
) -> pd.DataFrame:
    """Ejecuta t-SNE y retorna embedding 2D/3D."""
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        init="pca",
    )
    emb = tsne.fit_transform(data)
    cols = [f"TSNE{i+1}" for i in range(n_components)]
    return pd.DataFrame(emb, columns=cols)

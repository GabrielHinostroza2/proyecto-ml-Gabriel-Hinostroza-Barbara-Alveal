from kedro.pipeline import Pipeline, pipeline

from .clustering import pipeline as clustering_pipeline
from .dimensionality_reduction import pipeline as dimred_pipeline
# Si después haces anomaly_detection, aquí también lo importas.


def create_pipeline(**kwargs) -> Pipeline:
    """Pipeline maestro de aprendizaje no supervisado."""
    return pipeline(
        [
            clustering_pipeline.create_pipeline(),
            dimred_pipeline.create_pipeline(),
        ]
    )
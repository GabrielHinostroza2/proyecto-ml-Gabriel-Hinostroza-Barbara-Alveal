from typing import Dict

from kedro.pipeline import Pipeline, pipeline

from proyecto_ml_gabriel_hinostroza_pedro_barrientos.pipelines import (
    data_processing,
    data_science,
    classification,
    regression,
    reporting,
    unsupervised_learning,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    data_processing_pipeline = data_processing.create_pipeline()
    data_science_pipeline = data_science.create_pipeline()
    classification_pipeline = classification.create_pipeline()
    regression_pipeline = regression.create_pipeline()
    unsupervised_pipeline = unsupervised_learning.create_pipeline()
    reporting_pipeline = reporting.create_pipeline()

    return {
        # pipeline que corre todo
        "__default__": (
            data_processing_pipeline
            + data_science_pipeline
            + classification_pipeline
            + regression_pipeline
            + unsupervised_pipeline
            + reporting_pipeline
        ),
        # pipelines individuales
        "data_processing": data_processing_pipeline,
        "data_science": data_science_pipeline,
        "classification": classification_pipeline,
        "regression": regression_pipeline,
        "unsupervised_learning": unsupervised_pipeline,
        "reporting": reporting_pipeline,
    }

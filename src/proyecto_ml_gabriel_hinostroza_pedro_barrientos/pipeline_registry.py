from kedro.pipeline import Pipeline
from proyecto_ml_gabriel_hinostroza_pedro_barrientos.pipelines import data_processing
from proyecto_ml_gabriel_hinostroza_pedro_barrientos.pipelines import regression
from proyecto_ml_gabriel_hinostroza_pedro_barrientos.pipelines import classification

def register_pipelines() -> dict[str, Pipeline]:
    dp = data_processing.create_pipeline()
    reg = regression.create_pipeline()
    clf = classification.create_pipeline()
    full = dp + reg + clf
    return {
        "__default__": full,
        "data_processing": dp,
        "regression": reg,
        "classification": clf,
        "full": full,
    }

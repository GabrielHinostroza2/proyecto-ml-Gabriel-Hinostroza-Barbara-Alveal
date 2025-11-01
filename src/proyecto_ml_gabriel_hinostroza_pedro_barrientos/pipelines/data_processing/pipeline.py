
from kedro.pipeline import Pipeline, node
from .nodes import (
    clean_listings, build_calendar_agg, engineer_features,
    build_primary_datasets, make_feature_matrices, make_splits
)

def create_pipeline(**kwargs):
    return Pipeline([
        node(clean_listings, inputs="listings", outputs="listings_clean", name="clean_listings"),
        node(build_calendar_agg, inputs="calendar", outputs="calendar_agg", name="calendar_agg"),
        node(engineer_features, inputs=["listings_clean","calendar_agg","reviews"], outputs="data_enriched", name="engineer_features"),
        node(build_primary_datasets, inputs="data_enriched", outputs=["listings_for_regression","listings_for_classification"], name="build_primary_datasets"),
        node(make_feature_matrices, inputs=["listings_for_regression","listings_for_classification"], outputs=["X_reg","y_reg","X_clf","y_clf"], name="make_feature_matrices"),
        node(make_splits,
             inputs=dict(X_reg="X_reg", y_reg="y_reg", X_clf="X_clf", y_clf="y_clf",
                         test_size="params:test_size", random_state="params:random_state"),
             outputs=["X_train_reg","X_test_reg","y_train_reg","y_test_reg","X_train_clf","X_test_clf","y_train_clf","y_test_clf"],
             name="make_splits"),
    ])

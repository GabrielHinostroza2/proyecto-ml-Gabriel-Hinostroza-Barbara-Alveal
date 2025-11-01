from kedro.pipeline import Pipeline, node
from .nodes import train_and_tune_regression, evaluate_regression, summarize_cv

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            train_and_tune_regression,
            inputs=dict(
                X_train="X_train_reg",
                y_train="y_train_reg",
                cv_folds="params:cv_folds",
                scoring="params:reg_scoring",
            ),
            outputs=["best_regressor","reg_cv_results"],
            name="reg_train_tune",
        ),
        node(
            evaluate_regression,
            inputs=dict(
                model="best_regressor",
                X_test="X_test_reg",
                y_test="y_test_reg",
            ),
            outputs="reg_metrics",
            name="reg_evaluate",
        ),
        node(
            summarize_cv,
            inputs="reg_cv_results",
            outputs="reg_cv_summary",
            name="reg_summarize_cv",
        ),
    ])

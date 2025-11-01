from kedro.pipeline import Pipeline, node
from .nodes import train_and_tune_classification, evaluate_classification, summarize_cv

def create_pipeline(**kwargs):
    return Pipeline([
        node(
            train_and_tune_classification,
            inputs=dict(
                X_train="X_train_clf",
                y_train="y_train_clf",
                cv_folds="params:cv_folds",
                scoring="params:clf_scoring",
            ),
            outputs=["best_classifier","clf_cv_results"],
            name="clf_train_tune",
        ),
        node(
            evaluate_classification,
            inputs=dict(
                model="best_classifier",
                X_test="X_test_clf",
                y_test="y_test_clf",
            ),
            outputs=["clf_metrics","clf_confusion_matrix"],
            name="clf_evaluate",
        ),
        node(
            summarize_cv,
            inputs="clf_cv_results",
            outputs="clf_cv_summary",
            name="clf_summarize_cv",
        ),
    ])

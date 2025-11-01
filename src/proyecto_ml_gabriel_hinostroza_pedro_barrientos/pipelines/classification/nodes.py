import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def _build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return ColumnTransformer(
        [("num", Pipeline([("scaler", StandardScaler())]), num_cols),
         ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="drop",
    )

def _get_clf_search_space():
    return {
        "logreg": (LogisticRegression(max_iter=5000), {"model__C":[0.5,1,5]}),
        "knn": (KNeighborsClassifier(), {"model__n_neighbors":[5,11,21]}),
        "svc": (SVC(), {"model__C":[0.5,1,10], "model__kernel":["rbf","linear"], "model__probability":[True]}),
        "rf": (RandomForestClassifier(random_state=42), {"model__n_estimators":[200,400], "model__max_depth":[None,10,20]}),
        "gboost": (GradientBoostingClassifier(random_state=42), {"model__n_estimators":[200,400], "model__learning_rate":[0.05,0.1]}),
    }

def train_and_tune_classification(X_train, y_train, cv_folds:int, scoring:str):
    pre = _build_preprocessor(X_train)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = []
    best_score = -np.inf
    best_estimator = None

    for name, (est, grid) in _get_clf_search_space().items():
        pipe = Pipeline([("pre", pre), ("model", est)])
        gs = GridSearchCV(pipe, grid, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
        gs.fit(X_train, y_train)
        df = pd.DataFrame(gs.cv_results_)
        df["model_name"] = name
        results.append(df)
        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best_estimator = gs.best_estimator_

    clf_cv_results = pd.concat(results, ignore_index=True)
    return best_estimator, clf_cv_results

def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "roc_auc": (float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None),
    }

    cm = confusion_matrix(y_test, y_pred)
    cm_list = cm.tolist()          # ← CLAVE: convertir a lista para JSON

    return metrics, cm_list

def summarize_cv(cv_results: pd.DataFrame) -> pd.DataFrame:
    keep = ["model_name","params","mean_test_score","std_test_score","rank_test_score"]
    return cv_results[keep].sort_values("rank_test_score").reset_index(drop=True)

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def _build_preprocessor(X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scaler", StandardScaler())]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

def _get_reg_search_space():
    return {
        "ridge": (Ridge(), {"model__alpha":[0.1,1,10]}),
        "lasso": (Lasso(max_iter=20000), {"model__alpha":[0.001,0.01,0.1]}),
        "linear": (LinearRegression(), {"model__fit_intercept":[True,False]}),
        "rf": (RandomForestRegressor(random_state=42), {
            "model__n_estimators":[200,400],
            "model__max_depth":[None,10,20]
        }),
        "gbr": (GradientBoostingRegressor(random_state=42), {
            "model__n_estimators":[200,400],
            "model__learning_rate":[0.05,0.1]
        }),
        "svr": (SVR(), {"model__C":[0.5,1,10], "model__epsilon":[0.1,0.2]}),
    }

def train_and_tune_regression(X_train, y_train, cv_folds:int, scoring:str):
    pre = _build_preprocessor(X_train)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    results = []
    best_score = -np.inf
    best_estimator = None

    for name, (est, grid) in _get_reg_search_space().items():
        pipe = Pipeline([("pre", pre), ("model", est)])
        gs = GridSearchCV(pipe, grid, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
        gs.fit(X_train, y_train)
        df = pd.DataFrame(gs.cv_results_)
        df["model_name"] = name
        results.append(df)
        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best_estimator = gs.best_estimator_

    reg_cv_results = pd.concat(results, ignore_index=True)
    return best_estimator, reg_cv_results

def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}

def summarize_cv(cv_results: pd.DataFrame) -> pd.DataFrame:
    keep = ["model_name","params","mean_test_score","std_test_score","rank_test_score"]
    return cv_results[keep].sort_values("rank_test_score").reset_index(drop=True)

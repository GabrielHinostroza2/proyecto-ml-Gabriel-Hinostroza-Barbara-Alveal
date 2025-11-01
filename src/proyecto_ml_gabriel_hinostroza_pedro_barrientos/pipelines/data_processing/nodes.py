
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def _clean_price(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.str.replace(r"[\$,]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def _parse_bool_tf(s: pd.Series) -> pd.Series:
    return s.map({"t": True, "f": False}).astype("boolean")

def _add_date_parts(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        d = pd.to_datetime(df[col], errors="coerce")
        df[f"{col}_year"] = d.dt.year
        df[f"{col}_month"] = d.dt.month
        df[f"{col}_dow"] = d.dt.dayofweek
    return df

def clean_listings(listings: pd.DataFrame) -> pd.DataFrame:
    df = listings.copy()
    if "price" in df.columns:
        df["price_float"] = _clean_price(df["price"])
    for c in ["weekly_price","monthly_price","security_deposit","cleaning_fee","extra_people"]:
        if c in df.columns:
            df[c] = _clean_price(df[c])
    for c in ["instant_bookable","host_is_superhost"]:
        if c in df.columns and df[c].dtype == object:
            df[c] = _parse_bool_tf(df[c])
    for c in ["last_scraped","host_since","first_review","last_review"]:
        if c in df.columns:
            df = _add_date_parts(df, c)
    return df

def build_calendar_agg(calendar: pd.DataFrame) -> pd.DataFrame:
    cal = calendar.copy()

    if "available" in cal.columns:
        cal["booked"] = (cal["available"] == "f").astype(int)
    elif "booked" not in cal.columns:
        cal["booked"] = np.nan

    if "price" in cal.columns:
        cal["price_float"] = _clean_price(cal["price"])
    elif "price_float" not in cal.columns:
        cal["price_float"] = np.nan

    key = "listing_id" if "listing_id" in cal.columns else ("id" if "id" in cal.columns else None)
    if key is None or cal.empty:
        return pd.DataFrame(columns=["listing_id","booking_rate","mean_daily_price","days"])

    agg = cal.groupby(key).agg(
        booking_rate=("booked","mean"),
        mean_daily_price=("price_float","mean"),
        days=("booked","count"),
    ).reset_index()

    if key != "listing_id":
        agg = agg.rename(columns={key:"listing_id"})
    return agg


def engineer_features(listings_clean: pd.DataFrame, calendar_agg: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    df = listings_clean.copy()
    left_key = "id" if "id" in df.columns else ("listing_id" if "listing_id" in df.columns else None)
    base = df.copy()
    if left_key and calendar_agg is not None and len(calendar_agg) > 0:
        base = base.merge(calendar_agg, how="left", left_on=left_key, right_on="listing_id")
    if reviews is not None and "listing_id" in reviews.columns:
        tmp = reviews.copy()
        if "date" in tmp.columns:
            tmp["year_review"] = pd.to_datetime(tmp["date"], errors="coerce").dt.year
        grp = tmp.groupby("listing_id").agg(
            reviews_count=("id","count") if "id" in tmp.columns else ("listing_id","count"),
            last_review_year=("year_review","max"),
        ).reset_index()
        base = base.merge(grp, how="left",
                          left_on=("id" if "id" in base.columns else "listing_id"),
                          right_on="listing_id", suffixes=("", "_rv"))
        if "listing_id_rv" in base.columns:
            base = base.drop(columns=["listing_id_rv"])
    return base

def build_primary_datasets(full_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = full_df.copy()
    num_cols = [c for c in [
        "accommodates","bathrooms","bedrooms","beds","number_of_reviews","review_scores_rating",
        "weekly_price","monthly_price","security_deposit","cleaning_fee","extra_people",
        "booking_rate","mean_daily_price","reviews_count"
    ] if c in df.columns]
    cat_cols = [c for c in ["room_type","neighbourhood_cleansed","property_type"] if c in df.columns]
    reg_cols = ["price_float"] + num_cols + cat_cols
    reg_df = df[reg_cols].dropna(subset=["price_float"]).copy()

    target = None
    for cand in ["instant_bookable","host_is_superhost"]:
        if cand in df.columns:
            target = cand; break
    clf_df = pd.DataFrame()
    if target:
        tmp = df.copy()
        if tmp[target].dtype == object:
            tmp[target] = _parse_bool_tf(tmp[target])
        tmp[target] = tmp[target].astype("Int64").astype(int)
        cols = [target] + [c for c in (["price_float"] + num_cols + cat_cols) if c in tmp.columns]
        clf_df = tmp[cols].dropna(subset=[target]).copy()
    return reg_df, clf_df

def make_feature_matrices(reg_df: pd.DataFrame, clf_df: pd.DataFrame):
    X_reg = pd.DataFrame(); y_reg = pd.Series(dtype=float)
    if not reg_df.empty:
        y_reg = reg_df["price_float"].copy()
        Xr = reg_df.drop(columns=["price_float"]).copy()
        num_cols = Xr.select_dtypes(include=[np.number, "boolean"]).columns.tolist()
        cat_cols = [c for c in Xr.columns if c not in num_cols]
        for c in Xr.columns:
            if str(Xr[c].dtype) == "boolean":
                Xr[c] = Xr[c].astype(float)
        if num_cols:
            imp = SimpleImputer(strategy="median"); Xr[num_cols] = imp.fit_transform(Xr[num_cols])
        if cat_cols:
            imp = SimpleImputer(strategy="most_frequent"); Xr[cat_cols] = imp.fit_transform(Xr[cat_cols])
        X_reg = pd.get_dummies(Xr, columns=cat_cols, drop_first=False)

    X_clf = pd.DataFrame(); y_clf = pd.Series(dtype=int)
    if clf_df is not None and not clf_df.empty:
        target = [c for c in ["instant_bookable","host_is_superhost"] if c in clf_df.columns][0]
        y_clf = clf_df[target].astype(int).copy()
        Xc = clf_df.drop(columns=[target]).copy()
        num_cols = Xc.select_dtypes(include=[np.number, "boolean"]).columns.tolist()
        cat_cols = [c for c in Xc.columns if c not in num_cols]
        for c in Xc.columns:
            if str(Xc[c].dtype) == "boolean":
                Xc[c] = Xc[c].astype(float)
        if num_cols:
            imp = SimpleImputer(strategy="median"); Xc[num_cols] = imp.fit_transform(Xc[num_cols])
        if cat_cols:
            imp = SimpleImputer(strategy="most_frequent"); Xc[cat_cols] = imp.fit_transform(Xc[cat_cols])
        X_clf = pd.get_dummies(Xc, columns=cat_cols, drop_first=False)

    return X_reg, y_reg, X_clf, y_clf

def make_splits(X_reg, y_reg, X_clf, y_clf, test_size: float = 0.2, random_state: int = 42):
    from sklearn.model_selection import train_test_split
    if X_reg is not None and len(X_reg) > 0:
        Xtr_r, Xte_r, ytr_r, yte_r = train_test_split(X_reg, y_reg, test_size=test_size, random_state=random_state)
    else:
        Xtr_r = Xte_r = pd.DataFrame(); ytr_r = yte_r = pd.Series(dtype=float)
    if X_clf is not None and len(X_clf) > 0:
        try:
            Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X_clf, y_clf, test_size=test_size, random_state=random_state, stratify=y_clf)
        except Exception:
            Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X_clf, y_clf, test_size=test_size, random_state=random_state)
    else:
        Xtr_c = Xte_c = pd.DataFrame(); ytr_c = yte_c = pd.Series(dtype=int)
    return Xtr_r, Xte_r, ytr_r, yte_r, Xtr_c, Xte_c, ytr_c, yte_c

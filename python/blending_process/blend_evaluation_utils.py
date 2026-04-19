# ==============================================================================
# File: blend_evaluation_utils.py
# ==============================================================================
# Purpose
#   Utility functions for the 2025 weekly-bin blending evaluation pipeline.
#   Provides spec-driven formula building, raw/calibrated prediction helpers,
#   Platt scaling, blend weight optimization, metric computation, and mapping
#   utilities.
# ==============================================================================

import re
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from ..pipelines._shared.misc import coalesce

# ---------------------------------------------------------------------------
# Column name helpers
# ---------------------------------------------------------------------------

def get_forecast_variant_suffix(spec, variant):
    """Map variant name to column suffix string."""
    m = (spec.get("extras") or {}).get("forecast_variants") or {
        "base": "",
        "mok": "_mok",
        "clim_mok_date": "_clim_mok_date",
    }
    if variant not in m:
        raise ValueError(f"Unknown forecast variant: {variant}")
    return m[variant]


def forecast_prob_cols(forecast_name, variant_suffix):
    """Build dict of week1..week4 probability column names."""
    base = f"{forecast_name}_p_onset{variant_suffix}"
    return {
        "week1": f"{base}_week1",
        "week2": f"{base}_week2",
        "week3": f"{base}_week3",
        "week4": f"{base}_week4",
    }


def forecast_label(forecast_name, variant):
    return forecast_name if variant == "base" else f"{forecast_name}_{variant}"


def make_year_tag(years):
    years = sorted(set(years))
    if not years:
        return ""
    if len(years) == 1:
        return f"_{years[0]}"
    return f"_{min(years)}_{max(years)}"


def make_cutoff_tag(cutoff_mode):
    if cutoff_mode == "clim_mok_date":
        return "_clim_mok_date"
    if cutoff_mode == "no_mok_filter":
        return "_no_mok_filter"
    return ""


def input_rds_from_cutoff(cutoff_mode, resolution=""):
    prefix = f"{resolution}_" if resolution else ""
    if cutoff_mode == "clim_mok_date":
        return f"cv_data_{prefix}clim_mok_date_new_pipeline.pkl"
    if cutoff_mode == "no_mok_filter":
        return f"cv_data_{prefix}no_mok_filter_new_pipeline.pkl"
    return f"cv_data_{prefix}new_pipeline.pkl"


# ---------------------------------------------------------------------------
# Formula expansion
# ---------------------------------------------------------------------------

def expand_formula_str(formula_str):
    """
    Expand formula shortcuts in a formula string.
    Terms containing '_qx' expand to '_week1'..'_week4'.
    Returns the expanded formula string.
    """
    if "~" in formula_str:
        lhs, rhs = formula_str.split("~", 1)
        lhs, rhs = lhs.strip(), rhs.strip()
    else:
        lhs, rhs = "", formula_str.strip()

    terms = [t.strip() for t in rhs.split("+")]
    expanded = []
    for term in terms:
        if "_qx" in term:
            for i in range(1, 5):
                expanded.append(term.replace("_qx", f"_week{i}"))
        else:
            expanded.append(term)

    new_rhs = " + ".join(expanded)
    return f"{lhs} ~ {new_rhs}" if lhs else new_rhs


def make_window_suffix(start_year, end_year):
    return f"_{start_year}_{end_year}"


def build_formulas_from_spec(spec, cutoff_mode):
    """
    Build dict of formula strings from spec["models"]["formulas"],
    with optional windowed variants.
    """
    formula_cfg = (spec.get("models") or {}).get("formulas") or {}
    if not formula_cfg:
        raise ValueError("Spec must define models.formulas with named entries containing 'text'.")

    base_texts = {}
    for nm, v in formula_cfg.items():
        if v.get("enabled", True):
            txt = v.get("text")
            if not txt:
                raise ValueError(f"Formula '{nm}' is enabled but has no non-empty 'text' in spec.")
            base_texts[nm] = expand_formula_str(txt)

    formulas = dict(base_texts)

    wcfg_all = (spec.get("models") or {}).get("window_variants")
    if wcfg_all:
        wcfg_list = wcfg_all if isinstance(wcfg_all, list) else [wcfg_all]
        for wcfg in wcfg_list:
            if not wcfg.get("enabled", False):
                continue
            only_if = wcfg.get("only_if_cutoff_mode")
            if only_if and cutoff_mode != only_if:
                continue
            base_name = wcfg.get("base_name")
            if not base_name or base_name not in base_texts:
                raise ValueError("window_variants.base_name must match a name in models.formulas.")
            start_years = [int(y) for y in (wcfg.get("start_years") or [])]
            end_year = int(wcfg.get("end_year", 0))
            if not start_years or not end_year:
                raise ValueError("window_variants must define start_years and end_year.")
            from_pat = wcfg.get("replace", {}).get("from")
            to_pat = wcfg.get("replace", {}).get("to")
            if not from_pat or not to_pat:
                raise ValueError("window_variants.replace must define 'from' and 'to'.")

            base_txt = base_texts[base_name]
            for sy in start_years:
                if sy == 1900:
                    continue
                window = make_window_suffix(sy, end_year)
                to_replaced = to_pat.replace("{window}", window)
                windowed = base_txt.replace(from_pat, to_replaced)
                formulas[f"{base_name}{window}"] = expand_formula_str(windowed)

    return formulas


# ---------------------------------------------------------------------------
# Raw and calibrated predictions
# ---------------------------------------------------------------------------

def make_raw_preds_from_wide(wide_df, forecast_name, variant, holdout_years, spec):
    """Extract raw model predictions for holdout years."""
    suf = get_forecast_variant_suffix(spec, variant)
    cols = forecast_prob_cols(forecast_name, suf)
    missing = [c for c in cols.values() if c not in wide_df.columns]
    if missing:
        warnings.warn(f"Skipping raw for {forecast_name} ({variant}); missing: {', '.join(missing)}")
        return None

    sub = wide_df[wide_df["year"].isin(holdout_years)].copy()
    sub["cv_week1"] = sub[cols["week1"]]
    sub["cv_week2"] = sub[cols["week2"]]
    sub["cv_week3"] = sub[cols["week3"]]
    sub["cv_week4"] = sub[cols["week4"]]
    w_sum = sub[["cv_week1", "cv_week2", "cv_week3", "cv_week4"]].sum(axis=1)
    sub["cv_later"] = np.clip(1.0 - w_sum, 0.0, 1.0)
    return sub[["outcome", "time", "id", "lat", "lon", "year",
                "cv_week1", "cv_week2", "cv_week3", "cv_week4", "cv_later"]]


def make_raw_preds_from_wide_logit_window(wide_df, base_col_prefix, holdout_years,
                                           start_year=None, end_year=None):
    """Extract raw clim-logit predictions (with optional window suffix)."""
    if start_year is not None and end_year is not None and start_year != 1900:
        window = f"_{start_year}_{end_year}"
    else:
        window = ""
    prefix = f"{base_col_prefix}{window}"
    wk_cols = [f"{prefix}_week{i}" for i in range(1, 5)]
    missing = [c for c in wk_cols if c not in wide_df.columns]
    if missing:
        warnings.warn(f"Skipping clim logit raw for prefix={prefix}; missing: {', '.join(missing)}")
        return None

    sub = wide_df[wide_df["year"].isin(holdout_years)].copy()
    sub["p1"] = expit(sub[wk_cols[0]])
    sub["p2"] = expit(sub[wk_cols[1]])
    sub["p3"] = expit(sub[wk_cols[2]])
    sub["p4"] = expit(sub[wk_cols[3]])
    sub["cv_later"] = np.clip(1.0 - (sub["p1"] + sub["p2"] + sub["p3"] + sub["p4"]), 0.0, 1.0)
    sub = sub.rename(columns={"p1": "cv_week1", "p2": "cv_week2", "p3": "cv_week3", "p4": "cv_week4"})
    return sub[["outcome", "time", "id", "lat", "lon", "year",
                "cv_week1", "cv_week2", "cv_week3", "cv_week4", "cv_later"]]


def make_raw_preds_from_wide_logit(wide_df, base_col_prefix, holdout_years,
                                    earlier_col=None, earlier_is_logit=True,
                                    add_cv_earlier=True, renormalize_6=True):
    """Extract raw clim-logit predictions with optional 'earlier' bin."""
    wk_cols = [f"{base_col_prefix}_week{i}" for i in range(1, 5)]
    missing = [c for c in wk_cols if c not in wide_df.columns]
    if missing:
        warnings.warn(f"Skipping raw-logit for {base_col_prefix}; missing: {', '.join(missing)}")
        return None
    if add_cv_earlier and (not earlier_col or earlier_col not in wide_df.columns):
        raise ValueError(f"unc_clim_raw requires earlier_col and it must exist in wide_df.")

    sub = wide_df[wide_df["year"].isin(holdout_years)].copy()
    sub["p1"] = expit(sub[wk_cols[0]])
    sub["p2"] = expit(sub[wk_cols[1]])
    sub["p3"] = expit(sub[wk_cols[2]])
    sub["p4"] = expit(sub[wk_cols[3]])

    if add_cv_earlier:
        pE = expit(sub[earlier_col]) if earlier_is_logit else sub[earlier_col].astype(float)
        sub["pE"] = np.clip(pE, 0.0, 1.0)
    else:
        sub["pE"] = 0.0

    sub["pL"] = np.clip(1.0 - (sub["pE"] + sub["p1"] + sub["p2"] + sub["p3"] + sub["p4"]), 0.0, 1.0)

    if renormalize_6 and add_cv_earlier:
        rs6 = sub[["pE", "p1", "p2", "p3", "p4", "pL"]].sum(axis=1)
        good = rs6.notna() & (rs6 > 0)
        for c in ["pE", "p1", "p2", "p3", "p4", "pL"]:
            sub.loc[good, c] = sub.loc[good, c] / rs6[good]

    sub = sub.rename(columns={"p1": "cv_week1", "p2": "cv_week2", "p3": "cv_week3",
                                "p4": "cv_week4", "pL": "cv_later", "pE": "cv_earlier"})
    cols = ["outcome", "time", "id", "lat", "lon", "year",
            "cv_week1", "cv_week2", "cv_week3", "cv_week4", "cv_later"]
    if add_cv_earlier:
        cols.append("cv_earlier")
    return sub[cols]


# ---------------------------------------------------------------------------
# Platt calibration
# ---------------------------------------------------------------------------

def fit_platt_weights_export(df, prob_cols, training_years,
                              outcome_col="outcome", year_col="year"):
    """Fit one-vs-rest Platt calibration weights on training years.

    IMPORTANT: matches R glm(y ~ p, ...) which uses raw probability as predictor,
    NOT logit(p). sklearn is used with raw p as the feature.
    """
    df_train = df[df[year_col].isin(training_years)].copy()
    weights_rows = []

    for bin_name in prob_cols:
        p_raw = np.clip(df_train[bin_name].values.astype(float), 1e-6, 1 - 1e-6)
        y = (df_train[outcome_col].values == bin_name).astype(float)

        ok = np.isfinite(p_raw) & ~np.isnan(y)
        if ok.sum() < 10 or len(np.unique(y[ok])) < 2:
            weights_rows.append({"bin": bin_name, "intercept": 0.0, "slope": 1.0})
            continue

        # Match R: use raw probability p as the predictor (not logit(p))
        X = p_raw[ok].reshape(-1, 1)
        clf = LogisticRegression(solver="lbfgs", C=1e9, max_iter=1000, tol=1e-5)
        clf.fit(X, y[ok])
        weights_rows.append({
            "bin": bin_name,
            "intercept": float(clf.intercept_[0]),
            "slope": float(clf.coef_[0][0]),
        })

    weights_df = pd.DataFrame(weights_rows)
    return {"weights_df": weights_df}


def platt_cv_multibin(df, prob_cols, holdout_years, true_holdout_years=(),
                       outcome_col="outcome", year_col="year",
                       cv_prefix="cv_", allowed_cells=None):
    """Cross-validated per-bin Platt calibration.

    IMPORTANT: matches R glm(y ~ p, ...) which regresses on raw probability p,
    NOT logit(p). This produces different calibrated values than the logit version.
    """
    bins = prob_cols
    out_list = []

    for test_year in holdout_years:
        train = df[(df[year_col] != test_year) & (~df[year_col].isin(true_holdout_years))].copy()
        test = df[df[year_col] == test_year].copy()
        if test.empty:
            continue
        if allowed_cells is not None:
            allowed_ids = set(allowed_cells["id"])
            train = train[train["id"].isin(allowed_ids)]

        cal_mat = np.full((len(test), len(bins)), np.nan)
        for j, b in enumerate(bins):
            p_tr = np.clip(train[b].values.astype(float), 1e-6, 1 - 1e-6)
            y_tr = (train[outcome_col].values == b).astype(float)
            ok = np.isfinite(p_tr) & ~np.isnan(y_tr)
            if ok.sum() < 10 or len(np.unique(y_tr[ok])) < 2:
                cal_mat[:, j] = test[b].values.astype(float)
                continue

            # Match R: use raw p as the predictor (not logit)
            X = p_tr[ok].reshape(-1, 1)
            clf = LogisticRegression(solver="lbfgs", C=1e9, max_iter=1000, tol=1e-5)
            clf.fit(X, y_tr[ok])

            # Predict on test using raw probability
            test_p = np.clip(test[b].values.astype(float), 1e-6, 1 - 1e-6).reshape(-1, 1)
            cal_mat[:, j] = expit(clf.intercept_[0] + clf.coef_[0][0] * test_p.flatten())

        rs = cal_mat.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        cal_mat /= rs

        test_out = test.copy()
        for j, b in enumerate(bins):
            test_out[f"{cv_prefix}{b}"] = cal_mat[:, j]
        out_list.append(test_out)

    if out_list:
        return pd.concat(out_list, ignore_index=True)
    return pd.DataFrame()


def make_calibrated_preds_from_wide(wide_df, forecast_name, variant,
                                    training_years, holdout_years, true_holdout_years,
                                    allowed_cells, spec):
    """Platt-calibrated predictions via platt_cv_multibin."""
    suf = get_forecast_variant_suffix(spec, variant)
    cols = forecast_prob_cols(forecast_name, suf)
    missing = [c for c in cols.values() if c not in wide_df.columns]
    if missing:
        warnings.warn(f"Skipping calibrated for {forecast_name} ({variant}); missing: {', '.join(missing)}")
        return None

    df = wide_df[wide_df["year"].isin(list(training_years) + list(holdout_years))].copy()
    df["week1"] = df[cols["week1"]]
    df["week2"] = df[cols["week2"]]
    df["week3"] = df[cols["week3"]]
    df["week4"] = df[cols["week4"]]
    df["later"] = np.maximum(0.0, 1.0 - (df["week1"] + df["week2"] + df["week3"] + df["week4"]))

    return platt_cv_multibin(
        df,
        prob_cols=["week1", "week2", "week3", "week4", "later"],
        holdout_years=holdout_years,
        true_holdout_years=true_holdout_years,
        outcome_col="outcome",
        year_col="year",
        cv_prefix="cv_",
        allowed_cells=allowed_cells,
    )


# ---------------------------------------------------------------------------
# Multinomial CV
# ---------------------------------------------------------------------------

def _make_multinom_clf():
    """
    Create a multinomial logistic regression classifier compatible with
    both old sklearn (<1.5, needs multi_class='multinomial') and new
    sklearn (>=1.5, multi_class removed; lbfgs is always multinomial for >2 classes).
    """
    import sklearn
    from packaging.version import Version
    kwargs = dict(solver="lbfgs", C=1e9, max_iter=5000, tol=1e-5)
    try:
        sk_version = Version(sklearn.__version__)
        if sk_version < Version("1.5"):
            kwargs["multi_class"] = "multinomial"
    except Exception:
        pass  # if version check fails, omit multi_class and rely on default
    return LogisticRegression(**kwargs)


def _fit_predict_multinom(train, test, feature_cols, outcome_col="outcome"):
    """Fit sklearn multinomial logistic regression and return predicted probs.

    Fixes vs original:
    - NaN rows in test are handled gracefully (NaN probs returned instead of crash)
    - max_iter increased to 5000 to match R nnet convergence behavior
    - Compatible with both sklearn <1.5 and >=1.5 (multi_class kwarg removed in 1.5)
    """
    classes = sorted(train[outcome_col].unique())
    if len(classes) < 2:
        return None

    X_tr = train[feature_cols].values.astype(float)
    y_tr = train[outcome_col].values
    X_te = test[feature_cols].values.astype(float)

    # Drop NaN/Inf rows from training
    bad_tr = ~np.isfinite(X_tr).all(axis=1)
    X_tr = X_tr[~bad_tr]
    y_tr = y_tr[~bad_tr]
    if len(X_tr) == 0:
        return None
    if len(np.unique(y_tr)) < 2:
        return None

    clf = _make_multinom_clf()
    try:
        clf.fit(X_tr, y_tr)
    except Exception as e:
        warnings.warn(f"_fit_predict_multinom fit failed: {e}")
        return None

    # Identify NaN/Inf rows in test — return NaN probs for those
    bad_te = ~np.isfinite(X_te).all(axis=1)
    probs = np.full((len(X_te), len(clf.classes_)), np.nan)
    if (~bad_te).any():
        probs[~bad_te] = clf.predict_proba(X_te[~bad_te])

    return pd.DataFrame(probs, columns=[f"cv_{c}" for c in clf.classes_], index=test.index)


def _parse_formula_cols(formula_str):
    """Extract RHS feature column names from an expanded formula string."""
    if "~" in formula_str:
        rhs = formula_str.split("~", 1)[1]
    else:
        rhs = formula_str
    cols = [t.strip() for t in re.split(r"[+\-\*:\(\)]", rhs)]
    return [c for c in cols if c and not c.isdigit() and c != "1"]


def compute_cv_global(formula_str, data_train, holdout_years,
                       true_holdout_years=(), data_pred=None, n_jobs=1):
    """Leave-one-year-out CV using multinomial logistic, pooling all cells.

    IMPORTANT: data_train should be restrict_to_allowed(wide_df, dissemination_cells)
    and data_pred should be wide_df (all cells). This matches R behavior where
    training uses only allowed cells but predictions are generated for all cells.
    """
#    print("data_pred  =  ", data_pred)
#    print("\ntrue_holdout_years  =  ", true_holdout_years)
    if data_pred is None:
        data_pred = data_train
    feature_cols = _parse_formula_cols(formula_str)

    results = []
    data_train = data_pred
    for test_year in holdout_years:
        train = data_train[
            (data_train["year"] != test_year) & (~data_train["year"].isin(true_holdout_years))
        ]
        test = data_pred[data_pred["year"] == test_year]
#        print("XXX train , ", train)
#        print("XXX test , ", test)
#        print("XXX feature , ", feature_cols)
#        import sys
#        sys.exit()
        if train.empty or test.empty:
            continue
        preds = _fit_predict_multinom(train, test, feature_cols)
        if preds is not None:
            result = pd.concat([test.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
            results.append(result)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def compute_cv_local(formula_str, data, holdout_years):
    """Per-cell leave-one-year-out CV."""
    feature_cols = _parse_formula_cols(formula_str)
    unique_cells = data[["id", "lat", "lon"]].drop_duplicates()
    results = []

    for _, cell in unique_cells.iterrows():
        cell_data = data[data["id"] == cell["id"]]
        cell_holdout = [y for y in holdout_years if y in cell_data["year"].values]
        for test_year in cell_holdout:
            train = cell_data[cell_data["year"] != test_year]
            test = cell_data[cell_data["year"] == test_year]
            if train.empty or test.empty:
                continue
            preds = _fit_predict_multinom(train, test, feature_cols)
            if preds is not None:
                result = pd.concat([test.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
                results.append(result)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def compute_cv_neighbors(formula_str, data, holdout_years, lat_band=2, lon_band=2):
    """Neighbor-cell CV (train on cells within +/-lat_band/lon_band)."""
    feature_cols = _parse_formula_cols(formula_str)
    unique_cells = data[["id", "lat", "lon"]].drop_duplicates()
    results = []

    for _, cell in unique_cells.iterrows():
        cell_data = data[data["id"] == cell["id"]]
        cell_holdout = [y for y in holdout_years if y in cell_data["year"].values]
        for test_year in cell_holdout:
            train = data[
                (data["year"] != test_year) &
                (abs(data["lat"] - cell["lat"]) <= lat_band) &
                (abs(data["lon"] - cell["lon"]) <= lon_band)
            ]
            test = data[(data["year"] == test_year) & (data["id"] == cell["id"])]
            if train.empty or test.empty:
                continue
            preds = _fit_predict_multinom(train, test, feature_cols)
            if preds is not None:
                result = pd.concat([test.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
                results.append(result)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def compute_cv_clusters(formula_str, data, holdout_years, cluster_var):
    """Cluster-based CV (train on cells in same cluster)."""
    feature_cols = _parse_formula_cols(formula_str)
    unique_cells = data[["id", "lat", "lon"]].drop_duplicates()
    results = []

    for _, cell in unique_cells.iterrows():
        cell_data = data[data["id"] == cell["id"]]
        cluster = cell_data[cluster_var].iloc[0]
        cell_holdout = [y for y in holdout_years if y in cell_data["year"].values]
        for test_year in cell_holdout:
            train = data[(data["year"] != test_year) & (data[cluster_var] == cluster)]
            test = data[(data["year"] == test_year) & (data["id"] == cell["id"])]
            if train.empty or test.empty:
                continue
            preds = _fit_predict_multinom(train, test, feature_cols)
            if preds is not None:
                result = pd.concat([test.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
                results.append(result)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def restrict_to_allowed(df, allowed_cells):
    """Filter df to rows whose id is in allowed_cells."""
    if allowed_cells is None:
        return df
    return df[df["id"].isin(set(allowed_cells["id"]))]


def clean_probs5(P):
    """Sanitize 5-bin probability matrix: replace non-finite with 0, normalize rows."""
    P = np.array(P, dtype=float)
    P[~np.isfinite(P)] = 0.0
    P = np.maximum(P, 0.0)
    rs = P.sum(axis=1, keepdims=True)
    zero = (rs <= 0).flatten()
    if zero.any():
        P[zero] = 1.0 / P.shape[1]
        rs = P.sum(axis=1, keepdims=True)
    return P / rs


def pooled_rps5(P, y_onehot):
    """Pooled RPS from 5-bin probability and one-hot matrices."""
    CP = np.cumsum(P, axis=1)
    CY = np.cumsum(y_onehot, axis=1)
    return float(np.mean(np.sum((CP - CY) ** 2, axis=1)))


def _fast_auc(y01, score):
    """Efficient AUC via Wilcoxon rank-sum (matches R's fast_auc)."""
    ok = np.isfinite(score) & ~np.isnan(y01)
    y = y01[ok].astype(int)
    s = score[ok]
    n1 = float(np.sum(y == 1))
    n0 = float(np.sum(y == 0))
    if n1 == 0 or n0 == 0:
        return np.nan
    ranks = pd.Series(s).rank(method="average").values
    return (np.sum(ranks[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)


def compute_fair_brier5(cv_preds, allowed_cells, n_train=30):
    """Compute fair (finite-sample corrected) Brier score for 5-bin predictions."""
    bins = ["week1", "week2", "week3", "week4", "later"]
    cols = [f"cv_{b}" for b in bins]
    sub = restrict_to_allowed(cv_preds, allowed_cells)
    sub = sub.dropna(subset=["outcome"] + cols)

    P = sub[cols].values
    Y = np.column_stack([(sub["outcome"] == b).astype(float) for b in bins])
    bs_rows = np.sum((P - Y) ** 2, axis=1)
    corr_rows = np.sum(P * (1 - P), axis=1) / (n_train - 1)
    return float(np.mean(bs_rows - corr_rows))


def compute_cell_metrics_fast(df, allowed_cells=None):
    """
    Compute per-cell and pooled Brier/RPS/AUC/pietra from cv_* probability columns.

    Matches R compute_cell_metrics_fast:
    - Brier uses 5 bins: week1..week4, later (NOT including 'earlier')
    - RPS uses up to 6 bins: earlier + week1..week4 + later (if earlier present)
    - Pooled AUC uses the fast Wilcoxon rank-sum method
    - Per-bin Brier and AUC are computed for the 'ALL' row
    - Pietra (KS statistic between pos/neg score distributions) is included
    - n (number of observations) is included
    - Returns per-cell rows PLUS an 'ALL' pooled row
    """
    # Bins for Brier (5-bin, no earlier)
    bins5 = ["week1", "week2", "week3", "week4", "later"]
    cv_cols5 = [f"cv_{b}" for b in bins5]

    # Bins for RPS (6-bin if earlier present, else 5)
    bins_rps_desired = ["earlier", "week1", "week2", "week3", "week4", "later"]
    present_rps = [b for b in bins_rps_desired if f"cv_{b}" in df.columns]
    cv_cols_rps = [f"cv_{b}" for b in present_rps]

    sub = df.dropna(subset=["outcome"] + cv_cols5).copy()
    if allowed_cells is not None:
        sub = sub[sub["id"].isin(set(allowed_cells["id"]))]

    if sub.empty:
        return pd.DataFrame()

    P5 = sub[cv_cols5].values.astype(float)
    Y5 = np.column_stack([(sub["outcome"] == b).astype(float) for b in bins5])

    # RPS matrix (6-bin if earlier available)
    if cv_cols_rps != cv_cols5:
        P_rps = np.zeros((len(sub), len(present_rps)), dtype=float)
        for j, b in enumerate(present_rps):
            col = f"cv_{b}"
            if col in sub.columns:
                vals = sub[col].values.astype(float)
                vals[~np.isfinite(vals)] = 0.0
                P_rps[:, j] = vals
        Y_rps = np.column_stack([(sub["outcome"] == b).astype(float) for b in present_rps])
    else:
        P_rps = P5.copy()
        Y_rps = Y5.copy()

    # --- Per-cell metrics ---
    rows = []
    for cell_id, g_idx in sub.groupby("id").groups.items():
        g_P5 = P5[sub.index.get_indexer(g_idx)]
        g_Y5 = Y5[sub.index.get_indexer(g_idx)]
        g_Prps = P_rps[sub.index.get_indexer(g_idx)]
        g_Yrps = Y_rps[sub.index.get_indexer(g_idx)]

        brier = float(np.mean(np.sum((g_P5 - g_Y5) ** 2, axis=1)))
        rps = pooled_rps5(g_Prps, g_Yrps)
        auc = _fast_auc(g_Y5.flatten().astype(int), g_P5.flatten())
        rows.append({"id": cell_id, "brier": brier, "rps": rps, "auc": auc, "n": len(g_idx)})

    cell_metrics = pd.DataFrame(rows)

    # --- Pooled 'ALL' row ---
    # Need positional indexing since sub may have non-contiguous index
    sub_reset = sub.reset_index(drop=True)
    P5_all = sub_reset[cv_cols5].values.astype(float)
    Y5_all = np.column_stack([(sub_reset["outcome"] == b).astype(float) for b in bins5])
    P_rps_all = sub_reset[cv_cols_rps].values.astype(float) if cv_cols_rps != cv_cols5 else P5_all.copy()
    Y_rps_all = (np.column_stack([(sub_reset["outcome"] == b).astype(float) for b in present_rps])
                 if cv_cols_rps != cv_cols5 else Y5_all.copy())
    for j in range(P_rps_all.shape[1]):
        P_rps_all[~np.isfinite(P_rps_all[:, j]), j] = 0.0

    brier_all = float(np.sum((P5_all - Y5_all) ** 2) / len(P5_all))
    rps_all = pooled_rps5(P_rps_all, Y_rps_all)
    auc_all = _fast_auc(Y5_all.flatten().astype(int), P5_all.flatten())

    # Per-bin Brier and AUC (5 bins, matches R)
    brier_by_bin = {}
    auc_by_bin = {}
    for j, b in enumerate(bins5):
        brier_by_bin[f"brier_{b}"] = float(np.mean((P5_all[:, j] - Y5_all[:, j]) ** 2))
        auc_by_bin[f"auc_{b}"] = _fast_auc(Y5_all[:, j].astype(int), P5_all[:, j])

    # Pietra: KS statistic between positive and negative score distributions
    p_flat = P5_all.flatten()
    y_flat = Y5_all.flatten().astype(int)
    pos_scores = p_flat[y_flat == 1]
    neg_scores = p_flat[y_flat == 0]
    if len(pos_scores) > 0 and len(neg_scores) > 0:
        pietra = float(ks_2samp(pos_scores, neg_scores).statistic)
    else:
        pietra = np.nan

    pooled_row = {
        "id": "ALL",
        "lat": np.nan,
        "lon": np.nan,
        "brier": brier_all,
        "rps": rps_all,
        "n": len(sub_reset),
        "auc": auc_all,
        "pietra": pietra,
    }
    pooled_row.update(brier_by_bin)
    pooled_row.update(auc_by_bin)

    pooled_df = pd.DataFrame([pooled_row])
    return pd.concat([cell_metrics, pooled_df], ignore_index=True)


def summarize_models_pooled(all_cells, baseline_model="unc_clim_raw"):
    """Compute skill scores vs a baseline from the 'ALL' row."""
    model_avg = all_cells[all_cells["id"] == "ALL"].copy()
    clim_rows = model_avg[model_avg["model"] == baseline_model]
    if clim_rows.empty:
        raise ValueError(f"Baseline model '{baseline_model}' not found in all_cells.")
    clim_row = clim_rows.iloc[0]
    brier_clim = clim_row["brier"]
    rps_clim = clim_row["rps"]
    auc_clim = clim_row["auc"]
    model_avg["brier_skill"] = 1.0 - (model_avg["brier"] / brier_clim)
    model_avg["rps_skill"] = 1.0 - (model_avg["rps"] / rps_clim)
    model_avg["AUC diff"] = (model_avg["auc"] - auc_clim) * 100.0
    return model_avg


def summarize_maps_compare(all_cells, method, clim_model="clim_raw", final_model="blended_model"):
    """Per-cell comparison of a final model vs climatology."""
    clim_df = (all_cells[all_cells["model"] == clim_model]
               .rename(columns={"brier": "brier_clim", "rps": "rps_clim", "auc": "auc_clim"})
               [["id", "brier_clim", "rps_clim", "auc_clim"]])
    final_df = (all_cells[all_cells["model"] == final_model]
                .rename(columns={"brier": "brier_final", "rps": "rps_final", "auc": "auc_final"})
                [["id", "brier_final", "rps_final", "auc_final"]])
    merged = clim_df.merge(final_df, on="id", how="outer")
    merged["auc_diff"] = merged["auc_final"] - merged["auc_clim"]
    merged["brier_skill"] = 1.0 - (merged["brier_final"] / merged["brier_clim"])
    merged["rps_skill"] = 1.0 - (merged["rps_final"] / merged["rps_clim"])
    merged["cv_method"] = method
    return merged


# ---------------------------------------------------------------------------
# MME blend optimization (Section 4 — was missing from Python translation)
# ---------------------------------------------------------------------------

def simplex_grid(n, step=0.1):
    """Generate all weight vectors on the N-simplex with given step size."""
    if n == 1:
        return np.array([[1.0]])
    vals = np.arange(0, 1 + step / 2, step)
    from itertools import product as iproduct
    grid = []
    for combo in iproduct(*([vals] * (n - 1))):
        s = sum(combo)
        if s <= 1.0 + 1e-9:
            last = 1.0 - s
            grid.append(list(combo) + [last])
    return np.array(grid)


def softmax_n(theta):
    """Softmax for N dimensions."""
    e = np.exp(theta - np.max(theta))
    return e / e.sum()


def optimize_mme_weights(P_allowed_clean, Y_allowed, w_col_names, mc_cores=1):
    """
    Optimize MME blend weights to minimize pooled RPS.

    Uses coarse simplex grid search (step=0.1) to initialize BFGS,
    matching R's Section 4 strategy exactly.

    Parameters
    ----------
    P_allowed_clean : list of np.ndarray
        List of (n_obs x 5) cleaned probability matrices, one per blend source.
    Y_allowed : np.ndarray
        (n_obs x 5) one-hot outcome matrix restricted to allowed cells.
    w_col_names : list of str
        Names for the weight columns (e.g. ["w_clim_raw", "w_ngcm_calibrated"]).
    mc_cores : int
        Number of parallel cores (not used here; kept for signature parity with R).

    Returns
    -------
    dict with keys:
        "weights": np.ndarray of optimal weights
        "rps": float optimal RPS
        "weights_df": pd.DataFrame with objective/weight columns
    """
    from scipy.optimize import minimize

    n_blend = len(P_allowed_clean)

    def fast_rps_from_w(w):
        P_mix = sum(wi * Pi for wi, Pi in zip(w, P_allowed_clean))
        P_mix = clean_probs5(P_mix)
        return pooled_rps5(P_mix, Y_allowed)

    # Coarse grid search
    weight_mat = simplex_grid(n_blend, step=0.1)
    best_rps = np.inf
    best_w = weight_mat[0]
    for w in weight_mat:
        rps = fast_rps_from_w(w)
        if np.isfinite(rps) and rps < best_rps:
            best_rps = rps
            best_w = w

    # BFGS refinement in unconstrained softmax space
    w0 = np.maximum(best_w, 1e-6)
    w0 = w0 / w0.sum()
    theta0 = np.log(w0)

    result = minimize(
        fun=lambda theta: fast_rps_from_w(softmax_n(theta)),
        x0=theta0,
        method="BFGS",
        options={"maxiter": 200},
    )
    w_star = softmax_n(result.x)
    opt_rps = fast_rps_from_w(w_star)

    weights_df = pd.DataFrame([{"objective": "rps", **dict(zip(w_col_names, w_star)), "rps": opt_rps}])
    return {"weights": w_star, "rps": opt_rps, "weights_df": weights_df}


def apply_mme_weights(mme_sources, blend_names, opt_w, id_vars, bins5=None):
    """
    Apply optimized MME weights to produce blended cv_* predictions.

    Parameters
    ----------
    mme_sources : dict name -> DataFrame with cv_* columns
    blend_names : list of str
    opt_w : np.ndarray of weights (same order as blend_names)
    id_vars : list of str (columns to keep as metadata)
    bins5 : list of str, defaults to ["week1","week2","week3","week4","later"]

    Returns
    -------
    DataFrame with id_vars + cv_week1..cv_later
    """
    if bins5 is None:
        bins5 = ["week1", "week2", "week3", "week4", "later"]
    cols5 = [f"cv_{b}" for b in bins5]

    # Join all sources on id_vars
    join_list = []
    for nm in blend_names:
        src = mme_sources[nm].copy()
        rename_map = {c: f"cv_{nm}_{c[3:]}" for c in cols5 if c in src.columns}
        src = src[id_vars + [c for c in cols5 if c in src.columns]].rename(columns=rename_map)
        join_list.append(src)

    base = join_list[0]
    for j in range(1, len(join_list)):
        base = base.merge(join_list[j], on=id_vars, how="inner")

    # Build P matrices and weighted sum
    P_list = []
    for nm in blend_names:
        sel_cols = [f"cv_{nm}_{b}" for b in bins5]
        P_list.append(base[sel_cols].values.astype(float))

    P_mix = sum(wi * Pi for wi, Pi in zip(opt_w, P_list))
    P_mix = clean_probs5(P_mix)

    result = base[id_vars].copy()
    for j, b in enumerate(bins5):
        result[f"cv_{b}"] = P_mix[:, j]
    return result


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def exclude_for_mapping(df):
    """Exclude NE and SW island cells."""
    return df[~((df["lon"] > 90) | ((df["lon"] < 75) & (df["lat"] < 12)))]


def create_cell_polygon(lat, lon, half_size=1):
    """Create a rectangular polygon for a grid cell center."""
    return pd.DataFrame({
        "lon": [lon - half_size, lon + half_size, lon + half_size, lon - half_size, lon - half_size],
        "lat": [lat - half_size, lat - half_size, lat + half_size, lat + half_size, lat - half_size],
    })


def build_polygons_for_mapping(grid_centers, allowed_cells, half_size=1):
    """Build cell polygons for choropleth mapping."""
    poly_data = grid_centers[["id", "lat", "lon"]].drop_duplicates()
    poly_data = exclude_for_mapping(poly_data)

    polys = []
    for _, cell in poly_data.iterrows():
        poly = create_cell_polygon(cell["lat"], cell["lon"], half_size=half_size)
        poly["id"] = cell["id"]
        poly["lat_center"] = cell["lat"]
        poly["lon_center"] = cell["lon"]
        polys.append(poly)

    polygons_df = pd.concat(polys, ignore_index=True) if polys else pd.DataFrame()
    allowed_ids = set(allowed_cells["id"])
    allowed_polygons_df = polygons_df[polygons_df["id"].isin(allowed_ids)] if not polygons_df.empty else polygons_df
    return {"polygons_df": polygons_df, "allowed_polygons_df": allowed_polygons_df}

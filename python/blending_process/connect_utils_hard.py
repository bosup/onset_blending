# ==============================================================================
# File: connect_utils.py
# ==============================================================================
# Purpose
#   Helper functions for 0_connect_prepare_data_to_2025_pipeline.py.
#   Provides day-to-week aggregation, logit-winsorization, rolling-window
#   rain summaries, and the main make_cv_rds_from_daylevel() converter.
#
# Function index
#   winsor_weekp(p, lo, hi)
#   logit_winsor(p, lo, hi)
#   sum_week_probs_from_dayprefix(df, day_prefix, out_prefix, ...)
#   sum_week_probs(df, prefix, ...)
#   sum_week_probs_with_day0(df, prefix, ...)
#   make_clim_logits_from_prefix(raw, input_prefix, output_tag, ...)
#   roll_sums_mat(mat, k)
#   week_max_over_starts(roll_mat, week_start_days)
#   week_min_over_starts(roll_mat, week_start_days)
#   make_cv_rds_from_daylevel(spec)
# ==============================================================================

import os
import pickle
import numpy as np
import pandas as pd
from scipy.special import logit, expit

from ..pipelines._shared.misc import coalesce


def winsor_weekp(p, lo=0.0001, hi=0.9999):
    """Winsorize probability to [lo, hi]."""
    return np.clip(p, lo, hi)


def logit_winsor(p, lo=0.0001, hi=0.9999):
    """Winsorize then apply logit transform."""
    return logit(winsor_weekp(np.asarray(p, dtype=float), lo, hi))


def sum_week_probs_from_dayprefix(df, day_prefix, out_prefix,
                                  day_max=28, days_per_week=7, n_weeks=4):
    """
    Sum daily columns <day_prefix>1..<day_prefix>N into weekly bins.
    Returns DataFrame with week columns named <out_prefix>_week1..N.
    """
    cols = [f"{day_prefix}{k}" for k in range(1, day_max + 1)]
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns for {out_prefix}: {', '.join(miss)}")
    mat = df[cols].values

    result = {}
    for w in range(1, n_weeks + 1):
        lo = (w - 1) * days_per_week
        hi = w * days_per_week
        result[f"{out_prefix}_week{w}"] = mat[:, lo:hi].sum(axis=1)
    return pd.DataFrame(result, index=df.index)


def sum_week_probs(df, prefix, day_max=28, days_per_week=7, n_weeks=4):
    """Sum <prefix>_p_onset_day_1..<day_max> into weekly bins."""
    cols = [f"{prefix}_p_onset_day_{k}" for k in range(1, day_max + 1)]
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"Missing required columns for {prefix}: {', '.join(miss)}")
    mat = df[cols].values

    result = {}
    for w in range(1, n_weeks + 1):
        lo = (w - 1) * days_per_week
        hi = w * days_per_week
        result[f"{prefix}_p_onset_week{w}"] = mat[:, lo:hi].sum(axis=1)
    return pd.DataFrame(result, index=df.index)


def sum_week_probs_with_day0(df, prefix, day_max=28, days_per_week=7, n_weeks=4):
    """Like sum_week_probs but also extracts day_0 column for 'earlier' bin."""
    col0 = f"{prefix}_p_onset_day_0"
    if col0 not in df.columns:
        raise ValueError(f"Missing required column: {col0}")
    week_tbl = sum_week_probs(df, prefix, day_max=day_max, days_per_week=days_per_week, n_weeks=n_weeks)
    return {"day0": df[col0].values.copy(), "week": week_tbl}


def make_clim_logits_from_prefix(raw, input_prefix, output_tag,
                                  day_max, days_per_week, n_weeks):
    """Aggregate climatology day probs to weeks and apply logit_winsor."""
    wk = sum_week_probs(raw, prefix=input_prefix, day_max=day_max,
                        days_per_week=days_per_week, n_weeks=n_weeks)
    result = {}
    for w in range(1, n_weeks + 1):
        col = f"{input_prefix}_p_onset_week{w}"
        result[f"prob_clim_mr_{output_tag}_week{w}"] = logit_winsor(wk[col].values)
    return pd.DataFrame(result, index=raw.index)


def roll_sums_mat(mat, k):
    """
    Rolling k-day row sums from a matrix [nrow x ncols].
    Returns [nrow x (ncols - k + 1)].
    """
    n = mat.shape[1]
    if n < k:
        return np.empty((mat.shape[0], 0))
    out = np.stack(
        [mat[:, s:s + k].sum(axis=1) for s in range(n - k + 1)],
        axis=1
    )
    return out


def week_max_over_starts(roll_mat, week_start_days):
    """Per-row max of rolling sums at specified start days (1-based)."""
    if roll_mat.shape[1] == 0:
        return np.full(roll_mat.shape[0], np.nan)
    ok = [s for s in week_start_days if 1 <= s <= roll_mat.shape[1]]
    if not ok:
        return np.full(roll_mat.shape[0], np.nan)
    idx = [s - 1 for s in ok]
    return roll_mat[:, idx].max(axis=1)


def week_min_over_starts(roll_mat, week_start_days):
    """Per-row min of rolling sums at specified start days (1-based)."""
    if roll_mat.shape[1] == 0:
        return np.full(roll_mat.shape[0], np.nan)
    ok = [s for s in week_start_days if 1 <= s <= roll_mat.shape[1]]
    if not ok:
        return np.full(roll_mat.shape[0], np.nan)
    idx = [s - 1 for s in ok]
    return roll_mat[:, idx].min(axis=1)


def make_cv_rds_from_daylevel(spec):
    """
    Main converter: reads daily combined pickle, builds weekly bins, onset
    outcomes, climatology logits, rain-based predictors, and writes wide
    pickle for the 2025 blending pipeline.

    Parameters
    ----------
    spec : dict  parsed YAML spec with mode, input_rds, output_rds, etc.

    Returns
    -------
    DataFrame  (also saved to spec["output_rds"])
    """
    input_rds = spec["input_rds"]
    output_rds = spec["output_rds"]
    day_max = coalesce(spec.get("day_max"), 28)
    days_per_week = coalesce(spec.get("days_per_week"), 7)
    n_weeks = coalesce(spec.get("n_weeks"), 4)

    with open(input_rds, "rb") as f:
        raw = pickle.load(f)
    if not isinstance(raw, pd.DataFrame):
        raw = pd.DataFrame(raw)

    if "id" not in raw.columns:
        raise ValueError("Expected an 'id' column like 'lat_lon' (e.g., 10_78).")

    id_parts = raw["id"].str.split("_", expand=True)
    if id_parts.shape[1] < 2:
        raise ValueError("Couldn't parse lat/lon from id. Expected format 'lat_lon'.")

    raw = raw.copy()
    raw["time"] = pd.to_datetime(raw["time"]).dt.date
    raw["lat"] = id_parts[0].astype(float)
    raw["lon"] = id_parts[1].astype(float)
    raw["year"] = pd.to_datetime(raw["time"]).dt.year

    # Onset threshold
    first_model = spec["forecast_models"][0]["name"]
    thresh_col = f"{first_model}_onset_thresh"
    if thresh_col not in raw.columns:
        raise ValueError(f"Missing {thresh_col} (onset threshold).")
    raw["onset_threshold"] = raw[thresh_col]

    # Outcome: bin true onset date relative to forecast init date
    if "true_onset_date" not in raw.columns:
        raise ValueError("Missing true_onset_date.")
    raw["true_onset_date"] = pd.to_datetime(raw["true_onset_date"]).dt.date
    raw["lead_day"] = (pd.to_datetime(raw["true_onset_date"]) - pd.to_datetime(raw["time"])).dt.days

    def _assign_outcome(ld):
        if pd.isna(ld) or ld <= 0:
            return None
        elif ld <= 7:
            return "week1"
        elif ld <= 14:
            return "week2"
        elif ld <= 21:
            return "week3"
        elif ld <= 28:
            return "week4"
        else:
            return "later"

    raw["outcome"] = raw["lead_day"].apply(_assign_outcome)

    # Climatology base prefix
    base_prefix = coalesce(spec.get("climatology", {}).get("base_prefix"), "clim")
    unc_prefix = coalesce(spec.get("climatology", {}).get("unconditional_prefix"), "clim_unc")

    clim_week_probs = sum_week_probs(raw, base_prefix, day_max=day_max,
                                     days_per_week=days_per_week, n_weeks=n_weeks)

    # Climatology window variants
    window_tags = spec.get("climatology", {}).get("window_tags") or []
    if window_tags:
        variant_parts = []
        for tag in window_tags:
            pref = f"{base_prefix}_{tag}"
            variant_parts.append(
                make_clim_logits_from_prefix(raw, pref, tag,
                                              day_max=day_max, days_per_week=days_per_week, n_weeks=n_weeks)
            )
        clim_variant_logits = pd.concat(variant_parts, axis=1)
    else:
        clim_variant_logits = pd.DataFrame(index=raw.index)

    # Forecast model week probabilities
    model_week_cols_list = {}
    for fm in spec["forecast_models"]:
        model_name = fm["name"]
        model_week_cols_list[model_name] = sum_week_probs(
            raw, model_name, day_max=day_max, days_per_week=days_per_week, n_weeks=n_weeks
        )
        for variant in (fm.get("variants") or []):
            variant_key = f"{model_name}_{variant}"
            model_week_cols_list[variant_key] = sum_week_probs_from_dayprefix(
                raw,
                day_prefix=f"{model_name}_p_onset_{variant}_day_",
                out_prefix=f"{model_name}_p_onset_{variant}",
                day_max=day_max,
                days_per_week=days_per_week,
                n_weeks=n_weeks,
            )
    model_week_cols = pd.concat(model_week_cols_list.values(), axis=1)

    # Unconditional climatology (has day_0 -> "earlier")
    unc = sum_week_probs_with_day0(raw, unc_prefix, day_max=day_max,
                                    days_per_week=days_per_week, n_weeks=n_weeks)
    unc_day0 = unc["day0"]
    unc_week_probs = unc["week"]

    # Build logit features
    clim_logits = pd.DataFrame({
        "prob_clim_mr_week1": logit_winsor(clim_week_probs[f"{base_prefix}_p_onset_week1"].values),
        "prob_clim_mr_week2": logit_winsor(clim_week_probs[f"{base_prefix}_p_onset_week2"].values),
        "prob_clim_mr_week3": logit_winsor(clim_week_probs[f"{base_prefix}_p_onset_week3"].values),
        "prob_clim_mr_week4": logit_winsor(clim_week_probs[f"{base_prefix}_p_onset_week4"].values),
        "prob_clim_mr_unc_earlier": logit_winsor(unc_day0),
        "prob_clim_mr_unc_week1": logit_winsor(unc_week_probs[f"{unc_prefix}_p_onset_week1"].values),
        "prob_clim_mr_unc_week2": logit_winsor(unc_week_probs[f"{unc_prefix}_p_onset_week2"].values),
        "prob_clim_mr_unc_week3": logit_winsor(unc_week_probs[f"{unc_prefix}_p_onset_week3"].values),
        "prob_clim_mr_unc_week4": logit_winsor(unc_week_probs[f"{unc_prefix}_p_onset_week4"].values),
    }, index=raw.index)

    # Rain-based predictors
    week_start_days_list = [
        list(range((w - 1) * days_per_week + 1, w * days_per_week + 1))
        for w in range(1, n_weeks + 1)
    ]
    rain_predictors_dict = {}

    for fm in spec["forecast_models"]:
        model_name = fm["name"]
        rain_preds = fm.get("rain_predictors") or []
        if not rain_preds:
            continue

        need_rain = [f"{model_name}_rain_mean_day_{k}" for k in range(1, day_max + 11)]
        miss_rain = [c for c in need_rain if c not in raw.columns]
        if miss_rain:
            raise ValueError(f"Missing {model_name} rain columns: {', '.join(miss_rain)}")
        rain_mat = raw[need_rain].values

        needs_5 = any(p in rain_preds for p in ("diff_5day", "max_5day"))
        needs_10 = "min_10day" in rain_preds

#        roll5 = roll_sums_mat(rain_mat, 5) if needs_5 else None
#        roll10 = roll_sums_mat(rain_mat, 10) if needs_10 else None
        roll5 = roll_sums_mat(rain_mat, 3) if needs_5 else None
        roll10 = roll_sums_mat(rain_mat, 7) if needs_10 else None

        if needs_5:
            max5_by_week = [week_max_over_starts(roll5, sd) for sd in week_start_days_list]
        if needs_10:
            min10_by_week = [week_min_over_starts(roll10, sd) for sd in week_start_days_list]

        for pred_type in rain_preds:
            if pred_type == "diff_5day":
                for w in range(1, n_weeks + 1):
                    rain_predictors_dict[f"diff_{model_name}_week{w}"] = (
                        max5_by_week[w - 1] - raw["onset_threshold"].values
                    )
            elif pred_type == "min_10day":
                for w in range(1, n_weeks + 1):
                    rain_predictors_dict[f"min_{model_name}_10day_week{w}"] = min10_by_week[w - 1]
            elif pred_type == "max_5day":
                for w in range(1, n_weeks + 1):
                    rain_predictors_dict[f"max_{model_name}_5day_week{w}"] = max5_by_week[w - 1]
            else:
                raise ValueError(f"Unknown rain predictor type: {pred_type}")

    rain_predictors = pd.DataFrame(rain_predictors_dict, index=raw.index)

    base_cols = ["id", "time", "year", "lat", "lon", "onset_threshold", "outcome"]
    wide_df = pd.concat(
        [raw[base_cols].reset_index(drop=True),
         clim_logits.reset_index(drop=True),
         clim_variant_logits.reset_index(drop=True),
         model_week_cols.reset_index(drop=True),
         rain_predictors.reset_index(drop=True)],
        axis=1
    )
    wide_df["outcome"] = wide_df["outcome"].astype(str).where(wide_df["outcome"].notna(), None)

    out_dir = os.path.dirname(output_rds)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_rds, "wb") as f:
        pickle.dump(wide_df, f)

    return wide_df

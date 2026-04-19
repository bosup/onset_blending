#!/usr/bin/env python3
# ==============================================================================
# Script: 1_blend_evaluation.py
# ==============================================================================
# Purpose
#   Cross-validate weekly-bin multinomial onset models using a YAML spec.
#   Saves metrics, reliability data, Platt weights, and MME blend weights.
#
# Usage (run from MO_Forecast_Code/ directory)
#   python pipelines/blending_process/1_blend_evaluation.py --spec_id cv_models
#   python pipelines/blending_process/1_blend_evaluation.py --spec_id hindcast_1965_1978
# ==============================================================================

import argparse
import os
import sys
import pickle
import yaml
import warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from python.pipelines._shared.misc import coalesce
from python.blending_process.blend_evaluation_utils import (
    build_formulas_from_spec,
    compute_cv_global,
    compute_cv_local,
    compute_cv_neighbors,
    compute_cv_clusters,
    compute_cell_metrics_fast,
    summarize_models_pooled,
    summarize_maps_compare,
    make_raw_preds_from_wide,
    make_raw_preds_from_wide_logit,
    make_raw_preds_from_wide_logit_window,
    make_calibrated_preds_from_wide,
    fit_platt_weights_export,
    platt_cv_multibin,
    input_rds_from_cutoff,
    make_cutoff_tag,
    make_year_tag,
    restrict_to_allowed,
    forecast_label,
)

def format_coord(series):
    """Format coordinate series consistently — int if whole numbers, else minimal decimal places."""
    if (series % 1 == 0).all():
        # All whole numbers — format as int
        return series.astype(int).astype(str)
    else:
        # Determine minimum decimal places needed to distinguish all values
        for decimals in range(1, 10):
            rounded = series.round(decimals)
            if (rounded == series.round(10)).all():  # stable at this precision
                return rounded.map(lambda v: f"{v:.{decimals}f}")
        # Fallback
#        return series.map(lambda v: f"{v:.6f}")
        return series.map(lambda v: f"{v:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Cross-validate weekly-bin blending models.")
    parser.add_argument("--spec_id", default="cv_models",
                        help="Spec file name (without .yml) in specs/2025_blend/")
    parser.add_argument("--cores", type=int, default=None,
                        help="Override number of parallel cores")
    args = parser.parse_args()

    spec_path = os.path.join("specs", "2025_blend", f"{args.spec_id}.yml")
    with open(spec_path, "r") as f:
        spec = yaml.safe_load(f)

    path_box = "Monsoon_Data/results/2025_model_evaluation"
    os.makedirs(path_box, exist_ok=True)

    cutoff_mode = coalesce(spec.get("run", {}).get("cutoff_mode"), "mok")
    training_years = list(map(int, coalesce(spec.get("run", {}).get("training_years"), list(range(2000, 2025)))))
    true_holdout_years = list(map(int, coalesce(spec.get("run", {}).get("true_holdout_years"), [])))
    cv_holdout_years = list(map(int, coalesce(spec.get("run", {}).get("cv_holdout_years"), [])))
    holdout_years = true_holdout_years + cv_holdout_years

    cv_methods = coalesce(spec.get("cv", {}).get("methods"), ["global"])
    cutoff_tag = make_cutoff_tag(cutoff_mode)
    year_tag = make_year_tag(holdout_years)
    output_tag = f"{cutoff_tag}{year_tag}"

    # Load dissemination cells
    dissemination_cells = pd.read_csv("Monsoon_Data/dissemination_cells.csv")
#    dissemination_cells["id"] = dissemination_cells["lat"].astype(str) + "_" + dissemination_cells["lon"].astype(str)


#    dissemination_cells["id"] = (
#        dissemination_cells["lat"].round(2).map(lambda v: f"{v:.2f}") + "_" +
#        dissemination_cells["lon"].round(2).map(lambda v: f"{v:.2f}")
#    )


    dissemination_cells["id"] = (
        format_coord(dissemination_cells["lat"]) + "_" +
        format_coord(dissemination_cells["lon"])
    )

    # Load data
    input_file = input_rds_from_cutoff(cutoff_mode)
    input_path = os.path.join("Monsoon_Data/Processed_Data/2025_pipeline_input", input_file)
    with open(input_path, "rb") as f:
        wide_df = pickle.load(f)
    wide_df = wide_df[wide_df["outcome"].notna()].copy()

    print(f"Loaded {len(wide_df)} rows from {input_path}")

    # Build formulas from spec
    formulas = build_formulas_from_spec(spec, cutoff_mode)
    print(f"Formulas: {list(formulas.keys())}")

    all_cells_list = []
    yearly_metrics_list = []

    # --------------------------------------------------------------------------
    # CV for each multinomial formula
    # --------------------------------------------------------------------------
    for model_name, formula_str in formulas.items():
        print(f"\nCV: {model_name}")
        for method in cv_methods:
            print(f"  Method: {method}")
            try:
                if method == "global":
                    cv_preds = compute_cv_global(
                        formula_str, wide_df, holdout_years,
                        true_holdout_years=true_holdout_years,
                    )
                elif method == "local":
                    cv_preds = compute_cv_local(formula_str, wide_df, holdout_years)
                elif method == "neighbors":
                    cv_preds = compute_cv_neighbors(formula_str, wide_df, holdout_years)
                elif method.startswith("cluster"):
                    cluster_var = spec.get("cv", {}).get("cluster_var", "cluster")
                    cv_preds = compute_cv_clusters(formula_str, wide_df, holdout_years, cluster_var)
                else:
                    warnings.warn(f"Unknown CV method: {method}. Skipping.")
                    continue

                if cv_preds.empty:
                    print(f"  No predictions for {model_name}/{method}")
                    continue

                # Compute metrics
                cell_metrics = compute_cell_metrics_fast(cv_preds, allowed_cells=dissemination_cells)
                cell_metrics["model"] = model_name
                cell_metrics["cv_method"] = method
                all_cells_list.append(cell_metrics)

                # Yearly metrics
                bins = ["week1", "week2", "week3", "week4", "later"]
                cv_cols = [f"cv_{b}" for b in bins]
                sub = cv_preds.dropna(subset=["outcome"] + cv_cols)
                for yr, g in sub.groupby("year"):
                    P = g[cv_cols].values.astype(float)
                    Y = np.column_stack([(g["outcome"] == b).astype(float) for b in bins])
                    brier = float(np.mean(np.sum((P - Y) ** 2, axis=1)))
                    from python.blending_process.blend_evaluation_utils import pooled_rps5
                    rps = pooled_rps5(P, Y)
                    y_long = Y.flatten()
                    p_long = P.flatten()
                    try:
                        from sklearn.metrics import roc_auc_score
                        auc = roc_auc_score(y_long, p_long) if len(np.unique(y_long)) == 2 else np.nan
                    except Exception:
                        auc = np.nan
                    yearly_metrics_list.append({
                        "year": yr, "model": model_name, "cv_method": method,
                        "brier": brier, "rps": rps, "auc": auc,
                    })

                # Save predictions
                pred_path = os.path.join(path_box, f"cv_preds_{model_name}_{method}{output_tag}.pkl")
                with open(pred_path, "wb") as f:
                    pickle.dump(cv_preds, f)

            except Exception as e:
                warnings.warn(f"Error in {model_name}/{method}: {e}")
                continue

    # --------------------------------------------------------------------------
    # Raw and calibrated forecasts (from spec$extras)
    # --------------------------------------------------------------------------
    extras = spec.get("extras") or {}

    # Clim logit baselines
    for clim_cfg in (extras.get("clim_logits") or []):
        name = clim_cfg.get("name", "clim_raw")
        base_col = clim_cfg.get("base_col_prefix", "prob_clim_mr")
        earlier_col = clim_cfg.get("earlier_col")
        add_earlier = bool(clim_cfg.get("add_cv_earlier", False))

        try:
            preds = make_raw_preds_from_wide_logit(
                wide_df, base_col, holdout_years,
                earlier_col=earlier_col,
                add_cv_earlier=add_earlier,
            )
            if preds is not None:
                cell_metrics = compute_cell_metrics_fast(preds, allowed_cells=dissemination_cells)
                cell_metrics["model"] = name
                cell_metrics["cv_method"] = "raw"
                all_cells_list.append(cell_metrics)
        except Exception as e:
            warnings.warn(f"Error in clim logit '{name}': {e}")

    # Forecast (raw + calibrated)
    for fc_cfg in (extras.get("forecasts") or []):
        fc_name = fc_cfg["name"]
        variant = coalesce(fc_cfg.get("variant"), "base")

        if fc_cfg.get("raw", False):
            try:
                preds = make_raw_preds_from_wide(wide_df, fc_name, variant, holdout_years, spec)
                if preds is not None:
                    lbl = f"{forecast_label(fc_name, variant)}_raw"
                    cell_metrics = compute_cell_metrics_fast(preds, allowed_cells=dissemination_cells)
                    cell_metrics["model"] = lbl
                    cell_metrics["cv_method"] = "raw"
                    all_cells_list.append(cell_metrics)
            except Exception as e:
                warnings.warn(f"Error in raw forecast '{fc_name}': {e}")

        if fc_cfg.get("calibrated", False):
            try:
                preds = make_calibrated_preds_from_wide(
                    wide_df, fc_name, variant, training_years, holdout_years,
                    true_holdout_years, dissemination_cells, spec,
                )
                if preds is not None and not preds.empty:
                    lbl = f"{forecast_label(fc_name, variant)}_calibrated"
                    cell_metrics = compute_cell_metrics_fast(preds, allowed_cells=dissemination_cells)
                    cell_metrics["model"] = lbl
                    cell_metrics["cv_method"] = "calibrated"
                    all_cells_list.append(cell_metrics)

                    # Save Platt weights
                    cv_cols = ["week1", "week2", "week3", "week4", "later"]
                    platt_df_full = wide_df[wide_df["year"].isin(training_years + holdout_years)].copy()
                    suf = spec.get("extras", {}).get("forecast_variants", {}).get(variant, "")
                    fc_cols = {
                        f"week{k}": f"{fc_name}_p_onset{suf}_week{k}" for k in range(1, 5)
                    }
                    platt_df_full["week1"] = platt_df_full[fc_cols["week1"]]
                    platt_df_full["week2"] = platt_df_full[fc_cols["week2"]]
                    platt_df_full["week3"] = platt_df_full[fc_cols["week3"]]
                    platt_df_full["week4"] = platt_df_full[fc_cols["week4"]]
                    import numpy as _np
                    platt_df_full["later"] = _np.maximum(
                        0.0, 1.0 - platt_df_full[["week1", "week2", "week3", "week4"]].sum(axis=1)
                    )
                    platt_result = fit_platt_weights_export(
                        platt_df_full, cv_cols, training_years, year_col="year"
                    )
                    platt_path = os.path.join(
                        path_box,
                        f"platt_weights_{lbl}_df{output_tag}.pkl"
                    )
                    with open(platt_path, "wb") as fp:
                        pickle.dump(platt_result["weights_df"], fp)
                    print(f"  Saved Platt weights: {platt_path}")

            except Exception as e:
                warnings.warn(f"Error in calibrated forecast '{fc_name}': {e}")

    # --------------------------------------------------------------------------
    # Save combined outputs
    # --------------------------------------------------------------------------
    if all_cells_list:
        all_cells = pd.concat(all_cells_list, ignore_index=True)

        summary_path = os.path.join(path_box, f"summary_models{output_tag}.pkl")
        with open(summary_path, "wb") as f:
            pickle.dump(all_cells, f)
        all_cells.to_csv(summary_path.replace(".pkl", ".csv"), index=False)
        print(f"Saved cell metrics: {summary_path}")

        # Pooled summary
        pooled = summarize_models_pooled(all_cells)
        pooled_path = os.path.join(path_box, f"summary_models_pooled{output_tag}.csv")
        pooled.to_csv(pooled_path, index=False)
        print(f"Saved pooled summary: {pooled_path}")

    if yearly_metrics_list:
        yearly_df = pd.DataFrame(yearly_metrics_list)
        yearly_path = os.path.join(path_box, f"yearly_metrics_global{output_tag}.pkl")
        with open(yearly_path, "wb") as f:
            pickle.dump(yearly_df, f)
        yearly_df.to_csv(yearly_path.replace(".pkl", ".csv"), index=False)
        print(f"Saved yearly metrics: {yearly_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()

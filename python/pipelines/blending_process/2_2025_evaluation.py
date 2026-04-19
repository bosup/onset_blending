#!/usr/bin/env python3
# ==============================================================================
# Script: 2_2025_evaluation.py
# ==============================================================================
# Purpose
#   Out-of-sample evaluation of 2025 monsoon onset forecasts. Computes Brier,
#   AUC, and RPS for the blended model, conditional and unconditional
#   climatologies (from the SENT forecast file), and Platt-calibrated NGCM
#   forecasts. Scores are computed against three ground-truth variants
#   (mr_mok_year, mr_mok_clim, mr_may1).
#
# Inputs
#   - Monsoon_Data/evaluation_2025/blended_forecasts_2025_sent.csv
#   - Monsoon_Data/evaluation_2025/ngcm_forecasts_2025_sent.csv
#   - Monsoon_Data/evaluation_2025/ngcm_forecasts_2025_sent_no_mok_filter.csv
#   - Monsoon_Data/evaluation_2025/mr_onset_with_mok_year_2025.csv
#   - Monsoon_Data/evaluation_2025/mr_onset_with_mok_clim_2025.csv
#   - Monsoon_Data/evaluation_2025/mr_onset_with_may1_2025.csv
#   - Monsoon_Data/results/2025_model_evaluation/platt_weights_*_calibrated_df_*_2000_2024.pkl
#
# Outputs
#   - Monsoon_Data/results/2025_model_evaluation/evaluation/metrics_min_brier_rps_auc_models.csv
#   - Monsoon_Data/results/2025_model_evaluation/evaluation/model_metrics_sent_vs_<on_lbl>.pkl
#   - Monsoon_Data/results/2025_model_evaluation/yearly_metrics_2025{_suffix}.pkl
#
# Usage (run from MO_Forecast_Code/ directory)
#   python pipelines/blending_process/2_2025_evaluation.py
# ==============================================================================

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from dateutil import parser as dateutil_parser

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from python.blending_process.evaluation_2025_utils import (
    INTERVAL_BINS_5,
    RPS_BINS,
    skill_score,
    diff_to_bin,
    read_onsets,
    rps_frame,
    make_probs_for_rps,
    apply_platt_5,
    read_platt_weights,
    platt_config_from_onset,
    ngcm_path_from_onset,
    ngcm_yearly_name_from_onset,
    yearly_suffix_from_onset,
)


def parse_any_date(x):
    """Parse date strings flexibly."""
    if pd.isna(x):
        return pd.NaT
    try:
        return pd.to_datetime(x)
    except Exception:
        try:
            return dateutil_parser.parse(str(x))
        except Exception:
            return pd.NaT


def pooled_auc(P, Y):
    """Pooled AUC: stack all bins into one long binary dataset."""
    from sklearn.metrics import roc_auc_score
    y_long = Y.flatten().astype(int)
    p_long = P.flatten().astype(float)
    if len(np.unique(y_long)) < 2:
        return np.nan
    try:
        return roc_auc_score(y_long, p_long)
    except Exception:
        return np.nan


def main():
    eval_dir = "Monsoon_Data/evaluation_2025"
    out_dir = "Monsoon_Data/results/2025_model_evaluation/evaluation"
    platt_dir = "Monsoon_Data/results/2025_model_evaluation"
    os.makedirs(out_dir, exist_ok=True)

    forecast_files = {
        "sent": os.path.join(eval_dir, "blended_forecasts_2025_sent.csv"),
        "ngcm": os.path.join(eval_dir, "ngcm_forecasts_2025_sent.csv"),
        "ngcm_no_mok_filter": os.path.join(eval_dir, "ngcm_forecasts_2025_sent_no_mok_filter.csv"),
    }

    onset_files = {
        "mr_mok_year": os.path.join(eval_dir, "mr_onset_with_mok_year_2025.csv"),
        "mr_mok_clim": os.path.join(eval_dir, "mr_onset_with_mok_clim_2025.csv"),
        "mr_may1":     os.path.join(eval_dir, "mr_onset_with_may1_2025.csv"),
    }

    clim_cols     = [f"clim_week{k}" for k in range(1, 5)] + ["clim_later"]
    clim_con_cols = [f"clim_con_week{k}" for k in range(1, 5)] + ["clim_con_later"]

    metrics_all_list = []

    for on_lbl, onset_path in onset_files.items():
        lbl = f"sent_vs_{on_lbl}"
        print(f"\nProcessing: {lbl}")

        # --- Read SENT baseline forecast file ---
        if not os.path.exists(forecast_files["sent"]):
            raise FileNotFoundError(f"Forecast file missing: {forecast_files['sent']}")

        sent_fc = pd.read_csv(forecast_files["sent"])
        sent_fc["time"] = sent_fc["time"].apply(parse_any_date)
        sent_fc["id"] = sent_fc["lat"].astype(str) + "_" + sent_fc["lon"].astype(str)
        sent_fc = sent_fc[sent_fc["week1"].notna()].copy()

        missing_clim = [c for c in clim_cols if c not in sent_fc.columns]
        missing_clim_con = [c for c in clim_con_cols if c not in sent_fc.columns]
        if missing_clim:
            raise ValueError(f"SENT file missing required climatology columns: {missing_clim}")
        if missing_clim_con:
            raise ValueError(f"SENT file missing required conditional climatology columns: {missing_clim_con}")

        # --- Read onsets ---
        on = read_onsets(onset_path)

        # Write pre-onset SENT file
        sent_pre_onset = sent_fc.merge(on, on="id", how="inner")
        sent_pre_onset = sent_pre_onset[
            sent_pre_onset["time"].notna() & sent_pre_onset["true_onset"].notna()
        ]
        sent_pre_onset = sent_pre_onset[sent_pre_onset["time"] < sent_pre_onset["true_onset"]]
        sent_pre_onset = sent_pre_onset.drop(columns=["true_onset"])
        pre_onset_path = os.path.join(out_dir, f"blended_forecasts_2025_sent_preonset_{on_lbl}.csv")
        sent_pre_onset.to_csv(pre_onset_path, index=False)

        # --- Join & compute outcome ---
        df = sent_fc.merge(on, on="id", how="left")
        df = df[df["true_onset"].notna() & df["time"].notna()].copy()
        df["diff_days"] = (df["true_onset"] - df["time"]).dt.days
        df["outcome"] = df["diff_days"].apply(lambda d: diff_to_bin(d) if pd.notna(d) else None)
        df = df[df["outcome"].notna() & (df["diff_days"] > 0)].copy()

        if df.empty:
            warnings.warn(f"No rows to score for {lbl}")
            continue

        outcome_chr = df["outcome"].values
        Y_base = np.column_stack([(outcome_chr == b).astype(float) for b in INTERVAL_BINS_5])

        P_forecast = df[INTERVAL_BINS_5].values.astype(float)
        P_clim     = df[clim_cols].values.astype(float)
        P_clim_con = df[clim_con_cols].values.astype(float)

        # Multiclass Brier
        brier_forecast = float(np.mean(np.sum((P_forecast - Y_base) ** 2, axis=1)))
        brier_clim     = float(np.mean(np.sum((P_clim - Y_base) ** 2, axis=1)))
        brier_clim_con = float(np.mean(np.sum((P_clim_con - Y_base) ** 2, axis=1)))

        auc_forecast = pooled_auc(P_forecast, Y_base)
        auc_clim     = pooled_auc(P_clim, Y_base)
        auc_clim_con = pooled_auc(P_clim_con, Y_base)

        brier_skill_forecast = skill_score(brier_forecast, brier_clim)
        brier_skill_clim_con = skill_score(brier_clim_con, brier_clim)

        # 6-bin RPS
        rps_forecast = float(np.nanmean(rps_frame(make_probs_for_rps(df, "forecast"), df["outcome"].values)))
        rps_clim     = float(np.nanmean(rps_frame(make_probs_for_rps(df, "clim"),     df["outcome"].values)))
        rps_clim_con = float(np.nanmean(rps_frame(make_probs_for_rps(df, "clim_con"), df["outcome"].values)))

        rps_skill_forecast = skill_score(rps_forecast, rps_clim)
        rps_skill_clim_con = skill_score(rps_clim_con, rps_clim)

        out = pd.DataFrame({
            "dataset": lbl,
            "model":   ["blended_model", "clim_raw", "unc_clim_raw"],
            "brier_skill": [brier_skill_forecast, brier_skill_clim_con, 0.0],
            "rps_skill":   [rps_skill_forecast,   rps_skill_clim_con,   0.0],
            "brier":       [brier_forecast, brier_clim_con, brier_clim],
            "rps":         [rps_forecast, rps_clim_con, rps_clim],
            "auc":         [auc_forecast, auc_clim_con, auc_clim],
        })

        # --- NGCM Platt-calibrated row ---
        ngcm_path = ngcm_path_from_onset(on_lbl, forecast_files)
        if not os.path.exists(ngcm_path):
            print(f"  NGCM forecast file missing for {lbl}: {ngcm_path} (skipping ngcm_calibrated)")
        else:
            ngcm_fc = pd.read_csv(ngcm_path)
            ngcm_fc["time"] = ngcm_fc["time"].apply(parse_any_date)
            ngcm_fc["id"] = ngcm_fc["lat"].astype(str) + "_" + ngcm_fc["lon"].astype(str)
            ngcm_fc = ngcm_fc[ngcm_fc["week1"].notna()].copy()

            # Join clim cols from SENT if missing
            missing_clim_ngcm = [c for c in clim_cols if c not in ngcm_fc.columns]
            if missing_clim_ngcm:
                join_cols = ["id", "time"] + clim_cols
                ngcm_fc = ngcm_fc.merge(
                    sent_fc[[c for c in join_cols if c in sent_fc.columns]],
                    on=["id", "time"], how="left"
                )

            df_ngcm = ngcm_fc.merge(on, on="id", how="left")
            df_ngcm = df_ngcm[df_ngcm["true_onset"].notna() & df_ngcm["time"].notna()].copy()
            df_ngcm["diff_days"] = (df_ngcm["true_onset"] - df_ngcm["time"]).dt.days
            df_ngcm["outcome"] = df_ngcm["diff_days"].apply(lambda d: diff_to_bin(d) if pd.notna(d) else None)
            df_ngcm = df_ngcm[df_ngcm["outcome"].notna() & (df_ngcm["diff_days"] > 0)].copy()

            platt_cfg = platt_config_from_onset(on_lbl)
            platt_obj = read_platt_weights(platt_dir, platt_cfg["label"], platt_cfg["cutoff_tag"])

            if not platt_obj["ok"]:
                print(f"  Platt weights not found for {lbl}: {platt_obj['path']} (skipping ngcm_calibrated)")
            elif df_ngcm.empty:
                print(f"  No NGCM rows to score for {lbl} (skipping ngcm_calibrated)")
            else:
                P_raw = df_ngcm[INTERVAL_BINS_5].values.astype(float)
                P_cal = apply_platt_5(P_raw, platt_obj["df"])

                outcome_chr_ngcm = df_ngcm["outcome"].values
                Y_ngcm = np.column_stack([(outcome_chr_ngcm == b).astype(float) for b in INTERVAL_BINS_5])

                brier_ngcm = float(np.mean(np.sum((P_cal - Y_ngcm) ** 2, axis=1)))
                auc_ngcm   = pooled_auc(P_cal, Y_ngcm)

                # 6-bin RPS for NGCM
                probs_ngcm_rps = pd.DataFrame({
                    "week1": P_cal[:, 0],
                    "week2": P_cal[:, 1],
                    "week3": P_cal[:, 2],
                    "week4": P_cal[:, 3],
                    "later": P_cal[:, 4],
                })
                probs_ngcm_rps["earlier"] = np.maximum(
                    0.0, 1.0 - probs_ngcm_rps[["week1", "week2", "week3", "week4", "later"]].sum(axis=1)
                )
                probs_ngcm_rps = probs_ngcm_rps[RPS_BINS]
                rps_ngcm = float(np.nanmean(rps_frame(probs_ngcm_rps, df_ngcm["outcome"].values)))

                brier_ref = float(out.loc[out["model"] == "unc_clim_raw", "brier"].iloc[0])
                rps_ref   = float(out.loc[out["model"] == "unc_clim_raw", "rps"].iloc[0])

                ngcm_row = pd.DataFrame({
                    "dataset": [lbl],
                    "model":   ["ngcm_calibrated"],
                    "brier_skill": [skill_score(brier_ngcm, brier_ref)],
                    "rps_skill":   [skill_score(rps_ngcm, rps_ref)],
                    "brier":       [brier_ngcm],
                    "rps":         [rps_ngcm],
                    "auc":         [auc_ngcm],
                })
                out = pd.concat([out, ngcm_row], ignore_index=True)

        # Save per-dataset pickle
        pkl_path = os.path.join(out_dir, f"model_metrics_{lbl}.pkl")
        with open(pkl_path, "wb") as fp:
            pickle.dump(out, fp)
        print(f"  Saved: {pkl_path}")

        # Save yearly metrics
        ngcm_yearly_nm  = ngcm_yearly_name_from_onset(on_lbl)
        yearly_suffix   = yearly_suffix_from_onset(on_lbl)
        yearly_out = out.copy()
        yearly_out["model"] = yearly_out["model"].apply(
            lambda m: ngcm_yearly_nm if m == "ngcm_calibrated" else m
        )
        yearly_out["year"] = 2025
        yearly_out = yearly_out[["year", "model", "brier", "brier_skill", "rps", "rps_skill", "auc"]].copy()
        yearly_out = yearly_out.rename(columns={"rps_skill": "rpss"})
        yearly_pkl = os.path.join(platt_dir, f"yearly_metrics_2025{yearly_suffix}.pkl")
        with open(yearly_pkl, "wb") as fp:
            pickle.dump(yearly_out, fp)
        yearly_out.to_csv(yearly_pkl.replace(".pkl", ".csv"), index=False)
        print(f"  Saved yearly metrics: {yearly_pkl}")

        metrics_all_list.append(out)

    # ---- Save final outputs ----
    if metrics_all_list:
        metrics_all = pd.concat(metrics_all_list, ignore_index=True)
        metrics_all.to_csv(
            os.path.join(out_dir, "metrics_min_brier_rps_auc_models.csv"), index=False
        )
        sent_mask = metrics_all["dataset"].str.startswith("sent_vs_")
        metrics_all[sent_mask].to_csv(
            os.path.join(out_dir, "metrics_min_SENT_plus_NGCMcalibrated.csv"), index=False
        )
        print(f"\nSaved combined metrics: {os.path.join(out_dir, 'metrics_min_brier_rps_auc_models.csv')}")

    print("\nDone.")


if __name__ == "__main__":
    main()

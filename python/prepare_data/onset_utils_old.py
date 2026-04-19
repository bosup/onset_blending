# ==============================================================================
# File: onset_utils.py
# ==============================================================================
# Purpose
#   Low-level utilities for computing monsoon onset indices from
#   rainfall time series.
#
# Functions
#   read_mok_dates(spec)
#   read_thresholds(spec)
#   roll_sum_na_rm_left(x, k)
#   roll_sum_na_propagate_left(x, k)
#   find_onset(series, win, thresh, reject_if_short_followup, start_day)
#   find_onset_precomp(series, win, thresh, wsum_all, pre_bad, last10start, ...)
# ==============================================================================

import os
import numpy as np
import pandas as pd
from datetime import date


def read_mok_dates(spec):
    """
    Read Monsoon Onset Kerala (MOK) dates from the file specified in spec.

    Returns a DataFrame with columns: year (int), mok_date (datetime.date).
    Returns None if spec["mok"]["file"] is not set.
    """
    if spec.get("mok") is None or spec["mok"].get("file") is None:
        return None

    f = spec["mok"]["file"]
    if not os.path.exists(f):
        raise FileNotFoundError(f"MOK file not found: {f}")

    mok = pd.read_csv(f)
    ycol = spec["mok"]["year_col"]
    dcol = spec["mok"]["day_col"]
    if ycol not in mok.columns or dcol not in mok.columns:
        raise ValueError(f"MOK file must contain columns '{ycol}' and '{dcol}'")

    base_md = spec["mok"]["base_date"]  # e.g. "01-01"
    mok["mok_date"] = mok.apply(
        lambda row: pd.to_datetime(f"{int(row[ycol])}-{base_md}") + pd.Timedelta(days=int(row[dcol])),
        axis=1
    ).dt.date
    return mok[[ycol, "mok_date"]].rename(columns={ycol: "year"}).assign(year=lambda d: d["year"].astype(int))


def read_thresholds(spec):
    """
    Read per-grid-cell onset thresholds.

    Returns a DataFrame with columns: lat, lon, onset_thresh.
    Returns None if spec["thresholds"]["file"] is not set.
    """
    if spec.get("thresholds") is None or spec["thresholds"].get("file") is None:
        return None

    f = spec["thresholds"]["file"]

    # Case 1: scalar threshold (int or float)
    if isinstance(f, (int, float)):
        return f

    # Optional: handle numeric strings like "0.5"
    try:
        val = float(f)
        return val
    except (TypeError, ValueError):
        pass

    if not os.path.exists(f):
        raise FileNotFoundError(f"Threshold file not found: {f}")

    ext = os.path.splitext(f)[1].lower()

    if ext in (".nc", ".nc4", ".netcdf"):
        import netCDF4 as nc4
        ds = nc4.Dataset(f)
        lat_name = spec["thresholds"].get("lat_var") or spec["thresholds"].get("lat_col") or "lat"
        lon_name = spec["thresholds"].get("lon_var") or spec["thresholds"].get("lon_col") or "lon"
        thr_name = spec["thresholds"].get("thresh_var") or spec["thresholds"].get("thresh_col") or "MWmean"
        lats = np.array(ds[lat_name][:]).flatten()
        lons = np.array(ds[lon_name][:]).flatten()
        thresh = np.array(ds[thr_name][:]).flatten()
        ds.close()
        return pd.DataFrame({"lat": lats, "lon": lons, "onset_thresh": thresh}).drop_duplicates()

    if ext == ".mat":
        try:
            import scipy.io as sio
        except ImportError:
            raise ImportError("scipy is required to read .mat threshold files.")
        lat_name = spec["thresholds"].get("mat_lat_name", "lat")
        lon_name = spec["thresholds"].get("mat_lon_name", "lon")
        thr_name = spec["thresholds"].get("mat_thresh_name", "onset.day.thres")
        md = sio.loadmat(f)
        lats = np.array(md[lat_name]).flatten()
        lons = np.array(md[lon_name]).flatten()
        thresh = np.array(md[thr_name])
        grid = pd.DataFrame(
            [(la, lo) for la in lats for lo in lons],
            columns=["lat", "lon"]
        )
        grid["onset_thresh"] = thresh.flatten()
        return grid.drop_duplicates()

    # CSV/TSV
    th = pd.read_csv(f)
    lc = spec["thresholds"].get("lat_col", "lat")
    oc = spec["thresholds"].get("lon_col", "lon")
    tc = spec["thresholds"].get("thresh_col", "onset_thresh")
    th = th.rename(columns={lc: "lat", oc: "lon", tc: "onset_thresh"})
    return th[["lat", "lon", "onset_thresh"]].drop_duplicates()


# ---------------------------------------------------------------------------
# Fast rolling helpers
# ---------------------------------------------------------------------------

def roll_sum_na_rm_left(x, k):
    """
    Left-aligned rolling sum of length k with NA treated as 0 (na.rm=TRUE).
    Returns array of length max(0, n - k + 1).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if k <= 0 or n < k:
        return np.array([], dtype=float)
    x0 = np.where(np.isnan(x), 0.0, x)
    cs = np.concatenate([[0.0], np.cumsum(x0)])
    return cs[k:] - cs[:n - k + 1]


def roll_sum_na_propagate_left(x, k):
    """
    Left-aligned rolling sum where any NA in the window propagates to the sum.
    Returns array of length max(0, n - k + 1).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if k <= 0 or n < k:
        return np.array([], dtype=float)
    na = np.isnan(x).astype(float)
    x0 = np.where(np.isnan(x), 0.0, x)
    cs = np.concatenate([[0.0], np.cumsum(x0)])
    cna = np.concatenate([[0.0], np.cumsum(na)])
    s = cs[k:] - cs[:n - k + 1]
    na_ct = cna[k:] - cna[:n - k + 1]
    s[na_ct > 0] = np.nan
    return s


# ---------------------------------------------------------------------------
# Unified onset finder
# ---------------------------------------------------------------------------

def find_onset(series, win=5, thresh=None, reject_if_short_followup=False, start_day=0):
    """
    Find the first onset day in a rainfall series.

    Parameters
    ----------
    series : array-like
        Daily rainfall values.
    win : int
        Rolling window size (days).
    thresh : float
        Rainfall threshold for onset.
    reject_if_short_followup : bool
        If True, require full 30-day follow-up check.
    start_day : float
        Skip candidates before this 1-based index.

    Returns
    -------
    int or None  (1-based index, or None if no onset found)
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < win or thresh is None or np.isnan(thresh):
        return None

    max_candidate = n - win + 1
    if max_candidate < 1:
        return None

    min_i = max(1, int(np.ceil(start_day)))
    if min_i > max_candidate:
        return None

    idx = np.arange(min_i, max_candidate + 1)  # 1-based
    wsum_all = roll_sum_na_rm_left(series, win)  # length max_candidate
    wsum = wsum_all[idx - 1]

    base_ok = (series[idx - 1] > 1) & (wsum > thresh)
    base_ok = np.where(np.isnan(base_ok), False, base_ok)
    cand = idx[base_ok]
    if len(cand) == 0:
        return None

    if n < 10:
        return int(cand[0])

    sum10 = roll_sum_na_propagate_left(series, 10)
    bad10 = (~np.isnan(sum10)) & (sum10 < 5)
    pre_bad = np.concatenate([[0], np.cumsum(bad10.astype(int))])
    last10start = n - 10 + 1

    return _apply_followup(cand, n, pre_bad, last10start, reject_if_short_followup)


def find_onset_precomp(series, win, thresh, wsum_all, pre_bad, last10start,
                       reject_if_short_followup=False, start_day=0):
    """
    Onset finder with precomputed rolling sums (faster for batch use).

    Parameters
    ----------
    series : ndarray
    win : int
    thresh : float
    wsum_all : ndarray  precomputed roll_sum_na_rm_left(series, win)
    pre_bad : ndarray   precomputed cumsum of bad 10-day windows
    last10start : int
    reject_if_short_followup : bool
    start_day : float

    Returns
    -------
    int or None
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < win or thresh is None or np.isnan(thresh):
        return None

    max_candidate = n - win + 1
    if max_candidate < 1:
        return None

    min_i = max(1, int(np.ceil(start_day)))
    if min_i > max_candidate:
        return None

    idx = np.arange(min_i, max_candidate + 1)
    wsum = wsum_all[idx - 1]
    base_ok = (series[idx - 1] > 1) & (wsum > thresh)
    base_ok = np.where(np.isnan(base_ok), False, base_ok)
    cand = idx[base_ok]
    if len(cand) == 0:
        return None

    if n < 10:
        return int(cand[0])

    return _apply_followup(cand, n, pre_bad, last10start, reject_if_short_followup)


def _apply_followup(cand, n, pre_bad, last10start, reject_if_short_followup):
    """Internal: apply the 10-day / 30-day follow-up check to onset candidates."""
    if reject_if_short_followup:
        cand = cand[cand + 30 <= n]
        if len(cand) == 0:
            return None
        end_start = cand + 21
        has_bad = (pre_bad[end_start] - pre_bad[cand - 1]) > 0
        ok = ~has_bad
        return int(cand[np.where(ok)[0][0]]) if np.any(ok) else None
    else:
        follow_end = np.minimum(n, cand + 30)
        follow_len = follow_end - cand + 1
        needs_check = follow_len >= 10
        ok = np.ones(len(cand), dtype=bool)

        if np.any(needs_check):
            c2 = cand[needs_check]
            fe = follow_end[needs_check]
            end_start = np.minimum(last10start, fe - 10 + 1)
            has_bad = (pre_bad[end_start] - pre_bad[c2 - 1]) > 0
            ok[needs_check] = ~has_bad

        return int(cand[np.where(ok)[0][0]]) if np.any(ok) else None

"""
test_onset.py
=============
Test script for find_onset() covering both dry-spell modes, with time-series
plots that highlight wet/dry spells and mark the detected onset date.

Usage
-----
    python test_onset.py

Output
------
    test_onset_results.png  — grid of per-case plots
    (also prints a PASS/FAIL summary to stdout)

Configuration
-------------
Edit TEST_CASES below to add, remove, or tweak cases.
Each case is a dict with keys:

    name          : str   — label shown on the plot
    series        : list  — daily rainfall values (mm)
    thresh        : float — per-cell trigger accumulation threshold (mm)
    params_spec   : dict  — passed to read_onset_params(); see yml layout below
    expected      : int | None — expected 1-based onset day (None = no onset)
    description   : str   — free-text explanation shown in plot subtitle

params_spec mirrors the yml "options" block:

    {
      "window": 3,
      "onset_definition": {
        "wet_day_min_mm": 1.0,
        "follow_days": 21,
        "dry_spell": {
          "mode": "consecutive_dry",   # or "window_sum"
          "min_dry_days": 7,
          "dry_day_min_mm": 1.0,
          # window_sum only:
          "sum_window": 10,
          "sum_min_mm": 5.0,
        }
      }
    }
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Path setup — adjust if running from a different working directory
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)

from et_blending_for_claude.python.prepare_data.onset_utils import (
    find_onset, read_onset_params, roll_sum_na_propagate_left
)

# ---------------------------------------------------------------------------
# Helpers for building params
# ---------------------------------------------------------------------------

def cd_spec(win=3, wet_mm=1.0, follow=21, min_dry=7, dry_mm=None):
    """Shorthand: consecutive_dry params spec dict."""
    return {
        "window": win,
        "onset_definition": {
            "wet_day_min_mm": wet_mm,
            "follow_days": follow,
            "dry_spell": {
                "mode": "consecutive_dry",
                "min_dry_days": min_dry,
                "dry_day_min_mm": dry_mm if dry_mm is not None else wet_mm,
            }
        }
    }


def ws_spec(win=5, wet_mm=1.0, follow=30, sum_win=10, sum_mm=5.0):
    """Shorthand: window_sum params spec dict."""
    return {
        "window": win,
        "onset_definition": {
            "wet_day_min_mm": wet_mm,
            "follow_days": follow,
            "dry_spell": {
                "mode": "window_sum",
                "sum_window": sum_win,
                "sum_min_mm": sum_mm,
            }
        }
    }


# ---------------------------------------------------------------------------
# TEST CASES
# ---------------------------------------------------------------------------

TEST_CASES = [

    # ── consecutive_dry ──────────────────────────────────────────────────────

    dict(
        name="CD-1: Clean onset, no dry spell",
        series=[8, 9, 8] + [5] * 24,
        thresh=20,
        params_spec=cd_spec(win=3, follow=21, min_dry=7),
        expected=1,
        description="All 3 trigger days wet, sum=25>20, no dry spell → onset day 1",
    ),

    dict(
        name="CD-2: First candidate vetoed, second valid",
        series=[8, 9, 8,          # trigger at day 1 (sum=25>20) ✓
                0, 0, 0, 0, 0, 0, 0,  # 7 dry days in follow-up → veto day 1
                8, 9, 8] + [5] * 21,  # trigger at day 11 (sum=25>20), no dry spell ✓
        thresh=20,
        params_spec=cd_spec(win=3, follow=21, min_dry=7),
        expected=11,
        description="Day 1 trigger vetoed by 7-day dry spell; onset at day 11",
    ),

    dict(
        name="CD-3: Dry spell just short (6 days) — no veto",
        series=[8, 9, 8, 0, 0, 0, 0, 0, 0, 5] + [5] * 21,
        thresh=20,
        params_spec=cd_spec(win=3, follow=21, min_dry=7),
        expected=1,
        description="6-day dry spell < 7 → no veto → onset day 1",
    ),

    dict(
        name="CD-4: Dry spell outside follow-up — no veto",
        series=[8, 9, 8] + [5] * 21 + [0] * 7,
        thresh=20,
        params_spec=cd_spec(win=3, follow=21, min_dry=7),
        expected=1,
        description="Dry spell starts at day 25 (after follow-up ends at day 24) → no veto",
    ),

    dict(
        name="CD-5: Trigger sum below threshold — no onset",
        series=[3, 3, 3] + [5] * 24,
        thresh=20,
        params_spec=cd_spec(win=3, follow=21, min_dry=7),
        expected=None,
        description="3-day sum=9 < thresh=20; no valid trigger in series → None",
    ),

    dict(
        name="CD-6: Middle trigger day dry — skip day 1",
        series=[8, 0.5, 9] + [5] * 24,
        thresh=20,
        params_spec=cd_spec(win=3, wet_mm=1.0, follow=21, min_dry=7),
        expected=None,
        description="Day 2 < 1 mm → day 1 fails wet-spell check; "
                    "day 2 sum=0.5+9+5=14.5<20 → no valid onset",
    ),

    dict(
        name="CD-7: Custom thresholds (win=4, min_dry=5, follow=14)",
        series=[6, 6, 6, 6] + [0, 0, 0, 0, 0] + [5] * 20,
        thresh=20,
        params_spec=cd_spec(win=4, wet_mm=1.0, follow=14, min_dry=5),
        expected=None,
        description="4-day sum=24>20 ✓ but 5-day dry spell in 14-day follow-up → veto; "
                    "no later valid trigger → None",
    ),

    dict(
        name="CD-8: start_day restriction",
        series=[8, 9, 8] + [5] * 24,
        thresh=20,
        params_spec=cd_spec(win=3, follow=21, min_dry=7),
        expected=1,
        description="Standard case; start_day=1 (default) → onset day 1",
    ),

    # ── window_sum ───────────────────────────────────────────────────────────

    dict(
        name="WS-1: Clean onset, no bad window",
        series=[5, 5, 5, 5, 5] + [5] * 35,
        thresh=20,
        params_spec=ws_spec(win=5, follow=30, sum_win=10, sum_mm=5.0),
        expected=1,
        description="5-day sum=25>20, all wet, no 10-day window < 5 mm → onset day 1",
    ),

    dict(
        name="WS-2: Bad 10-day window in follow-up — veto day 1",
        series=[5, 5, 5, 5, 5,        # trigger day 1
                0.4, 0.4, 0.4, 0.4, 0.4,
                0.4, 0.4, 0.4, 0.4, 0.4,  # 10 days × 0.4mm = 4mm < 5 → bad window
                5, 5, 5, 5, 5] + [5] * 20,
        thresh=20,
        params_spec=ws_spec(win=5, follow=30, sum_win=10, sum_mm=5.0),
        expected=16,
        description="Day 1 vetoed (bad 10-day window); onset at day 16 after dry patch",
    ),

    dict(
        name="WS-3: Bad window outside follow-up — no veto",
        series=[5, 5, 5, 5, 5] + [5] * 30 + [0.4] * 10,
        thresh=20,
        params_spec=ws_spec(win=5, follow=30, sum_win=10, sum_mm=5.0),
        expected=1,
        description="Bad 10-day window starts at day 36 (outside 30-day follow-up) → no veto",
    ),

    dict(
        name="WS-4: Trigger day not wet enough",
        series=[0.5, 5, 5, 5, 5] + [5] * 35,
        thresh=20,
        params_spec=ws_spec(win=5, follow=30, sum_win=10, sum_mm=5.0),
        expected=2,
        description="Day 1 < 1mm (dry) → trigger fails at day 1; "
                    "day 2: [5,5,5,5,5] all wet, sum=25>20 → onset day 2",
    ),

    dict(
        name="WS-5: Custom sum_mm threshold",
        series=[5, 5, 5, 5, 5,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 10-day sum=10 > sum_mm=8 → not bad
                5] + [5] * 24,
        thresh=20,
        params_spec=ws_spec(win=5, follow=30, sum_win=10, sum_mm=8.0),
        expected=1,
        description="sum_mm=8: 10-day sum=10 > 8 → not a bad window → onset day 1",
    ),

]


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def classify_days(series, params):
    """Return arrays marking wet and dry days per params."""
    s = np.asarray(series, dtype=float)
    wet = s >= params.wet_day_min_mm
    dry = (~np.isnan(s)) & (s < params.dry_day_min_mm)
    return wet, dry


def find_consecutive_dry_runs(dry, min_days):
    """Return list of (start, end) 0-based inclusive ranges for dry runs >= min_days."""
    runs = []
    n = len(dry)
    i = 0
    while i < n:
        if dry[i]:
            j = i
            while j < n and dry[j]:
                j += 1
            if j - i >= min_days:
                runs.append((i, j - 1))
            i = j
        else:
            i += 1
    return runs


def find_bad_windows(series, sum_win, sum_mm):
    """Return 0-based start positions of rolling windows with sum < sum_mm."""
    s = np.asarray(series, dtype=float)
    sw = roll_sum_na_propagate_left(s, sum_win)
    bad_starts = np.where((~np.isnan(sw)) & (sw < sum_mm))[0]
    return bad_starts


def plot_case(ax, case, params):
    """Draw a single test case on ax."""
    s = np.asarray(case["series"], dtype=float)
    n = len(s)
    days = np.arange(1, n + 1)
    onset = find_onset(s, thresh=case["thresh"], params=params)

    # --- background shading: wet spells (trigger window) ---
    wet_day = s >= params.wet_day_min_mm
    # shade individual wet days light blue
    for i, w in enumerate(wet_day):
        if w:
            ax.axvspan(i + 0.5, i + 1.5, color="#cce5ff", alpha=0.5, lw=0)

    # --- dry spell highlighting ---
    if params.mode == "consecutive_dry":
        dry_day = (~np.isnan(s)) & (s < params.dry_day_min_mm)
        runs = find_consecutive_dry_runs(dry_day, params.min_dry_days)
        for (rs, re) in runs:
            ax.axvspan(rs + 0.5, re + 1.5, color="#ffcccc", alpha=0.6, lw=0,
                       label="_dry_run")
    else:  # window_sum
        bad_starts = find_bad_windows(s, params.sum_window, params.sum_min_mm)
        for bs in bad_starts:
            ax.axvspan(bs + 0.5, bs + params.sum_window + 0.5,
                       color="#ffcccc", alpha=0.3, lw=0, label="_bad_win")

    # --- rainfall bars ---
    ax.bar(days, s, color="#4a90d9", width=0.8, zorder=3, label="Rainfall (mm)")

    # --- onset vertical line ---
    if onset is not None:
        ax.axvline(onset, color="red", lw=2, zorder=5,
                   label=f"Onset: day {onset}")
    else:
        ax.text(0.97, 0.92, "No onset", transform=ax.transAxes,
                ha="right", va="top", color="red", fontsize=8,
                bbox=dict(fc="white", ec="red", pad=2))

    # --- thresh line ---
    ax.axhline(case["thresh"] / params.win, color="orange", lw=1.2,
               ls="--", zorder=4, label=f"thresh/{params.win}={case['thresh']/params.win:.1f}")

    # --- labels ---
    status = ""
    if onset == case["expected"]:
        status = "✓ PASS"
        col = "green"
    else:
        status = f"✗ FAIL (got {onset}, exp {case['expected']})"
        col = "red"

    mode_tag = "CD" if params.mode == "consecutive_dry" else "WS"
    ax.set_title(f"{case['name']}  [{status}]", fontsize=8, color=col, pad=4)
    ax.set_xlabel("Day", fontsize=7)
    ax.set_ylabel("mm", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0.5, n + 0.5)

    # description as subtitle
    ax.text(0.01, 0.97, case["description"], transform=ax.transAxes,
            fontsize=6, va="top", wrap=True,
            bbox=dict(fc="lightyellow", ec="grey", alpha=0.7, pad=2))

    # legend (deduplicated)
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if not l.startswith("_") and l not in seen:
            seen[l] = h
    wet_patch  = mpatches.Patch(color="#cce5ff", alpha=0.8, label="Wet day")
    dry_patch  = mpatches.Patch(color="#ffcccc", alpha=0.8,
                                label="Dry spell" if params.mode == "consecutive_dry"
                                      else "Bad window")
    legend_handles = list(seen.values()) + [wet_patch, dry_patch]
    legend_labels  = list(seen.keys())  + [wet_patch.get_label(), dry_patch.get_label()]
    ax.legend(legend_handles, legend_labels, fontsize=6, loc="upper right",
              framealpha=0.7)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_tests():
    n_cases = len(TEST_CASES)
    ncols = 2
    nrows = (n_cases + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.array(axes).flatten()

    print(f"\n{'='*60}")
    print(f"  Onset detection test suite  ({n_cases} cases)")
    print(f"{'='*60}")

    passed = 0
    for i, case in enumerate(TEST_CASES):
        params = read_onset_params({"options": case["params_spec"]})
        result = find_onset(np.asarray(case["series"], dtype=float),
                            thresh=case["thresh"], params=params)
        ok = result == case["expected"]
        passed += ok
        status = "PASS ✓" if ok else f"FAIL ✗  (got {result}, expected {case['expected']})"
        print(f"  [{i+1:2d}] {case['name']:<45}  {status}")
        plot_case(axes[i], case, params)

    # hide unused axes
    for j in range(n_cases, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"Onset detection tests — {passed}/{n_cases} passed\n"
        f"Blue bars = rainfall | Blue shading = wet day | "
        f"Red shading = dry spell / bad window | Red line = onset | "
        f"Orange dashed = thresh/win",
        fontsize=9, y=1.01
    )
    fig.tight_layout()
    out = os.path.join(REPO_ROOT, "test_onset_results.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\n{'='*60}")
    print(f"  {passed}/{n_cases} passed")
    print(f"  Plot saved → {out}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_tests()

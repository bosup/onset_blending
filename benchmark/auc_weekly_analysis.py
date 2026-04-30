import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

# ── 1. Load data ──────────────────────────────────────────────────────────────
#case = "wet_spell_aifs_gencast"
#case = "wet_spell_aifs_aifs_ens"
case = "dry_spell_aifs_gencast_box6"
dir_in = "Monsoon_Data/results/"
data_in =  "summary_models_pooled_clim_mok_date_2000_2022.csv"
file_path = os.path.join(dir_in, case, data_in)
df = pd.read_csv(file_path)

dir_out = "benchmark/figure/"
os.makedirs(dir_out, exist_ok=True)


# ── 2. Models to plot ─────────────────────────────────────────────────────────
plot_bins   = ["week1", "week2", "week3", "week4"]
auc_models  = {
    "unc_clim_raw":           "Unc Clim Raw",
    "clim_raw":               "Clim Raw",
    "blended_model":          "Blended Model",
    "ngcm_clim_mok_date_raw": "GenCast",
    #"ngcm_clim_mok_date_raw": "AIFS_ENS",
}
colors = {
    "Unc Clim Raw":  "#8172B2",
    "Clim Raw":      "#4C72B0",
    "Blended Model": "#DD8452",
    "GenCast":       "#55A868",
    #"AIFS_ENS":       "#55A868",
}

# ── 3. Bar plot ───────────────────────────────────────────────────────────────
x        = np.arange(len(plot_bins))
width    = 0.19
gap      = 0.02
offsets  = np.array([-1.5, -0.5, 0.5, 1.5]) * (width + gap)

fig, ax = plt.subplots(figsize=(11, 5.5))

for i, (model_key, label) in enumerate(auc_models.items()):
    row  = df[df["model"] == model_key].iloc[0]
    vals = [row[f"auc_{b}"] for b in plot_bins]
    bars = ax.bar(x + offsets[i], vals, width=width,
                  color=colors[label], edgecolor="white", linewidth=0.8,
                  zorder=3, label=label)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=7, fontweight="medium", color="#333333")

ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#cccccc")
ax.set_xticks(x)
ax.set_xticklabels(["Week 1", "Week 2", "Week 3", "Week 4"], fontsize=11)
ax.set_ylabel("AUC", fontsize=11)
ax.set_title("AUC by Forecast Week", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10, framealpha=0.9, edgecolor="#cccccc")
ax.spines[["top", "right"]].set_visible(False)

# Zoom y-axis so bars are distinguishable
ymin = min(df[df["model"].isin(auc_models)][f"auc_{b}"].min() for b in plot_bins) - 0.04
ax.set_ylim(bottom=max(0, ymin))

plt.tight_layout()
#plt.savefig("auc_weekly_barplot.png", dpi=150, bbox_inches="tight")
fout = os.path.join(dir_out, f"auc_barplot_{case}.png")
plt.savefig(fout, dpi=150, bbox_inches="tight")
plt.show()

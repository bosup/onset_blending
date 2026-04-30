import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

# ── 1. Load data ──────────────────────────────────────────────────────────────
#case = "wet_spell_aifs_aifs_ens"
#case = "wet_spell_aifs_gencast_test"
case = "dry_spell_aifs_gencast_box6"
dir_in = "Monsoon_Data/results/"
data_in =  "summary_models_pooled_clim_mok_date_2000_2022.csv"
#data_in =  "summary_models_pooled_clim_mok_date_2019_2022.csv"
file_path = os.path.join(dir_in, case, data_in)
df = pd.read_csv(file_path)

dir_out = "benchmark/figure/"
os.makedirs(dir_out, exist_ok=True)

# ── 2. Compute Brier Skill Scores (reference: unc_clim_raw) ──────────────────
clim = df[df["model"] == "unc_clim_raw"].iloc[0]
bins = ["week1", "week2", "week3", "week4", "later"]
for b in bins:
    df[f"brier_skill_{b}"] = 1.0 - (df[f"brier_{b}"] / clim[f"brier_{b}"])

print(df[["model"] + [f"brier_skill_{b}" for b in bins]])

# ── 3. Models to plot ─────────────────────────────────────────────────────────
plot_bins   = ["week1", "week2", "week3", "week4"]
plot_models = {
    "clim_raw":               "Clim Raw",
    "blended_model":          "Blended Model",
    "ngcm_clim_mok_date_raw": "GenCast",
    #"ngcm_clim_mok_date_raw": "AIFS_ENS",
}
colors = {
    "Clim Raw":      "#4C72B0",
    "Blended Model": "#DD8452",
    "GenCast":       "#55A868",
    #"AIFS_ENS":       "#55A868",
}

# ── 4. Bar plot ───────────────────────────────────────────────────────────────
x       = np.arange(len(plot_bins))
width   = 0.24
gap     = 0.03
offsets = np.array([-1, 0, 1]) * (width + gap)

fig, ax = plt.subplots(figsize=(10, 5.5))

for i, (model_key, label) in enumerate(plot_models.items()):
    row  = df[df["model"] == model_key].iloc[0]
    vals = [row[f"brier_skill_{b}"] for b in plot_bins]
    bars = ax.bar(x + offsets[i], vals, width=width,
                  color=colors[label], edgecolor="white", linewidth=0.8,
                  zorder=3, label=label)
    for bar, v in zip(bars, vals):
        va  = "bottom" if v >= 0 else "top"
        off = 0.004  if v >= 0 else -0.004
        ax.text(bar.get_x() + bar.get_width()/2, v + off,
                f"{v:.3f}", ha="center", va=va,
                fontsize=7.5, fontweight="medium", color="#333333")

ax.axhline(0, color="#555555", linewidth=0.9, linestyle="--", zorder=2)
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="#cccccc")
ax.set_xticks(x)
ax.set_xticklabels(["Week 1", "Week 2", "Week 3", "Week 4"], fontsize=11)
ax.set_ylabel("Brier Skill Score", fontsize=11)
ax.set_title("Brier Skill Score by Forecast Week\n(Reference: unc_clim_raw)",
             fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10, framealpha=0.9, edgecolor="#cccccc")
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
#plt.savefig("bss_barplot"+case+".png", dpi=150, bbox_inches="tight")
fout = os.path.join(dir_out, f"bss_barplot_{case}.png")
plt.savefig(fout, dpi=150, bbox_inches="tight")
plt.show()

import pandas as pd
from pathlib import Path
import sys
import os

#sys.path.insert(0, ".")   # so Python can find the blend_final package

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Add the repo root to sys.path so the 'python' package can be found
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.chdir(REPO_ROOT)

from utils.maps import make_maps

#dir_in = "/Users/bodong/Code/project/et_blending/Monsoon_Data/results/wet_spell_aifs_aifs_ens/exports/"
dir_in = "Monsoon_Data/results/wet_spell_aifs_aifs_ens/exports/"
fname = os.path.join(dir_in, "blend_output_summary_20220617.csv")

#summary = pd.read_csv("blend_output_summary_20220617.csv", parse_dates=["time"])
summary = pd.read_csv(fname, parse_dates=["time"])

make_maps(summary, use_cartopy=True, region='Ethiopia', output_path=Path("predict/output/"))

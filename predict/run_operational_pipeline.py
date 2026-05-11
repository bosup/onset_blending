"""
run_operational_pipeline.py
===========================
Runs the full operational blending pipeline for a given forecast year/issue date
in a single command. Each step is executed sequentially; the script verifies that
expected output files exist before proceeding to the next step.

Steps
-----
1. 1_process_raw_nc_files.py  --spec_id <aifs_spec>
2. 1_process_raw_nc_files.py  --spec_id <aifs_ens_spec>
3. 2_build_climatology.py     --spec_id <clim_spec>
4. 3_combine_datasets.py      --spec_id <combine_spec>
5. 0_connect_prepare_data_to_2025_pipeline.py --spec_id <connect_spec>
6. apply_blend_model.py       (with user-supplied coef args)
7. export_blend_output.py     (with --preds_file)
8. run_maps.py                (with --input_file)

When --aifs_nc_folder, --aifs_ens_nc_folder, --gt_path are supplied, the script
patches the relevant yml fields before running each step by writing temporary
spec files (suffixed _op) into the specs/ directories. These are cleaned up on
exit.

Usage (run from repo root)
--------------------------
    python predict/run_operational_pipeline.py \\
        --year            2026 \\
        --issue_date      2026-06-09 \\
        --aifs_spec       aifs_2026 \\
        --aifs_ens_spec   aifs_ens_2026 \\
        --clim_spec       imd_clim_mok_date_2026 \\
        --combine_spec    combine_template_clim_mok_date_2026 \\
        --connect_spec    connect_clim_mok_date_2026 \\
        --blend_spec      cv_models_clim_mok_date_2026 \\
        --coef_dir        Monsoon_Data/results/wet_spell_aifs_aifs_ens \\
        --coef_tag        clim_mok_date_2022_year2022 \\
        --blend_input     Monsoon_Data/Processed_Data/2026/cv_data_clim_mok_date_new_pipeline_2026.pkl \\
        --work_dir        Monsoon_Data/Processed_Data/2026 \\
        [--aifs_nc_folder     /path/to/aifs/nc/files] \\
        [--aifs_ens_nc_folder /path/to/aifs_ens/nc/files] \\
        [--gt_path            Monsoon_Data/Processed_Data/Models/wet_spell/imd_clim_mok_date_wide.pkl] \\
        [--map_output_path    predict/output/2026/] \\
        [--skip_to STEP] \\
        [--dry_run]

Notes
-----
--aifs_nc_folder / --aifs_ens_nc_folder
    Override the input.nc_folder field in the aifs / aifs_ens yml specs.

--gt_path
    Path to the historical ground truth wide pkl. Simultaneously overrides:
      - input.gt_path  in the clim spec     (imd_clim_mok_date_2026.yml)
      - ground_truth_wide_rds in the combine spec (combine_template_*_2026.yml)
    Both fields must point to the same file, so a single arg controls both.

--skip_to N
    Skip steps 1..N-1 and start from step N (1-indexed). Useful for resuming
    after a failure without rerunning expensive earlier steps.

--dry_run
    Print the commands that would be run without executing them.
"""

import os
import sys
import argparse
import subprocess
import textwrap
import shutil
import atexit
from datetime import datetime

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.chdir(REPO_ROOT)

# ---------------------------------------------------------------------------
# Temp spec management
# ---------------------------------------------------------------------------

_temp_spec_files = []   # paths to clean up on exit

def _cleanup_temp_specs():
    for path in _temp_spec_files:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

atexit.register(_cleanup_temp_specs)


def write_patched_spec(base_spec_id, spec_type, patches):
    """
    Load specs/<spec_type>/<base_spec_id>.yml, apply nested key patches,
    write to specs/<spec_type>/<base_spec_id>_op.yml, and return the new spec_id.

    patches is a list of (dotted_key, value) tuples, e.g.:
        [("input.nc_folder", "/new/path"), ("input.gt_path", "/other/path")]
    """
    src_path = os.path.join("specs", spec_type, f"{base_spec_id}.yml")
    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Base spec not found: {src_path}")

    with open(src_path) as f:
        spec = yaml.safe_load(f)

    for dotted_key, value in patches:
        keys = dotted_key.split(".")
        node = spec
        for k in keys[:-1]:
            if k not in node:
                node[k] = {}
            node = node[k]
        node[keys[-1]] = value

    new_spec_id = f"{base_spec_id}_op"
    dst_path = os.path.join("specs", spec_type, f"{new_spec_id}.yml")
    with open(dst_path, "w") as f:
        yaml.dump(spec, f, default_flow_style=False, allow_unicode=True)

    _temp_spec_files.append(dst_path)
    log(f"Patched spec written: {dst_path}")
    return new_spec_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def abort(msg):
    log(msg, level="ERROR")
    sys.exit(1)


def check_output_exists(path, step_name):
    if not os.path.exists(path):
        abort(
            f"Step '{step_name}' completed but expected output not found:\n"
            f"  {path}\n"
            f"Check the step's logs above for errors."
        )
    log(f"Output verified: {path}")


def run_step(step_num, total, name, cmd, expected_output, dry_run):
    log(f"── Step {step_num}/{total}: {name} ──────────────────────────────")
    cmd_str = " ".join(cmd)
    log(f"Command: {cmd_str}")

    if dry_run:
        log("(dry run — skipping execution)")
        return

    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        abort(f"Step '{name}' failed with exit code {result.returncode}.")

    if expected_output:
        check_output_exists(expected_output, name)

    log(f"Step {step_num}/{total} complete.\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the full operational blending pipeline in one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Example
            -------
            python predict/run_operational_pipeline.py \\
                --year 2026 \\
                --issue_date 2026-06-09 \\
                --aifs_spec aifs_2026 \\
                --aifs_ens_spec aifs_ens_2026 \\
                --clim_spec imd_clim_mok_date_2026 \\
                --combine_spec combine_template_clim_mok_date_2026 \\
                --connect_spec connect_clim_mok_date_2026 \\
                --blend_spec cv_models_clim_mok_date_2026 \\
                --coef_dir Monsoon_Data/results/wet_spell_aifs_aifs_ens \\
                --coef_tag clim_mok_date_2022_year2022 \\
                --blend_input Monsoon_Data/Processed_Data/2026/cv_data_clim_mok_date_new_pipeline_2026.pkl \\
                --work_dir Monsoon_Data/Processed_Data/2026 \\
                --aifs_nc_folder /data/forecasts/aifs/2026 \\
                --aifs_ens_nc_folder /data/forecasts/aifs_ens/2026 \\
                --gt_path Monsoon_Data/Processed_Data/Models/wet_spell/imd_clim_mok_date_wide.pkl
        """),
    )

    # ── Required ─────────────────────────────────────────────────────────
    parser.add_argument("--year",          required=True)
    parser.add_argument("--issue_date",    required=True,
                        help="Forecast issue date, e.g. 2026-06-09")
    parser.add_argument("--aifs_spec",     required=True,
                        help="Base spec ID for aifs 1_process_raw_nc_files, e.g. aifs_2026")
    parser.add_argument("--aifs_ens_spec", required=True,
                        help="Base spec ID for aifs_ens 1_process_raw_nc_files, e.g. aifs_ens_2026")
    parser.add_argument("--clim_spec",     required=True,
                        help="Base spec ID for 2_build_climatology, e.g. imd_clim_mok_date_2026")
    parser.add_argument("--combine_spec",  required=True,
                        help="Base spec ID for 3_combine_datasets, e.g. combine_template_clim_mok_date_2026")
    parser.add_argument("--connect_spec",  required=True,
                        help="Spec ID for 0_connect_prepare_data_to_2025_pipeline")
    parser.add_argument("--blend_spec",    required=True,
                        help="Spec ID for apply_blend_model")
    parser.add_argument("--coef_dir",      required=True,
                        help="Directory containing the blending model coef pkl")
    parser.add_argument("--coef_tag",      required=True,
                        help="Coef tag passed to apply_blend_model --coef_tag")
    parser.add_argument("--blend_input",   required=True,
                        help="Path to the wide pipeline pkl for apply_blend_model --input_path")
    parser.add_argument("--work_dir",      required=True,
                        help="Working output directory for intermediate and final files")

    # ── yml field overrides ───────────────────────────────────────────────
    parser.add_argument("--aifs_nc_folder", default=None,
                        help="Override input.nc_folder in the aifs spec yml")
    parser.add_argument("--aifs_ens_nc_folder", default=None,
                        help="Override input.nc_folder in the aifs_ens spec yml")
    parser.add_argument("--gt_path", default=None,
                        help="Path to historical ground truth wide pkl. Overrides "
                             "input.gt_path in the clim spec AND ground_truth_wide_rds "
                             "in the combine spec (both must point to the same file)")

    # ── Optional ─────────────────────────────────────────────────────────
    parser.add_argument("--map_output_path", default=None,
                        help="Output directory for maps. Default: predict/output/{year}/")
    parser.add_argument("--blend_model",   default="blended_model",
                        help="Blended model name (default: blended_model)")
    parser.add_argument("--region",        default="Ethiopia",
                        help="Region passed to run_maps (default: Ethiopia)")
    parser.add_argument("--skip_to",       type=int, default=1,
                        help="Skip to step N (1-indexed). Default: 1 (run all steps)")
    parser.add_argument("--dry_run",       action="store_true",
                        help="Print commands without executing them")
    args = parser.parse_args()

    # ── Derived paths ─────────────────────────────────────────────────────
    year       = args.year
    issue_date = args.issue_date
    work_dir   = args.work_dir
    map_out    = args.map_output_path or os.path.join("predict", "output", year)
    date_compact = issue_date.replace("-", "")

    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(map_out,  exist_ok=True)

    # ── Patch specs where overrides are provided ──────────────────────────
    aifs_spec     = args.aifs_spec
    aifs_ens_spec = args.aifs_ens_spec
    clim_spec     = args.clim_spec
    combine_spec  = args.combine_spec

    if args.aifs_nc_folder:
        aifs_spec = write_patched_spec(
            args.aifs_spec, "raw_data",
            [("input.nc_folder", args.aifs_nc_folder)]
        )

    if args.aifs_ens_nc_folder:
        aifs_ens_spec = write_patched_spec(
            args.aifs_ens_spec, "raw_data",
            [("input.nc_folder", args.aifs_ens_nc_folder)]
        )

    if args.gt_path:
        clim_spec = write_patched_spec(
            args.clim_spec, "raw_data",
            [("input.gt_path", args.gt_path)]
        )
        combine_spec = write_patched_spec(
            args.combine_spec, "combine",
            [("ground_truth_wide_rds", args.gt_path)]
        )

    # ── Expected output paths ─────────────────────────────────────────────
    aifs_pkl     = os.path.join(work_dir, f"aifs_{year}_wide.pkl")
    aifs_ens_pkl = os.path.join(work_dir, f"aifs_ens_{year}_wide.pkl")
    connect_pkl  = args.blend_input
    preds_pkl    = os.path.join(work_dir, f"{args.blend_model}_global_year{year}_preds.pkl")
    export_csv   = os.path.join(work_dir, f"blend_output_summary_{date_compact}.csv")

    TOTAL = 8

    steps = [
        (1, "Process aifs nc files", [
            sys.executable,
            "python/pipelines/prepare_data/1_process_raw_nc_files.py",
            "--spec_id", aifs_spec,
        ], aifs_pkl),

        (2, "Process aifs_ens nc files", [
            sys.executable,
            "python/pipelines/prepare_data/1_process_raw_nc_files.py",
            "--spec_id", aifs_ens_spec,
        ], aifs_ens_pkl),

        (3, "Build climatology", [
            sys.executable,
            "python/pipelines/prepare_data/2_build_climatology.py",
            "--spec_id", clim_spec,
        ], None),

        (4, "Combine datasets", [
            sys.executable,
            "python/pipelines/prepare_data/3_combine_datasets.py",
            "--spec_id", combine_spec,
        ], None),

        (5, "Connect/prepare pipeline input", [
            sys.executable,
            "python/pipelines/blending_process/0_connect_prepare_data_to_2025_pipeline.py",
            "--spec_id", args.connect_spec,
        ], connect_pkl),

        (6, "Apply blend model", [
            sys.executable,
            "predict/apply_blend_model.py",
            "--spec_id",    args.blend_spec,
            "--model",      args.blend_model,
            "--year",       year,
            "--coef_dir",   args.coef_dir,
            "--coef_tag",   args.coef_tag,
            "--input_path", args.blend_input,
            "--out_dir",    work_dir,
        ], preds_pkl),

        (7, "Export blend output", [
            sys.executable,
            "predict/export_blend_output.py",
            "--issue_date", issue_date,
            "--spec_id",    args.blend_spec,
            "--preds_file", preds_pkl,
            "--out_dir",    work_dir,
        ], export_csv),

        (8, "Generate maps", [
            sys.executable,
            "predict/run_maps.py",
            "--input_file",  export_csv,
            "--output_path", map_out,
            "--region",      args.region,
        ], None),
    ]

    # ── Run ───────────────────────────────────────────────────────────────
    log(f"Starting operational pipeline for year={year}, issue_date={issue_date}")
    log(f"Work dir    : {work_dir}")
    log(f"Map output  : {map_out}")
    if args.aifs_nc_folder:
        log(f"aifs nc_folder override      : {args.aifs_nc_folder}")
    if args.aifs_ens_nc_folder:
        log(f"aifs_ens nc_folder override  : {args.aifs_ens_nc_folder}")
    if args.gt_path:
        log(f"gt_path override (clim+combine): {args.gt_path}")
    if args.skip_to > 1:
        log(f"Skipping steps 1–{args.skip_to - 1} (--skip_to {args.skip_to})")
    if args.dry_run:
        log("DRY RUN — commands will be printed but not executed")
    print()

    for step_num, name, cmd, expected_output in steps:
        if step_num < args.skip_to:
            log(f"── Step {step_num}/{TOTAL}: {name} [SKIPPED] ──")
            continue
        run_step(step_num, TOTAL, name, cmd, expected_output, args.dry_run)

    if not args.dry_run:
        log(f"Pipeline complete. Outputs in : {work_dir}")
        log(f"Maps in                       : {map_out}")
    else:
        log("Dry run complete — no files were created.")


if __name__ == "__main__":
    main()

# Replication Package: Designing probabilistic AI monsoon forecasts to inform agricultural decision-making

Replication code for probabilistic Indian monsoon onset forecasts that blend rainfall observations (IMD) with AI weather prediction model forecasts (NeuralGCM, AIFS) through multinomial blending models. This package reproduces all results from data preparation through cross-validated evaluation to publication figures.

The data used in the original paper can be found at https://doi.org/10.5281/zenodo.18894299. Before running this pipeline, all three folders found in this dataset need to be added to `Monsoon_Data/raw_nc`.

> **Note:** This is the Python port of the original R pipeline. All pipeline logic, spec files, and output formats are equivalent to the R version, but all scripts are now `.py` files run with Python 3.10+.

---

## Adapting This Code to Your Own Data

All pipeline behaviour is controlled by YAML spec files. You can point the pipeline at your own data without modifying any Python code.

- **Different ground truth rainfall**: Create a new spec in `specs/raw_data/` modelled after `imd_clim_mok_date.yml`. Point `input.nc_folder` at your NetCDF files and configure the variable name, dimension mappings, and onset thresholds for your grid.
- **Different AI forecast models**: Create a new spec in `specs/raw_data/` modelled after `ngcm.yml` (for ensembles) or `aifs.yml`. The pipeline handles ensemble forecasts in NetCDF format with configurable dimension names and variable mappings. The `name` field under `forecast_models` in the connect spec controls all downstream column names and formula terms — changing it there propagates everywhere automatically.
- **Different grid resolutions**: Set `cell_transform_enabled: true` in your raw data spec and provide a weights file mapping source grid cells to target grid cells.
- **Different onset definitions**: Edit `options.onset_definition` in your raw data spec (see [Onset Definition](#onset-definition) below). All numerical parameters — trigger window, wet-day threshold, accumulation threshold, follow-up period, and dry-spell check — are fully configurable from the yml. No code changes needed.
- **Different blending model formulas**: Edit `models.formulas` in `specs/2025_blend/cv_models*.yml`. Formula terms use `_qx` as a shorthand that expands to `_week1` through `_week4` at runtime.
- **Different forecast systems in the MME**: Edit `mme.blend_models` in `specs/2025_blend/cv_models*.yml`.
- **Different rain predictors**: Edit `rain_predictors` under each model in `specs/2025_blend/connect_*.yml`. Supports both legacy string format (`diff_5day`) and new dict format (`{ agg: diff, window: 5 }`). The dict format is preferred as it generalises to any window size without code changes.

If you add new input sources, you will need to create matching `combine` and `2025_blend` spec files so they are included in the combined dataset and blending pipeline.

---

## Repository Layout

```
et_blending/
├── python/
│   ├── _shared/                      Core utilities
│   │   ├── misc.py                     Null-coalescing and general helpers
│   │   └── read_spec.py                YAML spec loading and validation
│   ├── prepare_data/                 Data preparation helpers
│   │   ├── nc_utils.py                 NetCDF reading, regridding, wide-table construction,
│   │   │                               onset processing pipeline driver
│   │   ├── onset_utils.py              Onset detection (two dry-spell modes), MOK dates,
│   │   │                               threshold loading, onset parameter parsing
│   │   ├── climatology_utils.py        KDE fitting, climatological forecasts
│   │   └── combine_forecasts_utils.py  Merging climatology + forecast families
│   └── blending_process/             Blending pipeline helpers
│       ├── connect_utils.py            Day-to-week aggregation, logit transforms,
│       │                               rain predictor computation (generic window/agg)
│       ├── blend_evaluation_utils.py   CV evaluation, multinomial logistic regression,
│       │                               Platt calibration, formula expansion
│       ├── evaluation_2025_utils.py    Out-of-sample scoring utilities
│       └── blend_figure_utils.py       Figure generation utilities
├── pipelines/
│   ├── prepare_data/
│   │   ├── 1_process_raw_nc_files.py
│   │   ├── 2_build_climatology.py
│   │   └── 3_combine_datasets.py
│   └── blending_process/
│       ├── 0_connect_prepare_data_to_2025_pipeline.py
│       ├── 1_blend_evaluation.py
│       ├── 2_2025_evaluation.py
│       └── 3_produce_figures.py
├── specs/
│   ├── raw_data/                     NetCDF input specs (aifs, ngcm, imd variants)
│   ├── combine/                      Data combination templates
│   └── 2025_blend/                   Blended model specs (formulas, MME config, connect specs)
├── Monsoon_Data/                     Data directory (not tracked in git)
│   ├── raw_nc/                         Raw NetCDF inputs (IMD, NGCM, AIFS)
│   ├── reference/                      Onset thresholds, MOK dates
│   ├── Processed_Data/
│   │   ├── Models/                       Per-system onset tables (.pkl)
│   │   ├── Climatology/                  KDE climatology forecasts (.pkl)
│   │   ├── Combined/                     Merged modeling-ready wide tables (.pkl)
│   │   └── 2025_pipeline_input/          Weekly-bin data for blending (.pkl)
│   ├── results/
│   │   └── 2025_model_evaluation/        Model metrics, blend weights, figures
│   └── evaluation_2025/                Out-of-sample forecast + ground truth files
└── test_onset.py                     Standalone onset detection test suite with plots
```

---

## Prerequisites

### Required Input Data

| Path | Description |
|------|-------------|
| `Monsoon_Data/raw_nc/IMD_2by2/` | IMD gridded rainfall NetCDF files (`data_YYYY.nc`), one per year |
| `Monsoon_Data/raw_nc/ngcm/` | NeuralGCM ensemble forecast NetCDF files, one per year |
| `Monsoon_Data/raw_nc/aifs/` | AIFS ensemble forecast NetCDF files, one per year |
| `Monsoon_Data/reference/thresholds_df.csv` | Per-grid-cell onset accumulation thresholds (`lat`, `lon`, `onset_threshold`) |
| `Monsoon_Data/reference/MOK Onset May.csv` | Monsoon Onset Kerala dates by year (`Unnamed: 0`, `Year`, `MOK`) |
| `Monsoon_Data/evaluation_2025/` | Out-of-sample forecast and ground truth CSVs (for stage 2) |

#### Reference file formats

**`thresholds_df.csv`** — one row per grid cell:
```
Unnamed: 0,lat,lon,onset_threshold
1,7.5,37.5,20.0
...
```

**`MOK Onset May.csv`** — one row per year; `MOK` is days since `base_date` (May 1):
```
Unnamed: 0,Year,MOK
1,2000,14
...
```

### Python Dependencies

```bash
pip install numpy pandas scipy netCDF4 pyyaml matplotlib
```

The pipeline has been tested on Python 3.10+. No R installation is required.

---

## Running the Pipeline

All scripts must be run from the repository root. Scripts use relative paths that break from other working directories.

### Stage 1: Prepare Data

```bash
# Process raw NetCDF files into per-system onset tables
python python/pipelines/prepare_data/1_process_raw_nc_files.py --spec_id imd_clim_mok_date
python python/pipelines/prepare_data/1_process_raw_nc_files.py --spec_id ngcm
python python/pipelines/prepare_data/1_process_raw_nc_files.py --spec_id aifs

# Build KDE climatology forecasts from IMD onset dates
python python/pipelines/prepare_data/2_build_climatology.py --spec_id imd_clim_mok_date

# Combine ground truth, model forecasts, and climatology into wide tables
python python/pipelines/prepare_data/3_combine_datasets.py --spec_id combine_template_clim_mok_date_2025
```

**Outputs**: `Monsoon_Data/Processed_Data/Combined/combine_template_clim_mok_date_2025_combined_wide.pkl`

### Stage 2: Blending Pipeline

```bash
# Convert daily onset probabilities to weekly bins
python python/pipelines/blending_process/0_connect_prepare_data_to_2025_pipeline.py --spec_id connect_clim_mok_date

# Cross-validated model evaluation + MME weight optimisation
python python/pipelines/blending_process/1_blend_evaluation.py

# Out-of-sample 2025 evaluation
python python/pipelines/blending_process/2_2025_evaluation.py

# Publication figures
python python/pipelines/blending_process/3_produce_figures.py
```

**Outputs**: `Monsoon_Data/results/2025_model_evaluation/`

---

## Operational Forecasting (Single New Forecast Year)

For generating a forecast for a new year (e.g. 2026), use `predict/run_operational_pipeline.py`. This runs all 8 pipeline steps in sequence with a single command, verifying that each step's output exists before proceeding to the next.

### What it does

| Step | Script | Description |
|------|--------|-------------|
| 1 | `1_process_raw_nc_files.py` | Process aifs NetCDF → onset probabilities pkl |
| 2 | `1_process_raw_nc_files.py` | Process aifs_ens NetCDF → onset probabilities pkl |
| 3 | `2_build_climatology.py` | Build KDE climatology for the forecast year |
| 4 | `3_combine_datasets.py` | Merge forecasts + climatology + ground truth into wide table |
| 5 | `0_connect_prepare_data_to_2025_pipeline.py` | Convert daily → weekly bins, compute rain predictors |
| 6 | `predict/apply_blend_model.py` | Apply trained blending model to produce predictions |
| 7 | `predict/export_blend_output.py` | Extract blended + climatology probabilities → summary CSV |
| 8 | `predict/run_maps.py` | Generate forecast maps from summary CSV |

### Prerequisites

Before running, ensure you have:
- Forecast NetCDF files for the new year (aifs and aifs_ens)
- Trained blending model coef pkl from `1_blend_evaluation.py` (historical training)
- Historical ground truth wide pkl (e.g. `imd_clim_mok_date_wide.pkl`) covering 2000–2022
- All 2026 yml specs created in `specs/raw_data/`, `specs/combine/`, `specs/2025_blend/`, and `specs/connect/`

### Minimal example

```bash
python predict/run_operational_pipeline.py \
    --year 2026 \
    --issue_date 2026-06-09 \
    --aifs_spec aifs_2026 \
    --aifs_ens_spec aifs_ens_2026 \
    --clim_spec imd_clim_mok_date_2026 \
    --combine_spec combine_template_clim_mok_date_2026 \
    --connect_spec connect_clim_mok_date_2026 \
    --blend_spec cv_models_clim_mok_date_2026 \
    --coef_dir Monsoon_Data/results/wet_spell_aifs_aifs_ens \
    --coef_tag clim_mok_date_2022_year2022 \
    --blend_input Monsoon_Data/Processed_Data/2026/cv_data_clim_mok_date_new_pipeline_2026.pkl \
    --work_dir Monsoon_Data/Processed_Data/2026
```

### With yml field overrides

Three frequently-changing inputs can be overridden directly from the command line without editing yml files. `--gt_path` controls the historical ground truth pkl and simultaneously patches both the clim spec (`input.gt_path`) and the combine spec (`ground_truth_wide_rds`) since both must point to the same file.

```bash
python predict/run_operational_pipeline.py \
    --year 2026 \
    --issue_date 2026-06-09 \
    --aifs_spec aifs_2026 \
    --aifs_ens_spec aifs_ens_2026 \
    --clim_spec imd_clim_mok_date_2026 \
    --combine_spec combine_template_clim_mok_date_2026 \
    --connect_spec connect_clim_mok_date_2026 \
    --blend_spec cv_models_clim_mok_date_2026 \
    --coef_dir Monsoon_Data/results/wet_spell_aifs_aifs_ens \
    --coef_tag clim_mok_date_2022_year2022 \
    --blend_input Monsoon_Data/Processed_Data/2026/cv_data_clim_mok_date_new_pipeline_2026.pkl \
    --work_dir Monsoon_Data/Processed_Data/2026 \
    --aifs_nc_folder /data/forecasts/aifs/2026 \
    --aifs_ens_nc_folder /data/forecasts/aifs_ens/2026 \
    --gt_path Monsoon_Data/Processed_Data/Models/wet_spell/imd_clim_mok_date_wide.pkl
```

When any of these overrides are provided, the script writes a patched temp spec (e.g. `aifs_2026_op.yml`) into the relevant `specs/` subdirectory, passes that to the downstream script, and deletes the temp file on exit.

### Resuming after a failure

Use `--skip_to N` to restart from a specific step without rerunning earlier (potentially expensive) steps:

```bash
# Re-run from step 6 onward (apply blend model through maps)
python predict/run_operational_pipeline.py \
    ... \
    --skip_to 6
```

### Dry run

Use `--dry_run` to print all commands that would be executed without running them — useful for verifying paths before committing to a full run:

```bash
python predict/run_operational_pipeline.py \
    ... \
    --dry_run
```

### All arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--year` | Yes | Forecast year, e.g. `2026` |
| `--issue_date` | Yes | Forecast issue date, e.g. `2026-06-09` |
| `--aifs_spec` | Yes | Spec ID for aifs `1_process_raw_nc_files`, e.g. `aifs_2026` |
| `--aifs_ens_spec` | Yes | Spec ID for aifs_ens `1_process_raw_nc_files`, e.g. `aifs_ens_2026` |
| `--clim_spec` | Yes | Spec ID for `2_build_climatology`, e.g. `imd_clim_mok_date_2026` |
| `--combine_spec` | Yes | Spec ID for `3_combine_datasets`, e.g. `combine_template_clim_mok_date_2026` |
| `--connect_spec` | Yes | Spec ID for `0_connect_prepare_data_to_2025_pipeline` |
| `--blend_spec` | Yes | Spec ID for `apply_blend_model` |
| `--coef_dir` | Yes | Directory containing the trained blending model coef pkl |
| `--coef_tag` | Yes | Coef tag passed to `apply_blend_model --coef_tag`, e.g. `clim_mok_date_2022_year2022` |
| `--blend_input` | Yes | Path to the wide pipeline pkl for `apply_blend_model --input_path` |
| `--work_dir` | Yes | Output directory for all intermediate and final files |
| `--aifs_nc_folder` | No | Override `input.nc_folder` in the aifs spec yml |
| `--aifs_ens_nc_folder` | No | Override `input.nc_folder` in the aifs_ens spec yml |
| `--gt_path` | No | Override ground truth pkl path in both the clim and combine specs |
| `--map_output_path` | No | Output directory for maps (default: `predict/output/{year}/`) |
| `--blend_model` | No | Blended model name (default: `blended_model`) |
| `--region` | No | Region for map generation (default: `Ethiopia`) |
| `--skip_to` | No | Skip to step N, 1-indexed (default: 1, run all) |
| `--dry_run` | No | Print commands without executing |

### Testing the Onset Detection Logic

```bash
python test_onset.py
```

Runs 13 test cases covering both dry-spell modes, prints a PASS/FAIL summary, and saves `test_onset_results.png` with annotated time-series plots (wet/dry periods highlighted, onset date marked).

---

## Data Flow

```
Raw NetCDF (Monsoon_Data/raw_nc/)
    │
    ▼
┌──────────────────────────────────┐
│  1_process_raw_nc_files.py       │  specs/raw_data/*.yml
│  → Processed_Data/Models/*.pkl   │
└──────────┬───────────────────────┘
           │
           ├──▶ 2_build_climatology.py
           │    → Processed_Data/Climatology/*.pkl
           │                 │
           ▼                 ▼
┌──────────────────────────────────┐
│  3_combine_datasets.py           │  specs/combine/*.yml
│  → Processed_Data/Combined/*.pkl │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  0_connect (day → week bins)             │  specs/2025_blend/connect_*.yml
│  → Processed_Data/2025_pipeline_input/   │
└──────────┬───────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  1_blend_evaluation.py                   │  specs/2025_blend/cv_models*.yml
│  Cross-validated multinomial logistic    │
│  Platt calibration, MME optimisation     │
│  → results/2025_model_evaluation/        │
└──────────┬───────────────────────────────┘
           │
           ├──▶ 2_2025_evaluation.py
           │    Out-of-sample scoring (Brier, RPS, AUC)
           ▼
┌──────────────────────────────────────────┐
│  3_produce_figures.py                    │
│  → figures/                              │
└──────────────────────────────────────────┘
```

---

## Onset Definition

The onset definition is fully configurable from the yml `options.onset_definition` block. Two dry-spell veto modes are supported.

### Trigger (both modes)

A candidate day `d` passes the trigger if:
- All days in the window `[d, d+win-1]` are wet: `rain >= wet_day_min_mm`
- The rolling sum over those `win` days exceeds `thresh` (per-cell value from `thresholds_df.csv`)

### Dry-spell veto modes

**`consecutive_dry`** (new definition): no run of `>= min_dry_days` consecutive dry days (`rain < dry_day_min_mm`) within `follow_days` days after the trigger window.

**`window_sum`** (original Moron-Robertson definition): no rolling window of `sum_window` days with total rainfall `< sum_min_mm` within `follow_days` days after the trigger window.

The first candidate day that passes both the trigger and the veto is returned as the onset date. If the first candidate is vetoed, the search continues to the next trigger candidate (option A behaviour). Setting `follow_days: 0` disables the veto entirely.

### yml configuration

```yaml
options:
  window: 3                        # trigger window length (days)
  onset_definition:
    wet_day_min_mm: 1.0            # minimum mm/day to count as wet
    follow_days: 21                # days after trigger window to check

    dry_spell:
      mode: consecutive_dry        # "consecutive_dry" | "window_sum"

      # --- consecutive_dry parameters ---
      min_dry_days: 7              # consecutive dry days needed to veto
      dry_day_min_mm: 1.0          # mm/day below this = dry day
                                   # (defaults to wet_day_min_mm if omitted)

      # --- window_sum parameters (original definition) ---
      # mode: window_sum
      # sum_window: 10             # rolling window length for dry-spell check
      # sum_min_mm: 5.0            # window sum below this = dry spell
```

To reproduce the **original Moron-Robertson definition** exactly:
```yaml
options:
  window: 5
  onset_definition:
    wet_day_min_mm: 1.0
    follow_days: 30
    dry_spell:
      mode: window_sum
      sum_window: 10
      sum_min_mm: 5.0
```

All parameters are optional — if `onset_definition` is omitted entirely, the `consecutive_dry` mode is used with `win=5`, `follow_days=21`, `min_dry_days=7`.

**Current default** (as set in all raw data specs): `follow_days: 0`, which disables the dry-spell veto entirely. Only the trigger condition is checked — the first day of a wet spell where all `window` days are wet and the accumulation exceeds the threshold is returned as onset. Set `follow_days` to a non-zero value (e.g. `21` for the Ethiopia/ICPAC definition) to re-enable the veto.

### Short series behaviour

If the series ends before the full follow-up window is available, the veto is checked over however many days remain. A candidate is only rejected if a dry spell is actually found in the available data. To reject any candidate whose follow-up window is incomplete, pass `reject_if_short_followup=True` in the `find_onset()` call.

---

## Onset Filter Variants

Three variants control which issue dates are included in training and evaluation:

| Variant | Spec suffix | Description |
|---------|-------------|-------------|
| **clim_mok_date** | `_clim_mok_date` | Only issue dates after a fixed climatological MOK date (June 1) |
| **mok** | _(default)_ | Only issue dates after the observed MOK date each year |
| **no_mok_filter** | `_no_mok_filter` | No MOK-based filtering; all issue dates from May 1 onward |

Each variant has its own connect and CV spec in `specs/2025_blend/`.

---

## Rain Predictors

Rain-based predictors are computed in `connect_utils.py` from the `rain_predictors` list under each model in `specs/2025_blend/connect_*.yml`. Two formats are supported:

**Legacy string format** (still supported for backward compatibility):
```yaml
rain_predictors: [diff_5day, min_10day, max_5day]
```

**Dict format** (preferred — generalises to any window size without code changes):
```yaml
rain_predictors:
  - { agg: diff, window: 3 }
  - { agg: min,  window: 7 }   # optional, commented out by default
  - { agg: max,  window: 3 }   # optional, commented out by default
```

The current default configuration uses only `diff` with a 3-day window, matching the 3-day trigger window:

```yaml
forecast_models:
  - name: ngcm
    variants: [clim_mok_date]
    rain_predictors:
      - { agg: diff, window: 3 }
  - name: aifs
    variants: [clim_mok_date]
    rain_predictors:
      - { agg: diff, window: 3 }
```

Three aggregation types are available:

| `agg` | Output column name | Description |
|-------|--------------------|-------------|
| `diff` | `diff_{model}_week{w}` | Max rolling sum over week minus per-cell onset threshold |
| `max`  | `max_{model}_{N}day_week{w}` | Max rolling N-day sum over week |
| `min`  | `min_{model}_{N}day_week{w}` | Min rolling N-day sum over week |

Output column names (e.g. `diff_ngcm_week1`, `min_ngcm_7day_week1`) are constructed at runtime from the model name and window size, and must match formula terms in `cv_models*.yml` using the `_qx` shorthand, which expands to `_week1` through `_week4` at runtime.

To add or remove a predictor: edit `rain_predictors` in the connect spec **and** add or remove the corresponding `_qx` term in the formula in `cv_models*.yml`. The two must stay in sync — a predictor present in the connect spec but absent from the formula is computed but silently unused; a formula term without a matching column will cause a runtime error.

---

## Spec-Driven Design

All pipeline behaviour is controlled by YAML specs. Spec files define input paths, variable selection, modelling options, and output configuration. Output basenames are derived from the `spec_id` (the yml filename), not from a field inside the YAML. Pipeline scripts are thin orchestration layers: parse args → load spec → call helpers → write artifacts.

### Spec directories

| Directory | Purpose | Used by |
|-----------|---------|---------|
| `specs/raw_data/` | NetCDF input config (paths, variables, thresholds, MOK, onset definition) | `1_process_raw_nc_files.py`, `2_build_climatology.py` |
| `specs/combine/` | Which processed datasets to merge into wide tables | `3_combine_datasets.py` |
| `specs/2025_blend/connect_*.yml` | Day-to-week conversion, forecast models, rain predictors | `0_connect_prepare_data_to_2025_pipeline.py` |
| `specs/2025_blend/cv_models*.yml` | Model formulas, MME config, forecast calibration | `1_blend_evaluation.py` |

### Key spec sections (`cv_models*.yml`)

- **`run.training_years` / `run.cv_holdout_years`**: Years used for cross-validated training and evaluation (currently 2019–2022).
- **`models.formulas`**: Named multinomial logistic regression formulas. Terms with `_qx` are expanded to `_week1`–`_week4` at runtime by `expand_formula_str()`. Current active formulas:
  - `ngcm_blend`: climatology × ngcm diff
  - `int_all`: climatology × ngcm diff × aifs diff (interaction)
  - `add_blend`: climatology + ngcm diff + aifs diff (additive)
  - `blended_model`: climatology × ngcm diff × aifs diff (currently same as `int_all`; `min` predictors commented out)
- **`models.window_variants`**: Training-window climatology variants — currently disabled (`enabled: false`).
- **`mme`**: Multi-model ensemble weight optimisation — currently disabled (`enabled: false`). When enabled, `blend_models` lists which calibrated models enter the MME.
- **`extras.forecasts`**: Per-system Platt calibration and raw/calibrated scoring options.
- **`extras.clim_logits`**: Climatology baseline configurations.

### Changing a model name

The model `name` field in `connect_*.yml` is the single source of truth for column naming. It propagates automatically into all output column names (`{name}_p_onset_week1`, `diff_{name}_week1`, etc.) and the wide pickle. If you rename a model, update:
1. `name` in `specs/2025_blend/connect_*.yml`
2. Formula terms in `specs/2025_blend/cv_models*.yml` (e.g. `diff_ngcm_qx` → `diff_newname_qx`)
3. Regenerate all downstream pickle files from stage 1 onward

---

## Key Python Functions

### `onset_utils.py`

| Function | Description |
|----------|-------------|
| `read_onset_params(spec)` | Parses `options.onset_definition` from spec; returns an `OnsetParams` namedtuple with all onset definition parameters and defaults |
| `find_onset(series, thresh, params)` | Finds first valid onset day in a rainfall series under the configured definition |
| `find_onset_precomp(series, win, thresh, ..., params)` | Batch-optimised version used by `calc_onsets_rowwise`; legacy positional args retained for call-site compatibility |
| `read_mok_dates(spec)` | Reads MOK dates CSV; returns DataFrame with `year`, `mok_date` |
| `read_thresholds(spec)` | Reads per-cell onset thresholds from CSV, NetCDF, or `.mat` |
| `roll_sum_na_rm_left(x, k)` | Left-aligned k-day rolling sum, NA treated as 0 |
| `roll_sum_na_propagate_left(x, k)` | Left-aligned k-day rolling sum, NA propagates |

### `nc_utils.py`

| Function | Description |
|----------|-------------|
| `run_single_pipeline(spec_id)` | Main driver: loads spec, processes all NetCDF years, writes output pickles |
| `calc_onsets_rowwise(df, day_cols, day_ints, win, params)` | Computes onset indices (raw, clim_mok_date, mok) for every row; passes `params` to `find_onset_precomp` |
| `process_rainfall_forecast_id(df, spec, ...)` | Forecast pipeline: reads onset params from spec, attaches thresholds and MOK dates, computes ensemble onset probabilities |
| `process_ground_truth_rainfall_id(df, spec, ...)` | Ground truth pipeline: computes true onset dates per cell-year |
| `attach_thresholds_id(df, thr_df)` | Left-joins per-cell `onset_thresh` onto the main DataFrame by `id` |

### `connect_utils.py`

| Function | Description |
|----------|-------------|
| `make_cv_rds_from_daylevel(spec)` | Main converter: reads daily combined pickle, builds weekly bins, onset outcomes, climatology logits, rain predictors, writes wide pickle for blending |
| `roll_sums_mat(mat, k)` | Rolling k-day row sums over a rainfall matrix |
| `week_max_over_starts(roll_mat, week_start_days)` | Per-row max of rolling sums at specified week-start positions |
| `week_min_over_starts(roll_mat, week_start_days)` | Per-row min of rolling sums at specified week-start positions |

---

## Conventions

- Spatial key: `id = f"{lat}_{lon}"`
- Time columns: `time` (date), `year` (int)
- Outcome categories: `week1` through `week4` plus `later` (five weekly bins)
- All intermediate data stored as pandas DataFrames serialised to `.pkl` (replacing `.rds` from the R version)
- Forecast probabilities stored in wide format with system-specific prefixes
- All scripts must be run from the repository root

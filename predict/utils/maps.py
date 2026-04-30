from datetime import timedelta
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import (
    BoundaryNorm,
    ListedColormap,
)
import geopandas as gpd
from pathlib import Path
import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s"
)

import xarray as xr

def load_boundary(base, use_cartopy=False, region='country',
                  resolution='10m', category='cultural',
                  name='admin_0_countries'):
    """Load country boundary geometry, returning a GeoDataFrame.

    Parameters
    ----------
    base : Path
        Root of the blend package (used to locate the local shapefile).
    use_cartopy : bool
        If True, download the boundary from Natural Earth via cartopy.
        If False (default), load the bundled local shapefile.
    region : str
        Country name to match when use_cartopy=True (e.g. 'India', 'Ethiopia').
    resolution : str
        Natural Earth resolution: '10m', '50m', or '110m'.
    category : str
        Natural Earth category (default 'cultural').
    name : str
        Natural Earth dataset name (default 'admin_0_countries').

    Returns
    -------
    geopandas.GeoDataFrame
        Country boundary in EPSG:4326.
    """
    if use_cartopy:
        import cartopy.io.shapereader as shpreader
        ne_path = shpreader.natural_earth(
            resolution=resolution, category=category, name=name
        )
        geom = None
        for country in shpreader.Reader(ne_path).records():
            if country.attributes['NAME'] == region:
                geom = country.geometry
                break
        if geom is None:
            raise ValueError(
                f"Region '{region}' not found in cartopy Natural Earth "
                f"({resolution}/{category}/{name}). "
                "Check the NAME attribute spelling."
            )
        gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
        logging.info(f"Boundary loaded from Natural Earth (cartopy): {region}")
    else:
        shp_path = base / "data" / "shapefile" / "Country_Boundary.shp"
        gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
        logging.info(f"Boundary loaded from local shapefile: {shp_path}")
    return gdf


def _period_labels(t):
    """Build legend label strings for each period, derived from forecast date t.

    Each week{i} covers days (i-1)*7+1 through i*7 from t:
      just_week1  : t+1  .. t+7
      weeks12     : t+1  .. t+14
      weeks23     : t+8  .. t+21
      weeks34     : t+15 .. t+28
      weeks4later : t+22 onwards
      later       : t+29 onwards

    Parameters
    ----------
    t : pd.Timestamp
        Forecast issue date (the 'time' value for this group).

    Returns
    -------
    dict mapping period key -> label string
    """
    fmt = '%m/%d/%Y'
    def date(offset): return (t + timedelta(days=offset)).strftime(fmt)

    return {
        'just_week1':  f"{date(1)} - {date(7)}",
        'weeks12':     f"{date(1)} - {date(14)}",
        'weeks23':     f"{date(8)} - {date(21)}",
        'weeks34':     f"{date(15)} - {date(28)}",
        'weeks4later': f"{date(22)}+",
        'later':       f"{date(29)}+",
    }


def _infer_resolution(cells_df):
    """Infer grid resolution (degrees) from the spacing of lat/lon values."""
    lat_diffs = pd.Series(sorted(cells_df['lat'].unique())).diff().dropna()
    lon_diffs = pd.Series(sorted(cells_df['lon'].unique())).diff().dropna()
    res = min(lat_diffs[lat_diffs > 0].min(), lon_diffs[lon_diffs > 0].min())
    return float(res)


def make_maps(summary, output_path, mok=False, all_cells_file=None,
              use_cartopy=False, region='country',
              resolution='10m', category='cultural',
              name='admin_0_countries', zoom_to_data=False):
    """Generate forecast maps from a blended summary DataFrame.

    Parameters
    ----------
    summary : pd.DataFrame
        Blended forecast output with columns: lat, lon, time,
        week1-week4, clim_week1-clim_week4, later (optional).
    output_path : Path
        Directory where output PNGs will be saved.
    mok : bool
        If True, save into maps_mok/ subfolder with _mok suffix.
    all_cells_file : Path or None
        Path to all_cells.csv. Defaults to data/support/all_cells.csv.
    use_cartopy : bool
        If True, load boundary via cartopy Natural Earth instead of
        the bundled local shapefile.
    region : str
        Country name for cartopy lookup (ignored when use_cartopy=False).
    resolution : str
        Natural Earth resolution for cartopy (default '10m').
    category : str
        Natural Earth category for cartopy (default 'cultural').
    name : str
        Natural Earth dataset name for cartopy (default 'admin_0_countries').
    zoom_to_data : bool
        If False (default), the map extent covers the full country boundary.
        If True, the extent is derived from the all_cells bounding box instead.
    """
    base = Path(__file__).resolve().parent.parent

    # Load country boundary (local shapefile or cartopy Natural Earth)
    country_gdf = load_boundary(
        base, use_cartopy=use_cartopy, region=region,
        resolution=resolution, category=category, name=name
    )

    if all_cells_file is None:
        all_cells_file = base / "data" / "support" / "all_cells.csv"
    all_cells = pd.read_csv(all_cells_file)

    exclude_cells_file = base / "data" / "support" / "exclude_cells.csv"
    df_exclude = pd.read_csv(exclude_cells_file, dtype={'lon': str, 'lat': str, 'flag': str})
    exclude_set = set(df_exclude.loc[df_exclude['flag'] == 'exclude', ['lon', 'lat']].itertuples(index=False, name=None))
    logging.info(f"Excluding {len(exclude_set)} cells from map generation.")

    first_date = pd.to_datetime(summary['time'].iloc[0])
    date_str_fmt = first_date.strftime("%Y%m%d")
    issue_date = first_date.date()

    # --------------------------------------------------------------------------
    # 0) Infer grid resolution from all_cells; derive half-width for rectangles
    # --------------------------------------------------------------------------
    res  = _infer_resolution(all_cells)   # e.g. 0.25
    half = res / 2.0                      # e.g. 0.125
    logging.info(f"Inferred grid resolution: {res}° → rectangle half-width: {half}°")

    # --------------------------------------------------------------------------
    # 1) Output folder
    # --------------------------------------------------------------------------
    output_dir = output_path / ("maps_mok" if mok else "maps")
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # 2) Prepare forecast data
    # --------------------------------------------------------------------------
    preds_df = summary.copy()
    preds_df = preds_df[~preds_df.set_index(['lon', 'lat']).index.isin(exclude_set)]

    for i in range(1, 5):
        preds_df[f'Climatology_p_{i}'] = preds_df[f'clim_week{i}']
        preds_df[f'Forecast_p_{i}']    = preds_df[f'week{i}']
    preds_df['Forecast_p_later']    = preds_df.get(
        'later', 1 - preds_df[[f'Forecast_p_{i}' for i in range(1, 5)]].sum(axis=1)
    )
    preds_df['Climatology_p_later'] = (
        1 - preds_df[[f'Climatology_p_{i}' for i in range(1, 5)]].sum(axis=1)
    )

    # Cell corners derived from inferred resolution
    preds_df['lon_min'] = preds_df['lon'] - half
    preds_df['lon_max'] = preds_df['lon'] + half
    preds_df['lat_min'] = preds_df['lat'] - half
    preds_df['lat_max'] = preds_df['lat'] + half

    # Map extent: full country boundary by default, or all_cells bbox if zoom_to_data=True
    if zoom_to_data:
        x_min = all_cells['lon'].min() - half
        x_max = all_cells['lon'].max() + half
        y_min = all_cells['lat'].min() - half
        y_max = all_cells['lat'].max() + half
        logging.info("Map extent: zoomed to all_cells bounding box")
    else:
        minx, miny, maxx, maxy = country_gdf.total_bounds
        x_min, x_max, y_min, y_max = minx, maxx, miny, maxy
        logging.info("Map extent: full country boundary")

    # Pre-build set of all dissemination cells for fast lookup
    all_cells_set = set(zip(all_cells['lat'], all_cells['lon']))

    # --------------------------------------------------------------------------
    # 3) Color schemes
    # --------------------------------------------------------------------------
    period_order  = ['just_week1', 'weeks12', 'weeks23', 'weeks34', 'weeks4later', 'later']
    plasma_cmap   = plt.get_cmap('plasma')
    stops         = np.linspace(0.2, 1.0, len(period_order))
    period_colors = {k: plasma_cmap(s) for k, s in zip(period_order, stops)}
    period_colors['none'] = '#d3d3d3'

    prob_bins = [0, 0.1, 0.2, 0.3, 0.4, 1.0]
    prob_cmap = ListedColormap(plt.get_cmap('plasma_r')(np.linspace(0, 1, len(prob_bins) - 1)))
    prob_norm = BoundaryNorm(prob_bins, ncolors=len(prob_bins) - 1, clip=True)

    # --------------------------------------------------------------------------
    # 4) Max-period helper
    # --------------------------------------------------------------------------
    def max_period(vf):
        if vf[0] >= 0.65: return 'just_week1'
        if vf[4] >= 0.65: return 'later'
        sums = [vf[0]+vf[1], vf[1]+vf[2], vf[2]+vf[3], vf[3]+vf[4]]
        return ['weeks12', 'weeks23', 'weeks34', 'weeks4later'][int(np.argmax(sums))]

    # --------------------------------------------------------------------------
    # SECTION: Weekly Probability Maps  (prob_weeks1-4_*)
    # --------------------------------------------------------------------------
    for t, grp in preds_df.groupby('time'):
        ds = t.strftime('%Y-%m-%d')
        week_titles = {
            i: f"{(t + timedelta(days=(i-1)*7+1)).strftime('%m/%d/%Y')} - "
               f"{(t + timedelta(days=i*7)).strftime('%m/%d/%Y')}"
            for i in range(1, 5)
        }
        # Index forecast rows by (lat, lon) for fast lookup
        grp_idx = grp.set_index(['lat', 'lon'])

        fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharex=True, sharey=True,
                                 gridspec_kw={'wspace': 0.03})
        for i, ax in enumerate(axes, 1):
            country_gdf.boundary.plot(ax=ax, linewidth=0.5, edgecolor='black')
            # Draw every dissemination cell: forecast colour or grey if missing
            for lat_c, lon_c in all_cells_set:
                lon0, lat0 = lon_c - half, lat_c - half
                if (lat_c, lon_c) in grp_idx.index:
                    r = grp_idx.loc[(lat_c, lon_c)]
                    v = r[f'Forecast_p_{i}'] if not isinstance(r, pd.DataFrame) else r[f'Forecast_p_{i}'].iloc[0]
                    if pd.isna(v):
                        fc = period_colors['none']
                    else:
                        fc = prob_cmap(prob_norm(v))
                else:
                    fc = period_colors['none']
                ax.add_patch(Rectangle((lon0, lat0), res, res,
                                       facecolor=fc, edgecolor='none', zorder=1))
            # Add climatology border only for cells within dissemination domain
            for _, r in grp.iterrows():
                if (r['lat'], r['lon']) not in all_cells_set: continue
                v    = r[f'Forecast_p_{i}']
                clim = r[f'Climatology_p_{i}']
                if pd.isna(v) or pd.isna(clim): continue
                if   v <= clim - 0.10: ec, lw = 'red',   0.8
                elif v >= clim + 0.10: ec, lw = 'green', 0.8
                else: continue
                ax.add_patch(Rectangle(
                    (r['lon_min'], r['lat_min']),
                    res, res,
                    fill=False, edgecolor=ec, linewidth=lw, zorder=2
                ))
            ax.set_title(week_titles[i], fontsize=10, pad=6)
            ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
            ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
            ax.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.5)

        sm = plt.cm.ScalarMappable(norm=prob_norm, cmap=prob_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=list(axes), orientation='horizontal', fraction=0.04, pad=0.08)
        cbar.set_label('Probability')
        legend_handles = [
            Patch(facecolor='none', edgecolor=c, linewidth=1.5, label=l)
            for c, l in [('red',   '≥10% lower than climatology'),
                         ('green', '≥10% higher than climatology')]
        ]
        axes[-1].legend(handles=legend_handles, loc='lower right',
                        fontsize=7, handlelength=1.5, handleheight=1.5,
                        borderpad=0.5, labelspacing=0.3, framealpha=0.8)
        fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.15)

        suffix = '_mok' if mok else ''
        fname = output_dir / f"prob_weeks1-4_{ds}{suffix}.png"
        plt.savefig(fname, dpi=150)
        logging.info(f"Saved weekly probability map: {fname}")

        # save map to netcdf
        # Define which columns represent the weekly probabilities
        prob_cols = [f'Forecast_p_{i}' for i in range(1, 5)] + ['Forecast_p_later']
        nc_probs_path = os.path.join(output_dir, f"weekly_probs_{date_str_fmt}{suffix}.nc")
        export_to_netcdf(grp_idx, prob_cols, nc_probs_path, issue_date)

        plt.close(fig)

    # --------------------------------------------------------------------------
    # SECTION: Max-Period Map  (map_max_period_*)
    # --------------------------------------------------------------------------
    for t, grp in preds_df.groupby('time'):
        ds = t.strftime('%Y-%m-%d')
        grp_idx = grp.set_index(['lat', 'lon'])

        labels = _period_labels(t)
        handles = [
            Patch(facecolor=period_colors[k], edgecolor='none', label=labels[k])
            for k in period_order
        ]
        handles.append(Patch(
            facecolor=period_colors['none'], edgecolor='none',
            label='No forecast / onset declared'
        ))

        fig, ax = plt.subplots(figsize=(6, 6))
        country_gdf.boundary.plot(ax=ax, linewidth=0.5, edgecolor='black')

        # Draw every dissemination cell
        for lat_c, lon_c in all_cells_set:
            lon0, lat0 = lon_c - half, lat_c - half
            if (lat_c, lon_c) in grp_idx.index:
                r  = grp_idx.loc[(lat_c, lon_c)]
                vf = [r[f'Forecast_p_{i}'] if not isinstance(r, pd.DataFrame)
                      else r[f'Forecast_p_{i}'].iloc[0] for i in range(1, 5)]
                vf_later = r['Forecast_p_later'] if not isinstance(r, pd.DataFrame) \
                           else r['Forecast_p_later'].iloc[0]
                vf = vf + [vf_later]
                if any(pd.isna(vf)):
                    fc = period_colors['none']
                else:
                    fc = period_colors[max_period(vf)]
            else:
                fc = period_colors['none']
            ax.add_patch(Rectangle((lon0, lat0), res, res,
                                   facecolor=fc, edgecolor='none', zorder=1))

        ax.legend(handles=handles, title='Period with Max Probability of Onset',
                  loc='lower left', ncol=2, fontsize=7, title_fontsize=7,
                  handlelength=1.5, handleheight=1.5,
                  borderpad=0.5, labelspacing=0.3, framealpha=0.8)
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.grid(True, linestyle='--', linewidth=0.4, color='gray', alpha=0.5)

        suffix = '_mok' if mok else ''
        fname = output_dir / f"map_max_period_{ds}{suffix}.png"
        plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close(fig)
        logging.info(f"Saved max-period map: {fname}")

        # 1. Calculate the max period index (1-5) for every row in the dataframe
        def identify_max_period(row):
            probs = [row[f'Forecast_p_{i}'] for i in range(1, 5)] + [row['Forecast_p_later']]
            if any(pd.isna(probs)):
                return np.nan
            return np.argmax(probs) + 1  # 1=W1, 2=W2, 3=W3, 4=W4, 5=Later
        
        # Add the column to grp_idx
        grp_idx['max_period_index'] = grp_idx.apply(identify_max_period, axis=1)
        
        # 2. Export to NetCDF
        nc_max_path = os.path.join(output_dir, f"max_period_index_{date_str_fmt}{suffix}.nc")
        export_to_netcdf(grp_idx, ['max_period_index'], nc_max_path, issue_date)

    logging.info(f"All maps saved under {output_dir}")


def export_to_netcdf(df, columns, output_path, issue_date):
    """
    Converts the grp_idx (which has lat/lon/time as index) to NetCDF.
    """
    # If it's grp_idx, the coordinates are in the index, so we reset them to columns
    temp_df = df.reset_index()

    # Ensure time is datetime
    temp_df['time'] = pd.to_datetime(temp_df['time'])

    # Pivot to a 3D grid (Time, Lat, Lon)
    ds = temp_df.set_index(['time', 'lat', 'lon'])[columns].to_xarray()

    # Add metadata
    ds.attrs['issue_date'] = str(issue_date)

    ds.to_netcdf(output_path)
    print(f"NetCDF saved: {output_path}")


def save_to_netcdf(df, columns, output_path):
    """Converts a flattened DataFrame to a gridded NetCDF file."""
    # Ensure time is a datetime object
    df['time'] = pd.to_datetime(df['time'])
    
    # Set multi-index for pivot/unstacking into a grid
    grid_df = df.set_index(['time', 'lat', 'lon'])[columns]
    
    # Convert to xarray Dataset
    ds = grid_df.to_xarray()
    
    # Save to disk
    ds.to_netcdf(output_path)
    print(f"Data saved to NetCDF: {output_path}")

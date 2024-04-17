import argparse
import pdb
from pathlib import Path

import fiona
import geopandas as gpd
import pandas as pd
from scipy import spatial
from scipy.interpolate import interp1d
import pickle as pk

from geo_utils import (check_if_transect_is_surrounded_by_shoreline_pts, compute_transect_points)


# parameters
# site = 'NARRABEEN'
site = 'DUCK'
use_tidal_correction_with_static_slope = True

# output directory
if use_tidal_correction_with_static_slope:
    output_dir = Path('/home/florent/Projects/benchmark_satellite_coastlines/deliverable/{site}/tidally_corrected_timeseries_MSL_static_slope/'.format(site=site))
else:
    output_dir = Path('/home/florent/Projects/benchmark_satellite_coastlines/deliverable/{site}/tidally_corrected_timeseries_MSL_dynamic_slope/'.format(site=site))
output_dir.mkdir(parents=True, exist_ok=True)

# read csv files of beach width
csv_timeseries_of_beach_width = Path('/home/florent/Projects/benchmark_satellite_coastlines/deliverable/{site}'.format(
    site=site))

# specify columns to read
if use_tidal_correction_with_static_slope:
    column_beach_width = 'beach_width_tidally_corrected_static_slope'
else:
    column_beach_width = 'beach_width_tidally_corrected_dynamic_slope'
columns = ['dates', column_beach_width, 'satname']

ls_timeseries_of_beach_width = csv_timeseries_of_beach_width.glob('*all.csv')
for f in sorted(ls_timeseries_of_beach_width):
    df = pd.read_csv(f, usecols=columns)
    transect_name = f.name.split('_')[0]

    # rename column 'beach width' to transect name
    df.rename(columns={column_beach_width: transect_name}, inplace=True)

    # save to csv
    df.to_csv(output_dir.joinpath(f'{transect_name}_timeseries_tidally_corrected.csv'),
              columns=['dates', transect_name, 'satname'],
              index=False)

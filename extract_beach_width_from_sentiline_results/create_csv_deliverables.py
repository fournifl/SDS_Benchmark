import argparse
import pdb
import numpy as np
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# parameters
# site = 'NARRABEEN'
# site = 'DUCK'
site = 'TRUCVERT'
# site = 'TORREYPINES'
use_tidal_correction_with_static_slope = False
if site == 'TRUCVERT':
    use_water_level_condition = True
    valid_interval_water_level = [0.2, np.inf]

# output directory
if use_tidal_correction_with_static_slope:
    output_dir = Path('/home/florent/Projects/benchmark_satellite_coastlines/deliverable/{site}/tidally_corrected_timeseries_MSL_static_slope/'.format(site=site))
else:
    output_dir = Path('/home/florent/Projects/benchmark_satellite_coastlines/deliverable/{site}/tidally_corrected_timeseries_MSL_dynamic_slope/'.format(site=site))
output_dir.mkdir(parents=True, exist_ok=True)

# csv path files of beach width
csv_timeseries_of_beach_width = Path('/home/florent/Projects/benchmark_satellite_coastlines/deliverable/{site}'.format(
    site=site))

# specify columns to read
if use_tidal_correction_with_static_slope:
    column_beach_width = 'beach_width_tidally_corrected_static_slope'
else:
    column_beach_width = 'beach_width_tidally_corrected_dynamic_slope'
    # column_beach_width = 'beach_width_tidally_corrected_dynamic_slope_polyfit'
columns = ['dates', column_beach_width, 'satname', 'tide']

# parse csv files of beach width
ls_timeseries_of_beach_width = csv_timeseries_of_beach_width.glob('*all.csv')
for f in sorted(ls_timeseries_of_beach_width):
    try:
        df = pd.read_csv(f, usecols=columns)
    except ValueError:
        print('Usecols do not match columns')
    else:
        transect_name = f.name.split('_')[0]

        # rename column 'beach width' to transect name
        df.rename(columns={column_beach_width: transect_name}, inplace=True)

        # keep only rows respecting a water level condition, if required
        if site == 'TRUCVERT' and use_water_level_condition:
            df = df[df['tide'] > valid_interval_water_level[0]]

        # save to csv
        df.to_csv(output_dir.joinpath(f'{transect_name}_timeseries_tidally_corrected.csv'),
                  columns=['dates', transect_name, 'satname'],
                  index=False)

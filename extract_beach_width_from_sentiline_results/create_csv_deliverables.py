import argparse
import pdb
import numpy as np
from pathlib import Path

import pandas as pd
from editing import filter_df_rolling_mean_std_two_pass, filter_df_rolling_mean_std, filter_df_rolling_mean_std_two_pass
import matplotlib.pyplot as plt


# parameters
# site = 'NARRABEEN'
# site = 'DUCK'
# site = 'TORREYPINES'
site = 'TRUCVERT'
# on_Sentinel2_data = False
on_Sentinel2_data = True

# editing parameters
editing = True
if site == 'NARRABEEN':
    limits = [10, 150]
elif site == 'DUCK':
    limits = [70, 180]
elif site == 'TRUCVERT':
    limits = None
elif site == 'TORREYPINES':
    limits = [50, 200]

# use_tidal_correction_with_static_slope = True
use_tidal_correction_with_static_slope = False
if site == 'TRUCVERT':
    use_water_level_condition = True
    valid_interval_water_level = [0.2, np.inf]

# output directory
if on_Sentinel2_data:
    output_subdir = 'Sentinel_2'
else:
    output_subdir = ''
if use_tidal_correction_with_static_slope:
    output_dir = Path(f'/home/florent/Projects/benchmark_satellite_coastlines/deliverable/{output_subdir}/{site}/tidally_corrected_timeseries_MSL_static_slope/')
else:
    output_dir = Path(f'/home/florent/Projects/benchmark_satellite_coastlines/deliverable/{output_subdir}/{site}/tidally_corrected_timeseries_MSL_dynamic_slope/')
output_dir.mkdir(parents=True, exist_ok=True)


# input csv path files of beach width
csv_timeseries_of_beach_width = Path(f'/home/florent/Projects/benchmark_satellite_coastlines/deliverable/{output_subdir}/{site}')

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

        # editing
        if editing:
            if on_Sentinel2_data:
                window = 15
                n1 = 2.6
                n2 = 2.6
            else:
                window = 10
                n1 = 2.6
                n2 = 2.6

            df['dates'] = pd.to_datetime(df['dates'])
            filter_df_rolling_mean_std_two_pass(df, column=transect_name, limits=limits, window=window, n1=n1, n2=n2)

            fig, ax = plt.subplots(figsize=(20, 6))
            ax.plot(df['dates'], df[f'{transect_name}_high'], color='lightgray')
            ax.plot(df['dates'], df[f'{transect_name}_low'], color='lightgray')
            ax.plot(df['dates'], df[transect_name], 'r')
            ax.plot(df['dates'][df[f'{transect_name}2_valid']], df[transect_name][df[f'{transect_name}2_valid']], 'g')
            ax.set_title(f'Editing of transect {transect_name}, window {window}, n1={n1} n2={n2}')
            ax.grid(True)
            fig.savefig(output_dir / f'editing_{transect_name}.png', bbox_inches='tight')

            # apply editing
            df = df[df[f'{transect_name}2_valid']]

        # save to csv
        df.to_csv(output_dir.joinpath(f'{transect_name}_timeseries_tidally_corrected.csv'),
                  columns=['dates', transect_name, 'satname'],
                  index=False)

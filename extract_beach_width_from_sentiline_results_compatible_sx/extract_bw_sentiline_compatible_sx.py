import argparse
from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd
from scipy import spatial
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from geo_utils import (check_if_transect_is_surrounded_by_shoreline_pts, compute_transect_points)
from editing import filter_df_rolling_mean_std_two_pass


# construct an argument parser
parser = argparse.ArgumentParser()

# add argument to the parser
parser.add_argument('config')

# get arguments
args = vars(parser.parse_args())
config_file = args['config']
settings = pd.read_json(config_file, orient='index').to_dict()[0]

# site
site = settings['site']

# list waterlines sentiline
mission = settings['mission']
if mission == 'Sentinel_2':
    glob_pattern = mission
elif mission == 'Landsat':
    glob_pattern = mission + '*'
base = Path(settings['dir_wl'].format(site=settings['site'],
                                               mission=glob_pattern))
if mission == 'Sentinel_2':
    f_wl = list(base.glob('shoreline_results.parquet'))[0]
elif mission == 'Landsat':
    ls_wl = list(base.glob('shoreline_results.parquet'))

gdf_sl = gpd.read_parquet(f_wl)

# output dir
output_dir = Path(settings['output_dir'].format(site=settings['site'], mission=settings['mission']))
output_dir.mkdir(parents=True, exist_ok=True)

# transects
transects_file = settings['transects'].format(site=settings['site'])
transects = gpd.read_file(transects_file)

# to just do calculations on some transects used for validation
if settings['site'] == "TORREYPINES":
    transects = transects.loc[transects['name'].isin(['PF525', 'PF535', 'PF585', 'PF595'])]
elif settings['site'] == "NARRABEEN":
    transects = transects.loc[transects['name'].isin(['PF1', 'PF2', 'PF4', 'PF6', 'PF8'])]
elif settings['site'] == "DUCK":
    transects = transects.loc[transects['name'].isin(['-91', '1', '1006', '1097'])]
elif settings['site'] == "TRUCVERT":
    transects = transects.loc[transects['name'].isin(['-400', '-300', '-200', '-100'])]
transects = transects.reset_index(drop=True)

# convert transects to epsg of study site
transects = transects.to_crs(int(settings['epsg_site']))

# compute transects arrays
tr_l = {}
tr_pts = {}
tr_coords = {}
tr_csd = {}
for i, transect in enumerate(transects.geometry):
    tr_name = transects.iloc[i]['name']
    tr_l[tr_name], tr_pts[tr_name], tr_coords[tr_name], tr_csd[tr_name] = compute_transect_points(transect)

# water level
tide = pd.read_csv(settings['tide_file'].format(site=settings['site']), usecols=['dates', 'tides'])
tide['dates'] = pd.to_datetime(tide['dates'])
tide['julian_dates'] = [t.to_julian_date() for t in tide['dates']]


# compute beach width at each transect, for each date
data = {'datetime_utc': [], 'beach_width_m': [], 'transect_id': [], 'mission': []}
for i in range(len(gdf_sl)):
    gdf = gdf_sl.iloc[i]

    for i_tr in range(len(transects)):
        tr = transects.iloc[i_tr]['name']

        # parse shorelines at every date and get the corresponding beach width at transect
        for i, line in enumerate(gdf.geometry.geoms):
            if 'valid' in gdf.keys():
                if gdf['valid'][i]:
                    valid_condition = True
                else:
                    valid_condition = False
            else:
                valid_condition = True

            if valid_condition:
                if line is not None:

                    # compute intersection between shoreline and transect
                    intersect = line.intersection(tr_l[tr])

                    # if intersection, compute beach width
                    if hasattr(intersect, 'x'):
                        sl_points_near_transect = check_if_transect_is_surrounded_by_shoreline_pts(
                            line, tr_l[tr], intersect,
                            circle_radius=settings[
                                'd_threshold_transect_pt_intersect_with_sl_points_each_side_of_transect'])

                        # get beach width if intersection point is surrounded by shoreline points on both sides of transect
                        if sl_points_near_transect:
                            indice_pt_transect_intersection_with_shoreline = spatial.KDTree(tr_coords[tr]).query(
                                [intersect.x, intersect.y])[1]
                            pt_transect_intersection_with_shoreline = tr_pts[tr][
                                indice_pt_transect_intersection_with_shoreline]
                            cross_shore_d = tr_csd[tr][indice_pt_transect_intersection_with_shoreline]

                            # fill in a new element in dictionnary of extracted beach width
                            data['datetime_utc'].append(gdf['date'])
                            data['beach_width_m'].append(cross_shore_d)
                            data['transect_id'].append(i_tr)
                            data['mission'].append(mission)

# store data extraction in a dataframe
df = pd.DataFrame.from_dict(data)

# get tide for every beach width
tmin = np.array(data['datetime_utc']).min()
tmax = np.array(data['datetime_utc']).max()
tide = tide[(tide['dates'] >= (tmin - pd.Timedelta(days=1)).to_datetime64()) * (tide['dates'] <= (tmax + pd.Timedelta(days=1)).to_datetime64())]
tide = tide.reset_index(drop=True)
function_interpolation = interp1d(tide['julian_dates'], tide['tides'], bounds_error=False)
df['julian_dates'] = [t.to_julian_date() for t in df['datetime_utc']]
df['tide_z'] = function_interpolation(df['julian_dates'])
df['tide_z'] = df['tide_z'] * 100 # convert in cm

# editing
if site == 'NARRABEEN':
    limits = [10, 150]
elif site == 'DUCK':
    limits = [70, 180]
elif site == 'TRUCVERT':
    limits = None
elif site == 'TORREYPINES':
    limits = [50, 200]
window = 40
n1 = 2.4
n2 = 2.4

# init valid to False
df['valid'] = False

for i in range(len(transects)):
    df_tr = df[df['transect_id'] == i]
    filter_df_rolling_mean_std_two_pass(df_tr, column='beach_width_m', limits=limits, window=window, n1=n1, n2=n2)
    valid_mask = df_tr['beach_width_m2_valid']
    plt.plot(df_tr['datetime_utc'], df_tr['beach_width_m'])
    plt.plot(df_tr.loc[valid_mask]['datetime_utc'], df_tr.loc[valid_mask]['beach_width_m'], color='r')
    plt.plot(df_tr.loc[valid_mask]['datetime_utc'], df_tr.loc[valid_mask]['beach_width_m2_high'], color='k')
    plt.plot(df_tr.loc[valid_mask]['datetime_utc'], df_tr.loc[valid_mask]['beach_width_m2_low'], color='k')
    plt.show()
    df.loc[df['transect_id'] == i, 'valid'] = df_tr['beach_width_m2_valid']

df = df[df['valid']]
df.to_parquet(output_dir / 'intersections_sentiline.parquet')
print('')
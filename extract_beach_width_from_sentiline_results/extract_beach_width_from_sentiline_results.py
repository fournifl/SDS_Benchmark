import argparse
import pdb
from pathlib import Path

import fiona
import geopandas as gpd
import pandas as pd
from scipy import spatial

from geo_utils import (check_if_transect_is_surrounded_by_shoreline_pts, compute_transect_points)


def read_gdpk_multilayer(file):
    tmp_list = []
    for layername in fiona.listlayers(file):
        tmp_list.append(gpd.read_file(file, layer=layername))
    gdf = gpd.GeoDataFrame(pd.concat(tmp_list, ignore_index=True))
    return gdf


# construct an argument parser
parser = argparse.ArgumentParser()

# add argument to the parser
parser.add_argument('config')

# get arguments
args = vars(parser.parse_args())
config_file = args['config']
settings = pd.read_json(config_file, orient='index').to_dict()[0]

# execution options
apply_tide_correction = settings['apply_tide_correction']
apply_tide_correction_and_wave_setup_correction = settings['apply_tide_correction_and_wave_setup_correction']

# sentiline results dir
sentiline_results_dir = Path(settings['sentiline_results_dir'].format(working_dir=settings['working_dir'],
                                                                      site=settings['site']))
# output dir
output_dir = Path(settings['output_dir'].format(site=settings['site']))
output_dir.mkdir(parents=True, exist_ok=True)

# transects
transects_file = settings['transects'].format(site=settings['site'])
transects = gpd.read_file(transects_file)

# convert transects to epsg of study site
transects = transects.to_crs(int(settings['epsg_transects']))

# water level
if apply_tide_correction:
    tide = pd.read_csv(settings['tide_file'].format(site=settings['site']))


# satellites' abbreviations
sat_abbreviations = {'Landsat_5': 'L5', 'Landsat_7': 'L7', 'Landsat_8': 'L8', 'Landsat_9': 'L9', 'Sentinel_2': 'S2'}

# initialize dictionnary of extracted beach width
data = {'dates': [], 'beach_width': [], 'satname':[]}

# compute beach width at each transect, for each sat
for ind_transect, transect in enumerate(transects.geometry):

    # compute transect
    transect_line, transect_points, transect_coords, transect_cross_shore_d = compute_transect_points(transect)

    for sat in settings['satellites']:
        # shoreline of a given sat
        shoreline_results_file = settings['shoreline_results_file'].format(sentiline_results_dir=sentiline_results_dir,
                                                                           sat=sat)
        shoreline = read_gdpk_multilayer(shoreline_results_file)

        # convert shoreline to epsg of study site
        shoreline = shoreline.to_crs(int(settings['epsg_transects']))

        # parse shorelines at every date
        for i, shorelines in enumerate(shoreline.geometry):
            print(transects.name[ind_transect], sat, shoreline['date'][i])
            # if shorleine['valid'][i]:

            if shorelines is not None:
                # parse every shoreline at a given date
                for line in shorelines.geoms:

                    # compute intersection between shoreline and transect
                    intersect = line.intersection(transect_line)

                    # if intersection, compute beach width
                    if hasattr(intersect, 'x'):
                        sl_points_near_transect = check_if_transect_is_surrounded_by_shoreline_pts(
                            line, transect_line, intersect,
                            circle_radius=settings['d_threshold_transect_pt_intersect_with_sl_points_each_side_of_transect'])

                        # get beach width if intersection point is surrounded by shoreline points on both sides of transect
                        if sl_points_near_transect:
                            indice_pt_transect_intersection_with_shoreline = spatial.KDTree(transect_coords).query(
                                [intersect.x, intersect.y])[1]
                            pt_transect_intersection_with_shoreline = transect_points[indice_pt_transect_intersection_with_shoreline]
                            cross_shore_d = transect_cross_shore_d[indice_pt_transect_intersection_with_shoreline]

                            # tidal correction
                            if apply_tide_correction:
                                pdb.set_trace()
                                # delta_cross_sh_d_from_wl = (sl['water_level'][i] - attrs['msl_ref_ign69']) \
                                #                            / np.tan(beach_slope)
                                # sl_cross_sh_d[stack].append(np.around(transect_cross_sh_d[stack][id_p], decimals=2) +
                                #                             delta_cross_sh_d_from_wl)

                            # fill in a new element in dictionnary of extracted beach width
                            data['dates'].append(shoreline['date'][i])
                            data['beach_width'].append(cross_shore_d)
                            data['satname'].append(sat_abbreviations[sat])

    # store data extraction in a dataframe
    df = pd.DataFrame.from_dict(data)

    # rename column 'beach width' to transect name
    transect_name = transects.name[ind_transect]
    df.rename(columns={'beach_width': transect_name}, inplace=True)

    # sort df by date
    df = df.set_index(df['dates'])
    df = df.sort_index()

    # save to csv
    df.to_csv(output_dir.joinpath(f'{transect_name}_timeseries_raw.csv'), index=False)

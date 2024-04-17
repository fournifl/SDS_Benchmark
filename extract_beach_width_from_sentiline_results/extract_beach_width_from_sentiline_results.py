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
    tide = pd.read_csv(settings['tide_file'].format(site=settings['site']), usecols=['dates', 'tides'])
    tide['dates'] = pd.to_datetime(tide['dates'])
    tide['julian_dates'] = [t.to_julian_date() for t in tide['dates']]

# beach slope
beach_slope_file = settings['groundtruth_slope_file'].format(site=settings['site'])
beach_slope = pk.load(open(beach_slope_file, 'rb'))

# satellites' abbreviations
sat_abbreviations = {'Landsat_5': 'L5', 'Landsat_7': 'L7', 'Landsat_8': 'L8', 'Landsat_9': 'L9', 'Sentinel_2': 'S2'}


# compute beach width at each transect, for each sat
for ind_transect, transect in enumerate(transects.geometry):

    # initialize dictionnary of extracted beach width
    data = {'dates': [], 'beach_width': [], 'satname': []}

    # compute transect
    transect_line, transect_points, transect_coords, transect_cross_shore_d = compute_transect_points(transect)

    for sat in (settings['satellites']):
        # shoreline of a given sat
        shoreline_results_file = settings['shoreline_results_file'].format(sentiline_results_dir=sentiline_results_dir,
                                                                           sat=sat)
        shoreline = read_gdpk_multilayer(shoreline_results_file)

        # convert shoreline to epsg of study site
        shoreline = shoreline.to_crs(int(settings['epsg_transects']))

        # parse shorelines at every date and get the corresponding beach width at transect
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

                            # fill in a new element in dictionnary of extracted beach width
                            data['dates'].append(shoreline['date'][i])
                            data['beach_width'].append(cross_shore_d)
                            data['satname'].append(sat_abbreviations[sat])

    # store data extraction in a dataframe
    df = pd.DataFrame.from_dict(data)

    # tidal correction on beach width
    if apply_tide_correction:

        # get water level from fes, corresponding to the considered shoreline date, doing an interpolation
        function_interpolation = interp1d(tide['julian_dates'], tide['tides'], bounds_error=False)
        df['julian_dates'] = [t.to_julian_date() for t in df['dates']]
        df['tide'] = function_interpolation(df['julian_dates'])

        # get beach slope from observations, corresponding to the considered shoreline date, doing an interpolation
        if transects.name[ind_transect] in beach_slope.keys():
            beach_slope_dates = beach_slope[transects.name[ind_transect]]['dates']
            beach_slope_julian_dates = [pd.Timestamp(t).to_julian_date() for t in beach_slope_dates]
            beach_slope_at_transect = beach_slope[transects.name[ind_transect]]['slopes']
            if len(beach_slope_julian_dates) == len(beach_slope_at_transect):
                function_interpolation = interp1d(beach_slope_julian_dates, beach_slope_at_transect, bounds_error=False)
                df['beach_slope'] = function_interpolation(df['julian_dates'])

                # express water level relative to msl
                df['tide'] = df['tide'] + settings['msl_relative_to_survey_datum']

                # apply horizontal correction on beach width, with a dynamic beach slope
                delta_x = df['tide'] / df['beach_slope']
                df['beach_width_tidally_corrected_dynamic_slope'] = df['beach_width'] + delta_x

        # apply horizontal correction on beach width, with a static beach slope
        delta_x = df['tide'] / settings['average_beach_slope_tan_beta']
        df['beach_width_tidally_corrected_static_slope'] = df['beach_width'] + delta_x

        # rename column 'beach width' to transect name
        transect_name = transects.name[ind_transect]
        # df.rename(columns={'beach_width': transect_name}, inplace=True)

        # sort df by date
        df = df.set_index(df['dates'])
        df = df.sort_index()

        # save to csv
        # df.to_csv(output_dir.joinpath(f'{transect_name}_timeseries_raw.csv'), index=False)
        df.to_csv(output_dir.joinpath(f'{transect_name}_timeseries_all.csv'), index=False)

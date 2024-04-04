import pdb
from pathlib import Path
import argparse
import pandas as pd
import geopandas as gpd
import fiona
from shapely import Point, plotting
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt


def compute_transect_points(pt_A, pt_B, d):
    dy = pt_B[1] - pt_A[1]
    dx = pt_B[0] - pt_A[0]
    teta = direction(float(dx), float(dy))

    transect_x_all = [pt_A[0]]
    transect_y_all = [pt_A[1]]
    transect_lists = [[pt_A[0], pt_A[1]]]
    transect_tuples = [(pt_A[0], pt_A[1])]

    cross_shore_d = [0]

    while Point(transect_tuples[-1][0], transect_tuples[-1][1]).distance(Point(pt_B[0], pt_B[1])) > 2 * d:
        transect_x_all.append(transect_x_all[-1] + d * np.cos(teta))
        transect_y_all.append(transect_y_all[-1] + d * np.sin(teta))
        transect_x = transect_tuples[-1][0] + d * np.cos(teta)
        transect_y = transect_tuples[-1][1] + d * np.sin(teta)
        transect_lists.append([transect_x, transect_y])
        transect_tuples.append((transect_x, transect_y))
        cross_shore_d.append(cross_shore_d[-1] + d)
    return transect_tuples, transect_lists, np.array(transect_x_all), np.array(transect_y_all), \
           np.array(cross_shore_d)


def read_gdpk_multilayer(file):
    tmp_list = []
    for layername in fiona.listlayers(file)[0:3]:
        print(layername)
        tmp_list.append(gpd.read_file(file, layer=layername))
    gdf = gpd.GeoDataFrame(pd.concat(tmp_list, ignore_index=True))
    return gdf


def determine_if_two_near_points_on_either_side_of_transect(vertices, transect_line):
    boundary = transect_line.boundary
    x1 = boundary.geoms[0].x
    y1 = boundary.geoms[0].y
    x2 = boundary.geoms[1].x
    y2 = boundary.geoms[1].y

    side_values = (vertices.x - x1) * (y2 - y1) - (vertices.y - y1) * (x2 - x1)
    if len(np.unique(np.sign(side_values))) > 1:
        return True
    else:
        return False


def check_if_transect_is_surrounded_by_shoreline_pts(line, transect, intersect, circle_radius=None):
    # intersection point
    pt_intersect = Point(intersect.x, intersect.y)
    circle = pt_intersect.buffer(circle_radius)

    # f, ax = plt.subplots()
    # ax.plot(line.xy[0], line.xy[1], '.b')
    # ax.plot(transect.xy[0], transect.xy[1], 'r')
    # plotting.plot_polygon(circle, ax=ax)
    # ax.plot(pt_intersect.x, pt_intersect.y, '+g')
    # plt.show()

    # Creating GeoDataFrame circle
    circle = gpd.GeoDataFrame({'geometry': [circle]})

    # Create lists with X and Y coordinates from LineString
    x = [i[0] for i in line.coords]
    y = [i[1] for i in line.coords]

    # Creating GeoDataFrame shoreline
    shoreline = gpd.GeoDataFrame(geometry=gpd.points_from_xy(x, y))

    # vertices inside circle
    vertices_in = shoreline[shoreline.within(circle.loc[0, 'geometry'])]

    if len(vertices_in) > 0:
        # check if near shoreline points are on each side of transect
        vertices_in = vertices_in.geometry
        near_points_either_side = determine_if_two_near_points_on_either_side_of_transect(vertices_in, transect)

        if near_points_either_side:
            return True
        else:
            return False
    else:
        return False


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
apply_wave_setup_correction = settings['apply_wave_setup_correction']

# sentiline results dir
sentiline_results_dir = Path(settings['sentiline_results_dir'].format(working_dir=settings['working_dir'],
                                                                      site=settings['site']))
# output dir
output_dir = Path(settings['output_dir'])
output_dir.mkdir(parents=True, exist_ok=True)

# transects
transects_file = settings['transects'].format(site=settings['site'])
transects = gpd.read_file(transects_file)

# convert transects to epsg of study site
transects = transects.to_crs(int(settings['epsg_transects']))

# compute beach width for each sat, at each transect
for sat in settings['satellites']:
    print(sat)
    # shorelines
    shoreline_results_file = settings['shoreline_results_file'].format(sentiline_results_dir=sentiline_results_dir,
                                                                       sat=sat)
    print(shoreline_results_file)
    shoreline = read_gdpk_multilayer(shoreline_results_file)

    # convert shorelines to epsg of study site
    shoreline = shoreline.to_crs(int(settings['epsg_transects']))


    for transect in transects.geometry:
        for i, shorelines in enumerate(shoreline.geometry):
            # if shorleine['valid'][i]:
            for line in shorelines.geoms:
                intersect = line.intersection(transect)
                if hasattr(intersect, 'x'):
                    sl_points_near_transect = check_if_transect_is_surrounded_by_shoreline_pts(line, transect,
                    intersect, circle_radius=settings['d_threshold_transect_pt_intersect_with_sl_points_each_side_of_transect'])

                    if sl_points_near_transect:
                        pdb.set_trace()
                        # test = transect['xy'][stack][
                        #     spatial.KDTree(transect['xy'][stack]).query([intersect.x, intersect.y])[1]]
                        test = transect['xy'][stack][
                            spatial.KDTree(transect['xy'][stack]).query([intersect.x, intersect.y])[1]]






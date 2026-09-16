import pdb

import numpy as np
import geopandas as gpd
from shapely import Point, LineString
import matplotlib.pyplot as plt
from shapely import plotting


def compute_transect_points(transect, d=0.2):
    boundary = transect.boundary
    pt_A = boundary.geoms[0]
    pt_B = boundary.geoms[1]

    dy = pt_B.y - pt_A.y
    dx = pt_B.x - pt_A.x
    teta = direction(float(dx), float(dy))

    list_transect_points = [pt_A]
    transect_cross_shore_d = [0]

    while list_transect_points[-1].distance(pt_B) > 2 * d:
        pt = list_transect_points[-1]
        x = pt.x + d * np.cos(teta)
        y = pt.y + d * np.sin(teta)
        list_transect_points.append(Point(x, y))
        transect_cross_shore_d.append(transect_cross_shore_d[-1] +  d)
    transect_line = LineString(list_transect_points)
    transect_coords = [[pt.x, pt.y] for pt in list_transect_points]
    return transect_line, list_transect_points, transect_coords, transect_cross_shore_d


def direction(dx, dy):
    """
    Computation of the direction between 2 points A and B, defined in world coordinates (x,y), with xB = xA + dx,
    yB = yA + dy

    Parameters
    ----------
    dx: float
    dy: float

    Returns
    -------
    teta: float
    """
    # print('dx: %s,  dy:%s' %(dx, dy))
    if (dx > 0):
        teta = np.arctan(dy / dx)
    elif (dx < 0) * (dy >= 0):
        teta = np.arctan(dy / dx) + np.pi
    elif (dx < 0) * (dy <= 0):
        teta = np.arctan(dy / dx) + np.pi
    elif (dx == 0) * (dy < 0):
        teta = -np.pi / 2
    elif (dx == 0) * (dy > 0):
        teta = np.pi / 2
    # print('teta: %s' %teta)
    return teta


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
    # ax.plot(transect.xy[0], transect.xy[1], '+-r')
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

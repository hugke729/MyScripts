# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 11:57:14 2015
http://www.geophysique.be/2011/02/19/matplotlib-basemap-tutorial-08-shooting-great-circles/
@author: Ken
"""
import numpy as np
from numpy import pi
import math
from math import atan2, sin, cos, radians, asin, sqrt


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    Distances in km

    https://gist.github.com/geografa/1366401 and somewhere else
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    dlon = lon2-lon1
    bearing = 180/pi*atan2(cos(lat1)*sin(lat2)-sin(lat1)*cos(lat2)*cos(dlon),
                           sin(dlon)*cos(lat2))

    b = atan2(sin(dlon)*cos(lat2),
              cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(dlon))
    bd = math.degrees(b)
    br, bearing = divmod(bd+360,360)

    dist = c*r
    return dist, bearing


def haversines(lon_vec, lat_vec):
    """Takes lon and lat vectors of length n and returns distance (km) vector of
    length n starting with a zero, and bearing vector (degrees) of length n"""
    dx_vec = np.zeros(len(lon_vec)-1)
    bearing_vec = np.zeros(len(lon_vec)-1)

    for i in range(len(lon_vec)-1):
        dx_vec[i], bearing_vec[i] = haversine(lon_vec[i], lat_vec[i], lon_vec[i+1], lat_vec[i+1])

    dist_vec = np.concatenate((np.array([0.0]), np.cumsum(dx_vec)))
    # make bearing vector have N elements
    bearing_vec = np.append(bearing_vec, np.array(bearing_vec[-1]))
    return dist_vec, bearing_vec


# Pretty sure this is never used
# def compass_bearing(pointA, pointB):
#     """
#     Calculates the bearing between two points.

#     The formulae used is the following:
#         θ = atan2(sin(Δlong).cos(lat2),
#                   cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))

#     :Parameters:
#       - `pointA: The tuple representing the latitude/longitude for the
#         first point. Latitude and longitude must be in decimal degrees
#       - `pointB: The tuple representing the latitude/longitude for the
#         second point. Latitude and longitude must be in decimal degrees

#     :Returns:
#       The bearing in degrees

#     :Returns Type:
#       float
#     """
#     if (type(pointA) != tuple) or (type(pointB) != tuple):
#         raise TypeError("Only tuples are supported as arguments")

#     lat1 = math.radians(pointA[0])
#     lat2 = math.radians(pointB[0])

#     diffLong = math.radians(pointB[1] - pointA[1])

#     x = sin(diffLong) * cos(lat2)
#     y = cos(lat1) * sin(lat2) - (sin(lat1)
#             * cos(lat2) * cos(diffLong))

#     initial_bearing = atan2(x, y)

#     # Now we have the initial bearing but math.atan2 return values
#     # from -180° to + 180° which is not what we want for a compass bearing
#     # The solution is to normalize the initial bearing as shown below
#     initial_bearing = math.degrees(initial_bearing)
#     compass_bearing = (initial_bearing + 360) % 360

#     return compass_bearing

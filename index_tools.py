# -*- coding: utf-8 -*-
"""
Created on Fri Oct 05 13:32:06 2012

@author: Garrett Berg
"""
import numpy as np
import itertools
import math


def index_to_coords(index, shape):
    '''convert index to coordinates given the shape'''
    coords = []
    for i in range(1, len(shape)):
        divisor = int(np.product(shape[i:]))
        value = index//divisor
        coords.append(value)
        index -= value * divisor
    coords.append(index)
    return tuple(coords)


def first_coords_et(data_matrix, value, start=0):
    '''the first coordinates that are equal to the value'''
    index = first_index_et(data_matrix.flatten(), value, start)
    shape = data_matrix.shape
    return index_to_coords(index, shape)


def first_coords_et_3d(data_matrix, value, start=0, axis=0):
    """the first coords in each column"""
    ni, nj = 0, 2
    ind_list = []
    for i in range(data_matrix.shape[ni]):
        for j in range(data_matrix.shape[nj]):
            new_ind = first_coords_et(data_matrix[i, :, j], value)
            ind_list = ind_list + [(i, new_ind[0], j)]

    return ind_list


def first_index_et(data_list, value, start=0):
    data_list = itertools.islice(data_list, start, None)
    '''same as data_list.index(value), except with exception handling
    Also finds 'nan' values and works with numpy arrays -- quickly!'''
    try:
        if type(value) == float and math.isnan(value):
            floats = (float, np.float64, np.float32, np.float96)
            isnan = math.isnan
            return next(data[0] for data in enumerate(data_list)
              if (type(data[1]) in floats
              and isnan(data[1]))) + start
        else:
            return next(data[0] for data in
            enumerate(data_list) if data[1] == value) + start
    except (ValueError, StopIteration):
        return - 1

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 14:57:23 2015

@author: Ken
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def splay_figures():
    """Get all figures and spread them across my secondary monitor"""
    fig_list = plt.get_fignums()
    wx = 640
    h = 500
    x1, x2, x3 = 1367, 1367 + wx, 1367 + wx*2
    y0 = 30
    y1 = 570
    points = np.array([[x1,y0,wx,h],
                       [x2,y0,wx,h],
                       [x3,y0,wx,h],
                       [x1,y1,wx,h],
                       [x2,y1,wx,h],
                       [x3,y1,wx,h]])
    for i in range(len(fig_list)):
        plt.figure(fig_list[i])
        plt.get_current_fig_manager().window.setGeometry(
            points[i,0],points[i,1], points[i,2], points[i,3])


def raster_and_save(fname, rasterize_list=None, fig=None, dpi=None,
                    savefig_kw={}):
    """Like plt.savefig, but rasterizes anything in rasterize_list

    Pass additional keywords to plt.savefig with savefig_kw

    Note: does not work correctly with round=True in Basemap"""
    # My trial and error showed that need to set_rasterization_zorder
    # in order for rasterizing to work properly

    fig = plt.gcf() if fig is None else fig

    zorder = -5  # Somewhat arbitrary, just ensuring less than 0

    if rasterize_list is None:
        # Have a guess at stuff that should be rasterised
        types_to_raster = ['QuadMesh', 'patches', 'Contour', 'collections']
        rasterize_list = []

        print('No rasterize_list specified, so the following objects will')
        print('be rasterized:')
        # Get all axes, and then get objects within axes
        for ax in fig.get_axes():
            for item in ax.get_children():
                if any(x in str(item) for x in types_to_raster):
                    rasterize_list.append(item)
    else:
        # Sometimes I forget to make rasterize_list a list, when only rasterizing
        # one part of the figure. Can't hurt to ensure its a list
        rasterize_list = list(rasterize_list)

    for item in rasterize_list:

        # Whether or not plot is a contour plot is important
        is_contour = hasattr(item, 'ax')

        # Whether or not collection of lines
        # is_lines = isinstance(item, matplotlib.collections.LineCollection)

        # Whether or not plot is a list of polygons
        try:
            is_polygons = isinstance(item[0], matplotlib.patches.Polygon)
        except TypeError:
            is_polygons = False

        # Get current axis object and set it's z order
        # curr_ax = item.ax.axes if is_contour else item.axes
        # curr_ax.set_rasterization_zorder(zorder)

        # Convert to rasterized mode and then change zorder properties
        if is_contour:
            curr_ax = item.ax.axes
            curr_ax.set_rasterization_zorder(zorder)
            # For contour plots, need to set each part of the contour
            # collection individually
            for contour_level in item.collections:
                contour_level.set_zorder(zorder - 1)
                contour_level.set_rasterized(True)
        elif is_polygons:
            # For list of polygons, need to set each part polygon individually
            for polygon in item:
                curr_ax = polygon.axes
                curr_ax.set_rasterization_zorder(zorder)
                polygon.set_zorder(zorder - 1)
                polygon.set_rasterized(True)
        else:
            # Otherwise, we can just do it all at once
            curr_ax = item.axes
            curr_ax.set_rasterization_zorder(zorder)
            item.set_rasterized(True)
            item.set_zorder(zorder - 1)

    # dpi is a savefig keyword argument, but treat it as special since it is
    # probably the one I'll use all the time
    if dpi is not None:
        savefig_kw['dpi'] = dpi

    # Save resulting figure
    fig.savefig(fname, **savefig_kw)


# Test raster and save
if __name__ is '__main__':
    fig, (ax1, ax2) = plt.subplots(2)
    cax1 = ax1.contourf(np.random.random((10, 10)))
    cax2 = ax2.scatter(np.random.random((11, 11)), np.random.random((11, 11)),
                       10*np.random.random((10, 10)))
    plt.draw()

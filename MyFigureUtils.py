# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 14:57:23 2015

@author: Ken
"""
from inspect import getmembers, isclass
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

    if len(fig_list) == 2:
        points = points[[2, 5]]
    if len(fig_list) == 3:
        points = points[[2, 4, 5]]
    if len(fig_list) == 4:
        points = points[[1, 2, 4, 5]]

    for i in range(len(fig_list)):
        plt.figure(fig_list[i])
        plt.get_current_fig_manager().window.setGeometry(
            points[i,0],points[i,1], points[i,2], points[i,3])


def raster_and_save(fname, rasterize_list=None, fig=None, dpi=None,
                    savefig_kw={}):
    """Save a figure with raster and vector components

    This function lets you specify which objects to rasterize at the export
    stage, rather than within each plotting call. Rasterizing certain
    components of a complex figure can significantly reduce file size.

    Inputs
    ------
    fname : str
        Output filename with extension
    rasterize_list : list (or object)
        List of objects to rasterize (or a single object to rasterize)
    fig : matplotlib figure object
        Defaults to current figure
    dpi : int
        Resolution (dots per inch) for rasterizing
    savefig_kw : dict
        Extra keywords to pass to matplotlib.pyplot.savefig

    If rasterize_list is not specified, then all contour, pcolor, and
    collects objects (e.g., ``scatter, fill_between`` etc) will be
    rasterized

    Note: does not work correctly with round=True in Basemap

    Example
    -------
    Rasterize the contour, pcolor, and scatter plots, but not the line

    >>> from numpy.random import random
    >>> X, Y, Z = random((9, 9)), random((9, 9)), random((9, 9))
    >>> fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)
    >>> cax1 = ax1.contourf(Z)
    >>> cax2 = ax2.scatter(X, Y, s=Z)
    >>> cax3 = ax3.pcolormesh(Z)
    >>> cax4 = ax4.plot(Z[:, 0])
    >>> rasterize_list = [cax1, cax2, cax3]
    >>> raster_and_save('out.svg', rasterize_list, fig=fig, dpi=300)
    """

    # Behave like pyplot and act on current figure if no figure is specified
    fig = plt.gcf() if fig is None else fig

    # Need to set_rasterization_zorder in order for rasterizing to work
    zorder = -5  # Somewhat arbitrary, just ensuring less than 0

    if rasterize_list is None:
        # Have a guess at stuff that should be rasterised
        types_to_raster = ['QuadMesh', 'Contour', 'collections']
        rasterize_list = []

        print("""
        No rasterize_list specified, so the following objects will
        be rasterized: """)
        # Get all axes, and then get objects within axes
        for ax in fig.get_axes():
            for item in ax.get_children():
                if any(x in str(item) for x in types_to_raster):
                    rasterize_list.append(item)
        print('\n'.join([str(x) for x in rasterize_list]))
    else:
        # Allow rasterize_list to be input as an object to rasterize
        if type(rasterize_list) != list:
            rasterize_list = [rasterize_list]

    for item in rasterize_list:

        # Whether or not plot is a contour plot is important
        is_contour = (isinstance(item, matplotlib.contour.QuadContourSet) or
                      isinstance(item, matplotlib.tri.TriContourSet))

        # Whether or not collection of lines
        # This is commented as we seldom want to rasterize lines
        # is_lines = isinstance(item, matplotlib.collections.LineCollection)

        # Whether or not current item is list of patches
        all_patch_types = tuple(
            x[1] for x in getmembers(matplotlib.patches, isclass))
        try:
            is_patch_list = isinstance(item[0], all_patch_types)
        except TypeError:
            is_patch_list = False

        # Convert to rasterized mode and then change zorder properties
        if is_contour:
            curr_ax = item.ax.axes
            curr_ax.set_rasterization_zorder(zorder)
            # For contour plots, need to set each part of the contour
            # collection individually
            for contour_level in item.collections:
                contour_level.set_zorder(zorder - 1)
                contour_level.set_rasterized(True)
        elif is_patch_list:
            # For list of patches, need to set zorder for each patch
            for patch in item:
                curr_ax = patch.axes
                curr_ax.set_rasterization_zorder(zorder)
                patch.set_zorder(zorder - 1)
                patch.set_rasterized(True)
        else:
            # For all other objects, we can just do it all at once
            curr_ax = item.axes
            curr_ax.set_rasterization_zorder(zorder)
            item.set_rasterized(True)
            item.set_zorder(zorder - 1)

    # dpi is a savefig keyword argument, but treat it as special since it is
    # important to this function
    if dpi is not None:
        savefig_kw['dpi'] = dpi

    # Save resulting figure
    fig.savefig(fname, **savefig_kw)


# Test raster_and_save
if __name__ is '__main__':
    from numpy.random import random
    X, Y, Z = random((9, 9)), random((9, 9)), random((9, 9))
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2)
    cax1 = ax1.contourf(Z)
    cax2 = ax2.scatter(X, Y, s=Z)
    cax3 = ax3.pcolormesh(Z)
    cax4 = ax4.plot(Z[:, 0])
    rasterize_list = [cax1, cax2, cax3]
    raster_and_save('out.svg', rasterize_list, fig=fig, dpi=300)

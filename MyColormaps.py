# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 09:38:31.0 201.05

@author: hugke729
"""
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import colorsys
from colorsys import rgb_to_hls, hls_to_rgb
from fractions import Fraction
import itertools


def make_N_colors(cmap_name, N):
    cmap = cm.get_cmap(cmap_name, N)
    return cmap(np.arange(N))


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    """Take a string cmap and get rid of colors at one or each end"""
    # http://stackoverflow.com/questions/18926031/
    # how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    cmap = plt.get_cmap(cmap)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def cmap_d(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet.
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)

    http://wiki.scipy.org/Cookbook/Matplotlib/ColormapTransformations
    """
    if type(cmap) == str:
        cmap = cm.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki])
                      for i in range(N + 1)]
    # Return colormap object.
    return LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


def red_blue(levels):
    cmap = colors.LinearSegmentedColormap.from_list(
        name='red_white_blue',
        colors=[(0, 0, 1), (1, 1, 1), (1, 0, 0)],
        N=len(levels) - 1)
    return cmap


def addNorm(cmapData):
    # http://stackoverflow.com/questions/18599625/
    # matplotlib-standard-colormap-usage
    cmapData['norm'] = colors.BoundaryNorm(
        cmapData['bounds'], cmapData['cmap'].N)
    return True


def discretize(cmap, bounds):
    # http://stackoverflow.com/questions/18599625/
    # matplotlib-standard-colormap-usage
    resCmap = {}
    resCmap['cmap'] = colors.ListedColormap(
        [cmap(i / len(bounds[1:])) for i in range(len(bounds[1:]))])
    resCmap['bounds'] = bounds
    addNorm(resCmap)
    return resCmap


def get_n_colors2(num_colors):
    """http://stackoverflow.com/questions/470690/
    how-to-automatically-generate-n-distinct-colors"""
    colors = []
    for i in np.arange(0., 360., 360. / num_colors):
        hue = i / 360.
        lightness = (50 + np.random.rand() * 10) / 100.
        saturation = (90 + np.random.rand() * 10) / 100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


def get_n_colors(num_colors):
    """http://stackoverflow.com/questions/470690/
    how-to-automatically-generate-n-distinct-colors

    Ken added a few changes to work as desired
    Consequently, there may be stuff in here that is unnecessary since I
    couldn't be bothered to simplify it"""

    def zenos_dichotomy():
        for k in itertools.count():
            yield Fraction(1, 2**k)

    def getfracs():
        """
        [Fraction(0, 1), Fraction(1, 2), Fraction(1, 4), Fraction(3, 4), Fraction(1, 8), Fraction(3, 8), Fraction(5, 8), Fraction(7, 8), Fraction(1, 16), Fraction(3, 16), ...]
        [0.0, 0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875, 0.0625, 0.1875, ...]
        """
        yield 0
        for k in zenos_dichotomy():
            i = k.denominator  # [1,2,4,8,16,...]
            for j in range(1, i, 2):
                yield Fraction(j, i)

    def bias(x):
        # can be used for the v in hsv to map linear values 0..1 to something
        # that looks equidistant
        return ((math.sqrt(x / 3) / Fraction(2, 3) + Fraction(1, 3)) /
                Fraction(6, 5))

    def genhsv(h):
        for s in [Fraction(6, 10)]:  # optionally use range
            for v in [Fraction(8, 10), Fraction(5, 10)]:  # could use range too
                yield (h, s, v)  # use bias for v here if you use range

    def genrgb(x):
        return colorsys.hsv_to_rgb(*x)

    flatten = itertools.chain.from_iterable

    def gethsvs():
        return flatten(map(genhsv, getfracs()))

    def getrgbs():
        return map(genrgb, gethsvs())

    def gencols(x):
        return np.array(x).astype('float64')

    def getcolors():
        return map(gencols, getrgbs())

    return np.array(list(itertools.islice(getcolors(), num_colors)))


def HTMLColorToRGB(colorstring):
    """ convert #RRGGBB to an (R, G, B) tuple in (0--1) format """
    colorstring = colorstring.strip()
    if colorstring[0] == '#':
        colorstring = colorstring[1:]

    err_msg = "input #%s is not in #RRGGBB format" % colorstring
    assert len(colorstring) == 6, err_msg

    r, g, b = colorstring[:2], colorstring[2:4], colorstring[4:]
    r, g, b = [int(n, 16)/255 for n in (r, g, b)]
    return (r, g, b)


def one_color_binary_cmap(color, reverse=False):
    """Create a colormap with only two colors, white and specified color

    Inputs
    ------
    color: any of usual ways of specifying color
    reverse: if True starts with color and goes to white

    Not currently set up to work with alpha
    """
    if type(color) is str:
        color = HTMLColorToRGB(color)

    color = np.array(color)
    white = np.array([1, 1, 1])

    C = np.c_[white, color].T

    if reverse:
        C = np.flipud(C)

    binary_cmap = mpl.colors.ListedColormap(C)
    return binary_cmap


def red_yellow_white_cyan_blue(N=256, reverse=True, white_to_grey=False):
    cols = [
        '#000055',
        '#0000f5',
        '#008cff',
        '#7affff',
        '#ffffff',
        '#ffff83',
        '#ff9d0c',
        '#ff0500',
        '#5f0000']
    if white_to_grey:
        cols[3:6] = ['#9ee6e6', '#ececec', '#eaea99']
    if reverse:
        cols = cols[::-1]
    return LinearSegmentedColormap.from_list('custom', cols, N)


def red_yellow_grey_cyan_blue(N=256, reverse=False):
    return red_yellow_white_cyan_blue(N, reverse=reverse, white_to_grey=True)


def purple_grey_green(N=256, reverse=False):
    cols = ['#46005f', '#ba00ff', '#ff64d0', '#f0b3cb', '#ececec',
            '#aee28f', '#0bb337', '#004d0d', '#001e0f']

    if reverse:
        cols = cols[::-1]

    return LinearSegmentedColormap.from_list('custom', cols, N)


def cmap_cold(reverse=False, green=False):
    C = np.array(
        [[0.9857, 0.9857, 0.9857],
         [0.9714, 0.9714, 0.9714],
         [0.9571, 0.9571, 0.9571],
         [0.9429, 0.9429, 0.9429],
         [0.9286, 0.9286, 0.9286],
         [0.9143, 0.9143, 0.9143],
         [0.9000, 0.9000, 0.9000],
         [0.8857, 0.8857, 0.8857],
         [0.8714, 0.8714, 0.8714],
         [0.8571, 0.8571, 0.8571],
         [0.8429, 0.8429, 0.8429],
         [0.8286, 0.8286, 0.8286],
         [0.8143, 0.8143, 0.8143],
         [0.8000, 0.8000, 0.8000],
         [0.7500, 0.7968, 0.8125],
         [0.7000, 0.7936, 0.8250],
         [0.6500, 0.7904, 0.8375],
         [0.6000, 0.7873, 0.8500],
         [0.5500, 0.7841, 0.8625],
         [0.5000, 0.7809, 0.8750],
         [0.4500, 0.7777, 0.8875],
         [0.4000, 0.7745, 0.9000],
         [0.3500, 0.7713, 0.9125],
         [0.3000, 0.7681, 0.9250],
         [0.2500, 0.7650, 0.9375],
         [0.2000, 0.7618, 0.9500],
         [0.1500, 0.7586, 0.9625],
         [0.1000, 0.7554, 0.9750],
         [0.0500, 0.7522, 0.9875],
         [0.0000, 0.749, 1.0000],
         [0.00560, 0.7076, 0.9678],
         [0.01120, 0.6661, 0.9356],
         [0.01680, 0.6246, 0.9034],
         [0.02240, 0.5832, 0.8711],
         [0.02800, 0.5417, 0.8389],
         [0.03360, 0.5003, 0.8067],
         [0.03920, 0.4588, 0.7745],
         [0.04480, 0.4174, 0.7423],
         [0.05040, 0.3759, 0.7101],
         [0.05600, 0.3345, 0.6779],
         [0.06160, 0.2930, 0.6457],
         [0.06720, 0.2515, 0.6134],
         [0.07280, 0.2101, 0.5812],
         [0.07840, 0.1686, 0.5490],
         [0.07320, 0.1574, 0.5299],
         [0.06800, 0.1461, 0.5108],
         [0.06270, 0.1349, 0.4918],
         [0.05750, 0.1237, 0.4727],
         [0.05230, 0.1124, 0.4536],
         [0.04710, 0.1012, 0.4345],
         [0.04180, 0.0899, 0.4154],
         [0.03660, 0.0787, 0.3963],
         [0.03140, 0.0675, 0.3772],
         [0.02610, 0.0562, 0.3581],
         [0.02090, 0.0450, 0.3391],
         [0.01570, 0.0337, 0.3200],
         [0.01050, 0.0225, 0.3009],
         [0.00520, 0.0112, 0.2818],
         [0.00000, 0.0000, 0.2627]])

    # if green:
    #    hls = np.r_[[rgb_to_hls(*row) for row in C]]
    #    h_green = 100/255
    #    h_blue = 136/255
    #    h_tmp = hls[:, 0].copy()
    #    h_tmp[h_tmp != 0] += h_green - h_blue
    #    hls[:, 0] = h_tmp
    #    C = np.r_[[hls_to_rgb(*row) for row in hls]]
    if green:
        C = np.array(
            ['#004320', '#034d26', '#086031', '#0d703b', '#13874b', '#10a556',
             '#09ce66', '#03ef72', '#1af983', '#4dec98', '#8cdcb2', '#bfcfc7',
             '#d7d7d7', '#e6e6e6', '#f4f4f4', '#fbfbfb'])

    if reverse:
        C = np.flipud(C)

    cold = mpl.colors.ListedColormap(C)
    return cold

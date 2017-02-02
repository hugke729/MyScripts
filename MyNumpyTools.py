# Convenient tools that numpy doesn't have
import numpy as np
import numpy.ma as ma
from pandas import DataFrame
from warnings import filterwarnings
from scipy.signal import convolve
from scipy.interpolate import RegularGridInterpolator


def cosd(angle):
    """cosine with argument in degrees"""
    out = np.cos(np.deg2rad(angle))
    return out


def sind(angle):
    """sine with argument in degrees"""
    out = np.sin(np.deg2rad(angle))
    return out


def tand(angle):
    """sine with argument in degrees"""
    out = np.tan(np.deg2rad(angle))
    return out


def minmax(x, axis=None, astype='float'):
    """returns tuple (min(x), max(x)) while ignoring nans"""
    extrema = (np.nanmin(x, axis=axis).astype(astype),
               np.nanmax(x, axis=axis).astype(astype))
    return extrema


def nan_or_masked(arr):
    """Returns true if value is either nan or masked"""
    try:
        masked_vals = arr.mask
        nan_vals = np.isnan(arr.data)
        invalid_val = np.logical_or(masked_vals, nan_vals)
    except AttributeError:
        invalid_val = np.isnan(arr)
    return invalid_val


def nan_to_value(x, value):
    x[np.isnan(x)] = value
    return x


def nan_or_masked_to_value(x, value):
    x[nan_or_masked(x)] = value
    return x


def argmin_nd(arr):
    """Like np.argmin, but return result as N-dimensional tuple"""
    # http://stackoverflow.com/questions/3230067/
    # numpy-minimum-in-row-column-format
    return np.unravel_index(arr.argmin(), arr.shape)


def argmax_nd(arr):
    """Like np.argmax, but return result as N-dimensional tuple"""
    # http://stackoverflow.com/questions/3230067/
    # numpy-minimum-in-row-column-format
    return np.unravel_index(arr.argmax(), arr.shape)


def logical_all(*args):
    """Like logical_and, but for any number of input arrays"""
    out = np.ones_like(args[0]).astype(bool)
    for arg in args:
        out = np.logical_and(out, arg)

    return out


def logical_any(*args):
    """Like logical_and, but for any number of input arrays"""
    out = np.zeros_like(args[0]).astype(bool)
    for arg in args:
        out = np.logical_or(out, arg)

    return out


def ma_percentile(a, q, axis=None, **kw_args):
    """Equivalent to np.percentile, but works on masked arrays

    Inputs a, q, and axis are as for np.percentile. Other inputs can
    be passed using kw_args"""
    filterwarnings('ignore', 'setting an item on a masked array')
    try:
        a[a.mask] = np.nan
        mask = True
    except AttributeError:
        # Array has no mask, so nothing to change to np.nan
        mask = False

    filterwarnings('ignore', 'All-NaN slice encountered')
    out = np.nanpercentile(a, q, axis=axis, **kw_args)

    # Return masked result
    if mask:
        out = ma.masked_invalid(out)

    return out


def maxabs(x, **kw_args):
    return np.nanmax(np.abs(x), **kw_args)


def max_to_lims(x):
    """Find maximum absolute value, and return (-max, max)"""
    max_val = maxabs(x)
    return np.array((-max_val, max_val))


def neg_pos(x):
    """Convenience function to convert ±x to (-x, x)"""
    return max_to_lims(x)


def ma_mad(x, axis=None):
    """Median absolute deviation"""
    median_x = ma.median(x, axis=axis)
    if axis is not None:
        median_x = ma.expand_dims(median_x, axis=axis)

    return ma.median(ma.abs(x - median_x), axis=axis)


def fillnan_pad(x, axis=0):
    """Replace NaN values with last finite value"""
    x = DataFrame(x).fillna(method='pad', axis=axis)
    return np.array(x)


def fillnan_pad3d(x):
    """Replace NaN values with last finite value"""

    # Have to use row if we want to take advantage of pandas' method
    for i, xi in enumerate(x):
        x[i, ...] = np.array(DataFrame(xi).fillna(method='pad', axis=1))

    return np.array(x)


def change_wrap(x, wrap_in, wrap_out):
    """Change angle wrapping limits

    Inputs
    ------
    x : array
        Angles to change
    wrap_in : int or float
        Wrap value for input array
    wrap_out : int or float
        Wrap value to convert to

    If input data are in range (0, 360), wrap_in is 360

    Returns
    -------
    x : array
        Rewrapped with different limits
    """

    # Copy x so weird things don't happen
    x = x.copy()

    # Ensure x is array
    x = np.asanyarray(x)

    # Other end of wrap
    wrap_in_0 = wrap_in - 360
    wrap_out_0 = wrap_out - 360

    # Ensure all x values are within correct range to start with
    assert np.nanmin(x) >= wrap_in_0 and np.nanmax(x) <= wrap_in

    filterwarnings('ignore', '.*invalid value encountered in less*.')
    filterwarnings('ignore', '.*invalid value encountered in greater*.')
    inc_inds = x < wrap_out_0
    dec_inds = x > wrap_out

    x[inc_inds] = wrap_in + (x[inc_inds] - wrap_in_0)
    x[dec_inds] = wrap_in_0 - (wrap_in - x[dec_inds])

    return x


def uneven_2D_convolve(in1, x1, y1, in2, Nx=1000, Ny=1000):
    """2D convolution on regular but uneven grid

    Interpolates to regular grid, runs convolution, then interpolates back

    Inputs
    ------
    in1 : 2D array
        First array used in scipy's convolve
    x1, y1 : 1D arrays
        Cartesian coordinates for in1
    in2 : 2D array
        |  Second array used in scipy's convolve
        |  in2 is assumed to be on the regular grid defined by edges of x1 and
        y1 and number of grid points Nx and Ny
    Nx, Ny : integers
        Number of points used in intermediate grid

    Returns
    -------
    convolve : 2D array
        Result of convolution on x1, y1 grid
    """
    # Create temporary, regular grids
    tmp_x1 = np.linspace(x1.min(), x1.max(), Nx)
    tmp_y1 = np.linspace(y1.min(), y1.max(), Ny)

    # Create meshes
    tmp_X1, tmp_Y1 = np.meshgrid(tmp_x1, tmp_y1, indexing='ij')
    X1, Y1 = np.meshgrid(x1, y1, indexing='ij')

    # Interpolate in1 and in2 to regular grid
    tmp_in1 = RegularGridInterpolator((x1, y1), in1)((tmp_X1, tmp_Y1))

    # Do convolution on regular grid
    tmp_conv = convolve(tmp_in1, in2, mode='same')

    # Convert convolution result back to original grid
    conv = RegularGridInterpolator((tmp_x1, tmp_y1), tmp_conv)((X1, Y1))

    return conv


def next_pow_2(x):
    return (2**np.ceil(np.log2(x))).astype(int)


def prev_pow_2(x):
    return (2**np.floor(np.log2(x))).astype(int)

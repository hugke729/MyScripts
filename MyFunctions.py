import math
import numpy as np
import numpy.ma as ma
import itertools
import warnings
from MyNumpyTools import nan_or_masked
from matplotlib._cntr import Cntr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def divisorGenerator(n):
    large_divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i is 0:
            yield i
            if i is not n / i:
                large_divisors.insert(0, n / i)
    for divisor in large_divisors:
        yield divisor


def divisors(n, type_out=int):
    """Compute all divisors of n"""
    # http://stackoverflow.com/questions/171765/
    # what-is-the-best-way-to-get-all-the-divisors-of-a-number
    div_array = np.array(list(divisorGenerator(n))).astype(type_out)

    return div_array


def central_diff_gradient(F, *x):
    """Gradient using central difference

    Forward and backward difference at ends
    F can be one or two dimensional
    If F is two dimensional, x can be one or two dimensional
    If no x is specified, it is assumed to be increments of one"""
    if not len(x):
        x = np.arange(F.shape[0])

    # Ensure that inputs are numpy arrays
    x, F = np.asanyarray(x), np.asanyarray(F)
    x = np.squeeze(x)

    # Ensure correct shape
    if F.size == x.size and F.ndim > 1:
        assert x.shape == F.shape, 'F and x are not the same shape'

    # Ensure correct shape
    F_needs_transpose = (F.shape[0] != x.shape[0])
    if F_needs_transpose:
        F = F.T

    # Ensure F and x have two dimensions, even if that means second dimension
    # have a length of 1
    if F.ndim == 1:
        F = np.expand_dims(F, 1)
    if x.ndim == 1:
        x = np.expand_dims(x, 1)

    # Account for masked values or NaN values in F (assuming none in x)
    # And assumes invalid values only at ends (none in middle of data)
    # zero, one, neg_one, neg_two take place of 0, 1, -1, -2 in indexing
    # They're respectively the first, second, last, and second to last values
    # that are not masked or NaN
    zero = np.argmin(nan_or_masked(F), axis=0)
    one = zero + 1
    two = zero + 2
    end = F.shape[0] - np.argmin(np.flipud(nan_or_masked(F)), axis=0)
    neg_one = end - 1
    neg_two = neg_one - 1

    # Preallocate array into which to save the result
    F_x = np.zeros(F.shape)

    # Forward difference at left end
    h = x[one, :] - x[zero, :]
    F_x[zero, :] = (F[one, :] - F[zero, :])/h

    # Backward difference at right end
    h = x[neg_one, :] - x[neg_two, :]
    F_x[neg_one, :] = (F[neg_one, :] - F[neg_two, :])/h

    # Central Difference in interior
    # Step sizes
    for col in range(F.shape[1]):
        xcol = 0 if x.shape[1] == 1 else col
        h_i = np.diff(x[zero[col]:neg_one[col], xcol])
        h_ip1 = np.diff(x[one[col]:end[col], xcol])

        # Differences
        dF_i = F[one[col]:neg_one[col], col] - F[zero[col]:neg_two[col], col]
        dF_ip1 = F[two[col]:end[col], col] - F[one[col]:neg_one[col], col]

    # Gradient at i using F[i-1], F[i] and F[i+1]. Weighted by how close
    # F[i] is to F[i-1] and F[i+1]
        F_x[one[col]:neg_one[col], col] = (
            (h_ip1/(h_i + h_ip1))*(dF_i/h_i) +
            (h_i/(h_i + h_ip1))*(dF_ip1/h_ip1))

    # Return output to shape of input
    if F_needs_transpose:
        F_x = F_x.T

    # Remove extra dimension that was added to one-dimensional arrays
    F, F_x = map(np.squeeze, [F, F_x])

    # Mask result based on input mask
    if np.any(nan_or_masked(F)):
        F_x = ma.masked_where(nan_or_masked(F), F_x)
    return F_x


def cosd(angle):
    err_msg = 'Import cosd from MyNumpyTools instead of MyFunctions'
    raise DeprecationWarning(err_msg)


def sind(angle):
    err_msg = 'Import sind from MyNumpyTools instead of MyFunctions'
    raise DeprecationWarning(err_msg)


def lat2str(deg):
    min = 60 * (deg - np.floor(deg))
    deg = np.floor(deg)
    if np.allclose(min, 0.0):
        min = 0
    elif deg < 0:
        deg += 1.0
        min -= 60.0
    return ("%d\N{DEGREE SIGN} %g$\prime$") % (np.abs(deg), np.abs(min))


def lon2strW(deg):
    deg = 360 - deg
    min = 60 * (deg - np.floor(deg))
    deg = np.floor(deg)
    if np.allclose(min, 0.0):
        min = 0
        no_min = True
    elif deg < 0:
        deg += 1.0
        min -= 60.0
        no_min = False
    if no_min:
        return ("%d\N{DEGREE SIGN}") % -np.abs(deg)
    else:
        return ("%d\N{DEGREE SIGN} %g$\prime$") % (-np.abs(deg), np.abs(min))


def scalar_projection(x0, y0, x1, y1, x, y):
    """Projection of vector (x - x0, y - y0) onto (x1 - x0, y1 - y0)"""
    A = (x - x0, y - y0)
    B = (x1 - x0, y1 - y0)
    return (A[0]*B[0] + A[1]*B[1])/np.hypot(*B)


def angle(x0, y0, x1, y1, x, y, degrees=False):
    """Angle between vectors (x - x0, y - y0) and (x1 - x0, y1 - y0)"""
    # Convert the three input points to two vectors x1, y1 and x2, y2
    x1, x2, y1, y2 = x1 - x0, x - x0, y1 - y0, y - y0
    inner_product = x1*x2 + y1*y2
    len1 = np.hypot(x1, y1)
    len2 = np.hypot(x2, y2)
    angle = np.arccos(inner_product/(len1*len2))
    return (np.rad2deg(angle) if degrees else angle)


def flatten2Dlist(list2d):
    # http://stackoverflow.com/questions/952914/
    # making-a-flat-list-out-of-list-of-lists-in-python
    try:
        return list(itertools.chain(*list2d))
    except TypeError:
        # Already flat?
        return list2d


def minmax(x, axis=None, astype='float'):
    err_msg = 'Import minmax from MyNumpyTools instead of MyFunctions'
    raise DeprecationWarning(err_msg)


def argmin_nd(arr):
    err_msg = 'Import argmin_nd from MyNumpyTools instead of MyFunctions'
    raise DeprecationWarning(err_msg)


def argmax_nd(arr):
    err_msg = 'Import argmax_nd from MyNumpyTools instead of MyFunctions'
    raise DeprecationWarning(err_msg)


def blockify(x, y):
    """Convert x and y vectors so that they can be used for line plots where
    the y values are piecewise constant

    x needs to be y.size + 1"""
    assert x.size == y.size + 1, "x should be one value longer than y"

    x_repeat, y_repeat = ma.zeros(2*y.size), ma.zeros(2*y.size)
    x_repeat[::2], x_repeat[1::2] = x[:-1], x[1:]
    y_repeat[::2], y_repeat[1::2] = y, y
    return x_repeat, y_repeat


def ma_percentile(a, q, axis=None, **kw_args):
    err_msg = 'Import ma_percentile from MyNumpyTools instead of MyFunctions'
    raise DeprecationWarning(err_msg)


def get_contour(x, y, Z, levels, fill_value=np.nan):
    """Get just the data that would be returned by plt.contour

    The output is interpolated onto the original x grid

    Inputs
    ------
    x, y : 1D arrays
        Arrays that you would input into plt.contour
    Z : 2D arrays
        Array that you would input into plt.contour
    levels : float, array, or list
        Value(s) at which contours are calculated
    sort_x : float or 'extrapolate'
        Value passed to interp1d
        Setting to 'extrapolate' may help remove NaNs

    Returns
    -------
    contour_values : 1D or 2D array
        Shape depends on number of levels
    """

    # Allow Z array to be either way around
    try:
        X, Y = np.meshgrid(x, y, indexing='ij')
        c = Cntr(X, Y, Z)
    except ValueError:
        X, Y = np.meshgrid(x, y)
        c = Cntr(X, Y, Z)

    # Ensure levels is iterable
    if not hasattr(levels, '__iter__'):
        levels = np.r_[levels]

    out = np.zeros((len(x), len(levels)))

    warnings.filterwarnings('ignore', 'invalid value encountered*')

    for i, level in enumerate(levels):
        res = c.trace(level)

        # result is a list of arrays of vertices and path codes
        # (see docs for matplotlib.path.Path)
        nseg = len(res) // 2
        x_cont = np.vstack(res[:nseg])[:, 0]
        z_cont = np.vstack(res[:nseg])[:, 1]

        f = interp1d(
            x_cont, z_cont, fill_value=fill_value, bounds_error=False)
        out[:, i] = f(x)

    return out.squeeze()


def fit_gaussian(x, y, init_params, print_result=False):
    """Fit gaussian curve

    Inputs
    ------
    x, y: arrays
    init_params: initial guess for amplitude, mean and std dev

    Output
    ------
    amplitude, mean, standard deviation
    """
    def gauss(x, *p):
        A, mu, sigma = p
        return A*np.exp(-(x-mu)**2/(2.*sigma**2))

    fit = curve_fit(gauss, x, y, init_params)[0]

    if print_result:
        print('Amplitude: ' + str(fit[0]))
        print('Mean: ' + str(fit[1]))
        print('Std Dev: ' + str(fit[2]))

    return fit


def calc_power_law(x, y, plot_results=False):
    x, y = np.log10(x), np.log10(y)
    p, V = np.polyfit(x, y, 1, cov=True)
    slope = p[0]
    # https://stackoverflow.com/questions/27634270/
    # how-to-find-error-on-slope-and-intercept-using-numpy-polyfit
    slope_err = np.sqrt(V[0][0])
    print(slope_err)
    p_high, p_low = p.copy(), p.copy()
    p_high[0] = p[0] + slope_err
    p_low[0] = p[0] - slope_err
    slope_str = 'Slope = {0:2.2f} ± {1:2.2f}'.format(slope, slope_err)
    print(slope_str)
    if plot_results:
        plt.plot(x, y, 'ko')
        plt.plot(x, np.polyval(p, x), 'k')
        plt.plot(x, np.polyval(p_high, x), 'k--')
        plt.plot(x, np.polyval(p_low, x), 'k--')
        plt.title(slope_str)

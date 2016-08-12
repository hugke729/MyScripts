# http://stackoverflow.com/questions/171765/
# what-is-the-best-way-to-get-all-the-divisors-of-a-number
import math
import numpy as np
import numpy.ma as ma
import itertools
from warnings import filterwarnings
from MyNumpyTools import nan_or_masked, nan_to_value
import inspect


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

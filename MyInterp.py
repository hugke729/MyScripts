import scipy.spatial.qhull as qhull
import numpy as np
import numpy.ma as ma
from warnings import filterwarnings
from scipy.interpolate import griddata
from MyNumpyTools import nan_or_masked, logical_all
from scipy.sparse.linalg import lsqr
from scipy.sparse import lil_matrix
from mpl_toolkits.mplot3d import axes3d
from MyGrids import telescope_centre_n


def interp_weights(xyz, uvw, preflattened=False):
    """Determine vertices and weights to use to interpolate between two grids

    Uses a Delaunay triangulation as described at
    ``http://stackoverflow.com/questions/20915502/
    speedup-scipy-griddata-for-multiple-interpolations-between
    -two-irregular-grids``

    Inputs
    ------
    xyz : tuple of two 1D or 2D arrays
        X and Y grids on which data to be interpolated lie
    uvw : tuple of two 1D or 2D arrays
        Coordinates of grid that data will be interpolated on to
    preflattened : bool
        If true, xyz is tuple of 1D arrays that are already flattened from 2D

    Returns
    -------
    vtx : array
        N x 3 vector of vertices
    wts : array
        N x 3 vector of corresponding weights
    N is total number of grid points in uvw
    """

    # If 1D arrays are passed in, convert them to 2D to get correct number
    # of total coordinate points
    if ((xyz[0].ndim + xyz[1].ndim) == 2) and xyz[0].size != xyz[1].size:
        if not preflattened:
            xyz = np.meshgrid(xyz[0], xyz[1])
    if ((uvw[0].ndim + uvw[1].ndim) == 2) and uvw[0].size != uvw[1].size:
        if not preflattened:
            uvw = np.meshgrid(uvw[0], uvw[1])

    # Ensure inputs array are flattened
    xyz = tuple(X.flatten() for X in xyz)
    uvw = tuple(X.flatten() for X in uvw)

    # Ensure inputs are of the correct form
    xyz, uvw = np.asanyarray(xyz), np.asanyarray(uvw)
    if len(xyz.shape) > 1 and (xyz.shape[0] < xyz.shape[1]):
        xyz = xyz.T
    if len(uvw.shape) > 1 and (uvw.shape[0] < uvw.shape[1]):
        uvw = uvw.T

    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    d = 2
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


def interpolate(values, vtx, wts, fill_value=None):
    """Interpolate values using weights and vertices from interp_weights

    Ultimately, this does the same job as scipy's interpolate. However, by
    calculating the vertices and weights in a previous step, it makes repeated
    interpolations much quicker. Useful for say model grids in which
    interpolation in x, y occurs over multiple z levels

    Idea comes from
    ``http://stackoverflow.com/questions/20915502/
    speedup-scipy-griddata-for-multiple-interpolations-between
    -two-irregular-grids``

    Inputs
    ------
    values : array (1D or 2D)
        The values to interpolate
    vtx : N x 3 array
        Vertices output from interp_weights
    wts : N x 3 array
        Weights output from interp_weights
    fill_value : None, float, or np.nan
        Value for points that are not interpolateable
    """
    # Notes on how this works:
    # np.einsum('nj,nj->n', A, B) is equivalent to np.sum(A*B, axis=1)
    # np.take(values, vtx) is equivalent to values[vtx]
    # original
    # ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    # new to allow for values to contain NaNs

    # Ensure values array is 1D
    if values.ndim > 1:
        values = values.flatten()

    # indices of where there are NaNs in values
    nans = np.array(np.where(np.isnan(values)))

    # mask[i, j] is true if vtx[i, j] is contained in nans
    mask = np.column_stack((np.in1d(vtx[:, 0], nans),
                            np.in1d(vtx[:, 1], nans),
                            np.in1d(vtx[:, 2], nans)))

    # copy values[wts] and vtx, but mask where vtx corresponds to a NaN
    vals_ma = ma.masked_where(mask, values[vtx])
    wts_ma = ma.masked_where(mask, wts)
    ret = np.sum(vals_ma*wts_ma, axis=1)/np.sum(wts_ma, axis=1)

    # if fill_value has a value other than 'None', then check for negative
    # weights. If fill_value=None (ie else case) then do nothing
    if fill_value:
        holes = np.any(wts < 0, axis=1)
        ret[holes] = fill_value
    else:
        pass

    return ret


def surrounding_pts(pt, x_size, y_size, n=3):
    """get indices of surrounding 8 points, or n^2 - 1 where n is size of box

    if point is on an edge then round up or down. e.g.
    if pt (i, pt) = (0, 1), then surrounding 8 points will be (x, y) where
    x = 0, 0, 1 and y = 0, 1, 2"""

    # convert from (n x n) box to number of points to the l, r, u, d from centre
    m = (n-1)//2
    # indices for the holes and the points to the left and right
    x = np.r_[pt[0]-m:pt[0]+m+1]
    # indices for the holes and the points above and below
    y = np.r_[pt[1]-m:pt[1]+m+1]

    # for points beyond right edge, round down
    x[x >= x_size] = pt[0]
    # for points beyond left edge, round up
    x[x < 0] = 0

    # for points above top edge, round down
    y[y >= y_size] = pt[1]
    # for points below bottom edge, round up
    y[y < 0] = 0

    pts_x, pts_y = np.meshgrid(x, y)
    pts_x = pts_x.flatten()
    pts_y = pts_y.flatten()

    # remove original element
    pts_x = np.delete(pts_x, n**2//2, axis=0)
    pts_y = np.delete(pts_y, n**2//2, axis=0)

    return pts_x, pts_y


def grid(x, y, z, resX=100, resY=100):
    "Convert 3 column data to matplotlib grid"
    # http://stackoverflow.com/questions/18764814/make-contour-of-scatter
    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((x, y), z, (X, Y))
    return X, Y, Z


def downsample(x, nx_avg, invalid_to_zero=False):
    """Downsample x by averaging over nx_avg values.

    nx_avg need not be an integer, but it len(x)/nx_avg must be an integer

    If invalid_to_zero is True, then masked or NaN values are converted to
    zero before averaging"""
    nx = len(x)

    # Check x can be divided by nx_avg
    assert (nx/nx_avg).is_integer(), 'len(x) is not divisible by nx_avg'

    # Convert invalid values to zero if chosen
    if invalid_to_zero:
        x = ma.filled(x, 0)
        x[x == np.nan] = 0

    # Example of how averaging is done:
    # Let len(x) be 73, and nx_avg be 73/12
    # Rearrange x into 7 columns (ceil(73/12)) and 12 rows as follows
    #
    # 0,  1,  2,   3,  4,  5,  6,
    # 6,  7,  8,   9, 10, 11, 12,
    # ...
    # 66, 67, 68, 69, 70, 71, 72
    #
    # Then take mean along row

    w = np.linspace(0, nx, nx/nx_avg + 1)
    weights = np.column_stack((np.ceil(w[:-1]) - w[:-1],
                               np.ones((w.shape[0] - 1, int(nx_avg) - 1)),
                               w[1:] - np.floor(w[1:])))
    weights[weights == 0] = 1

    if (type(nx_avg) is int) or nx_avg.is_integer():
        # Upcoming line uses cumsum, which gets screwed up if nx_avg is an
        # integer in which case data straddling nx_avg is double counted
        weights[:, -1] = 0

    x_inds = (np.ceil(np.cumsum(weights).reshape(weights.shape))-1).astype(int)
    weight_total = np.sum(weights*~nan_or_masked(x[x_inds]), axis=1)
    x_out = np.sum(weights*x[x_inds], axis=1)/weight_total

    if not invalid_to_zero:
        x_out = ma.masked_where(weight_total == 0, x_out)

    return x_out


def smooth1d_with_holes(velocity, n):
    """Making my own smoothing routine to deal with holes

    Overall, it's a simple n-step moving average, but one that deals with edges
    by doing the following (using n = 5 as example):
    1st point: avg(1st)
    2nd point: avg(1st, 2nd, 3rd)
    3rd point: avg(1st, ..., 5th)
    4th point: avg(2nd, ..., 6th)
    5th point: avg(3rd, ..., 7th)
    and similarly at the other end
    """
    half_n_ceil = np.ceil(n/2).astype(int)
    half_n_floor = np.floor(n/2).astype(int)
    # Create a matrix of values that incrementally shift to the right by one
    # index per row
    # For most rows, this means ith column will contain i-n//2 to i+n//2 values
    vel_matrix = ma.outer(np.ones(n), velocity)
    vel_matrix.unshare_mask()
    for i, row in enumerate(vel_matrix):
        vel_matrix[i, :] = np.roll(row, i - half_n_floor)

    # For earlier and later columns, we need to remove some of the values that
    # are shifted from the other end. Do this by concatenating triangular blocks
    # to make arrowhead boolean blocks at each end
    bottom_left = np.tri(half_n_ceil, half_n_ceil, k=-1).astype(bool)
    top_right = ~bottom_left
    left = np.row_stack((np.fliplr(top_right), bottom_left))

    # if n is odd (n % 2), remove first row of 'left' to leave symmetrical block
    left = left[1:, :] if (n % 2) else left

    right = np.fliplr(left)

    # Change unwanted values to NaN
    filterwarnings('ignore', '.*setting an item on a masked array*.')
    vel_matrix[:, :half_n_ceil][left] = np.nan
    vel_matrix[:, -half_n_ceil:][right] = np.nan

    vel_matrix = ma.masked_invalid(vel_matrix)

    # Take mean to give smoothed result
    smoothed = np.nanmean(vel_matrix, axis=0)

    # Any values that were masked or nan to start with are converted to nan
    smoothed[nan_or_masked(velocity)] = np.nan

    return smoothed


def get_springs(A, leave_mask=None, plot_springs=False,
                include_diagonals=False):
    """
    Get (x0, y0), (x1, y1) indices for all springs

    Inputs
    ------
    A : 2D array
        Input array containing NaN values
    leave_mask : 2D array
        True in locations we don't want springs
    plot_springs: bool
        Whether to display results (default is False)
    include_diagonals : bool
        If True, account for all 8 adjacent values, not 4

    Returns
    -------
    springs : N x 4 array
        Array with rows (x0, y0, x1, y1) describing all N springs
    """

    # Find indices of array that are NaN
    isnan = np.isnan(A)

    # Remove leave_mask if appropriate
    if leave_mask is not None:
        isnan[leave_mask] = False

    nan_inds = np.argwhere(isnan)

    # Create 4 or 8 element array of indices of surrounding points
    # [[-1, -1], [-1, 0], ..., [1, 1]]
    if include_diagonals:
        adjacent_pts = [[x[0] - 1, x[1] - 1]
                        for x in np.ndindex(3, 3) if x != (1, 1)]
    else:
        adjacent_pts = [[-1, 0], [0, -1], [1, 0], [0, 1]]

    N_adj = len(adjacent_pts)

    # Create two arrays.
    # 1) start_inds (x0, y0): 4 or 8 copies of the indices of NaN points
    # 2) end_inds (x1s, y1s): indices of the 4 or 8 surrounding points
    start_inds = np.concatenate(
        [nan_inds for i in range(N_adj)], axis=0)
    end_inds = np.concatenate(
        [nan_inds + xy for xy in adjacent_pts], axis=0)

    # Create N x 4 array of (x0, y0, x1, y1)
    springs = np.concatenate((start_inds, end_inds), axis=1)

    # Remove any springs beyond the buondary
    outside = np.any(np.concatenate(
        [springs[:, np.r_[0, 2]] < 0, springs[:, np.r_[0, 2]] > A.shape[0] - 1,
         springs[:, np.r_[1, 3]] < 0, springs[:, np.r_[1, 3]] > A.shape[1] - 1],
        axis=1), axis=1)
    springs = springs[~outside]

    # Create a set of lines where each set is itself a set of two elements
    # (x0, y0), (x1, y1)
    # By virtue of using a set, which has no order, we can remove duplicates
    # Need to use frozenset within the set comprehension because standard
    # sets are not hashable
    springs = {frozenset([(v[0], v[1]), (v[2], v[3])]) for v in springs}

    # Convert springs back to numpy array
    springs = np.array([np.concatenate(list(spring)) for spring in springs])

    # Show all springs if desired
    if plot_springs:
        Nr, Nc = A.shape
        X, Y = np.mgrid[:Nr, :Nc]
        fig, ax = plt.subplots()
        ax.scatter(X.flatten(), Y.flatten(), 100 * isnan, c='k')
        ax.set(xlim=(X.min(), X.max()), ylim=(Y.min(), Y.max()))

        for (x0, y0, x1, y1) in springs:
            ax.plot((x0, x1), (y0, y1), color='k', alpha=0.3, lw=6)

    return springs


def create_matrix_lhs_rhs(A, springs, nan_inds, XY_tuple=None):
    """See documentation for inpaint_nans"""

    # Preallocate output (A_mat is LHS, B_mat is RHS)
    N_nans = nan_inds.shape[0]
    N_springs = springs.shape[0]
    A_mat = lil_matrix((N_springs, N_nans))
    B_mat = np.zeros(N_springs)

    # Read in X and Y or set to evenly spaced grid
    if XY_tuple is None:
        X, Y = np.mgrid[:A.shape[0], :A.shape[1]]
    else:
        X, Y = XY_tuple

    # Create dict that contains (x, y) as key with associated index as value
    coeff_ind_dict = {tuple(row): i for i, row in enumerate(nan_inds)}

    # Each row of A_mat and B_mat corresponds to one spring
    for i, (x0, y0, x1, y1) in enumerate(springs):
        A0, A1 = A[x0, y0], A[x1, y1]

        # Calculate spring constant of spring (really dist between pts)
        dist = np.hypot(X[x1, y1] - X[x0, y0], Y[x1, y1] - Y[x0, y0])

        if np.isnan(A0):
            coeff_ind = coeff_ind_dict[x0, y0]
            A_mat[i, coeff_ind] = 1 / dist
        else:
            B_mat[i] = A0 / dist

        if np.isnan(A1):
            coeff_ind = coeff_ind_dict[x1, y1]
            A_mat[i, coeff_ind] = -1 / dist
        else:
            B_mat[i] = -A1 / dist

    return A_mat, B_mat


def inpaint_nans(A, XY_tuple=None, leave_mask=None, springs=None,
                 include_diagonals=False, update_progress=False):
    """
    Interpolate and extrapolate to cover NaNs within a 2D array

    Based on inpaint_nans.m from MatLab File Exchange by John D'Errico
    Uses method 4 (the springs method) but with option to specify distances

    Inputs
    ------
    A : 2D array with Nans
    XY_tuple : tuple of two 2D arrays
        Arrays are same shape as A and contain x and y coordinates
        If XY_tuple is not given, then evenly spaced grid is assumed
    leave_mask : 2D array
        True in spots where data are not to be solved for
    springs : output from get_springs
        This argument allows for specifying springs rather than recalculating
    include_diagonals : bool
        If True, account for all 8 adjacent values, not 4
    update_progress : bool
        If True, print steps of calculation

    Returns
    -------
    A : 2D array without NaNs
    """

    def print_progress(step):
        if update_progress:
            print(step, flush=True)

    # Copy A just in case
    A = A.copy()

    # Get N_springs x 4 array describing (x0, y0, x1, y1) of each spring
    print_progress('Progress of inpaint_nans:')

    if springs is None:
        print_progress('    Calculating springs')
        springs = get_springs(A, leave_mask, plot_springs=False,
                              include_diagonals=include_diagonals)

    # Get (row, column) indices of NaN locations
    print_progress('    Finding NaNs')
    nan_inds = np.argwhere(np.isnan(A))

    # Determine matrix based on the springs and values in u
    print_progress('    Setting up matrix equation')
    A_mat, B_mat = create_matrix_lhs_rhs(A, springs, nan_inds)

    print_progress('    Solving matrix equation')
    # Solve matrix equation in least-squares sense
    nan_replace = lsqr(A_mat, B_mat)[0]

    print_progress('    Finishing')
    # Replace NaNs with solution
    # Not sure why I have to have to invert the sign
    A[np.isnan(A)] = -nan_replace

    # Replace leave_mask with nan
    A[leave_mask] = np.nan

    return A.copy()


def inpaint_nan_example():
    """Example of inpaint_nans with unevenly spaced x and y grids and an
    island left in"""

    # Create x and y vectors that telescope out from the centre
    # _c and _f are centre and faces of cells, respectively
    dx = telescope_centre_n(1, 0.02, 0.5, 0.01, 20)
    x_f = np.insert(np.cumsum(dx), 0, 0)
    y_f = x_f
    X_f, Y_f = np.meshgrid(x_f, y_f)
    x_c, y_c = (x_f[1:] + x_f[:-1]) / 2, (y_f[1:] + y_f[:-1]) / 2
    X_c, Y_c = np.meshgrid(x_c, y_c)

    # Create a 2D array to work with and then add a bunch of NaNs
    z0 = np.exp(-(X_c - 0.5)**2 + -(Y_c - 0.5)**2)
    z0_start = z0.copy()
    z0[3:5, :] = np.nan
    z0[0:3, 0:3] = np.nan
    z0[3:8, 5:11] = np.nan
    z0[17:20, 7:9] = np.nan

    # Create an island near the bottom left to leave out
    leave_mask = np.zeros_like(z0).astype(bool)
    leave_mask[1:3, 1:3] = np.nan

    # Calculate the replacement values
    z_fixed = inpaint_nans(
        z0, (X_c, Y_c), leave_mask, include_diagonals=False,
        update_progress=True)

    # Just the values that have been calculated
    z_new = z_fixed.copy()
    z_new[np.isfinite(z0)] = np.nan

    # Pcolor plot
    vmin, vmax = minmax(z0_start)
    pcol_opts = dict(vmin=vmin, vmax=vmax)
    fig, (ax1, ax2) = plt.subplots(
        ncols=2, sharex=True, sharey=True, figsize=(12, 6))
    cax = ax1.pcolormesh(X_f, Y_f, ma.masked_invalid(z0), **pcol_opts)
    ax2.pcolormesh(X_f, Y_f, ma.masked_invalid(z_fixed), **pcol_opts)
    ax2.scatter(X_c.flatten(), Y_c.flatten(), 10 * np.isnan(z0).flatten())
    ax2.scatter(X_c[leave_mask], Y_c[leave_mask], 30, 'g')
    leave_mask[1:3, 1:3] = np.nan
    ax1.set(xlim=minmax(x_f), ylim=minmax(y_f))
    fig.colorbar(cax)

    # 3D plot
    # fig = plt.figure(figsize=(12, 6))
    # ax1 = fig.add_subplot(121, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')
    # rstride, cstride = 10, 10
    # ax1.plot_surface(X_c, Y_c, z0_start, rstride=rstride, cstride=rstride)
    # ax2.plot_surface(X_c, Y_c, z_fixed, rstride=rstride, cstride=cstride, color='r')
    # ax2.plot_surface(X_c, Y_c, z_new, rstride=rstride, cstride=cstride)


def bin_2d_transect(x, y, Z, x_out, y_out):
    """Bin transect Z(x, y), where x can be irregular

    Inputs
    ------
    x, y : 1D arrays
        x can be irregular, y cannot
    Z : 2D array
        Data at each point x, y. May be masked array
    x_out, y_out : 1D arrays
        Edges of grid on which to bin Z

    Returns
    -------
    Z_out : 2D array
        Shape (len(x_out) - 1, len(y_out) - 1)
    """
    # Preallocate result
    Nx, Ny = x_out.size - 1, y_out.size - 1
    Z_out = np.full((Nx, Ny), np.nan)

    filterwarnings('ignore', '.*Mean of empty slice*.')

    # Using loop for simplicity
    for i, j in np.ndindex(Nx, Ny):
        in_x_bin = np.logical_and(x > x_out[i], x < x_out[i + 1])
        in_y_bin = np.logical_and(y > y_out[j], y < y_out[j + 1])

        Z_in_bin = Z[in_y_bin, in_x_bin]
        Z_out[i, j] = np.nanmean(ma.filled(Z_in_bin, np.nan))

    if ma.isMA(Z):
        Z_out = ma.masked_invalid(Z_out)

    return Z_out


def fill_gaps_in_2d_transect_once(Z):
    was_masked = ma.isMA(Z)

    # Ensure array has NaNs, not mask
    Z = ma.filled(Z, np.nan).copy()

    fill_values = np.nanmean(
        np.dstack((np.roll(Z, 1, axis=1), np.roll(Z, -1, axis=1))), axis=2)

    inds_to_fill = np.logical_and(np.isnan(Z), ~np.isnan(fill_values))

    Z[inds_to_fill] = fill_values[inds_to_fill]

    if was_masked:
        Z = ma.masked_invalid(Z)

    return Z


def fill_gaps_in_2d_transect(Z, n=1):
    """Clean output of bin_2d_transect by filling in gaps laterally

    If there is a single column of data missing, this function fills it

    Inputs
    ------
    Z : 2D array
       Array with gaps to fill
    n : int
       Width of gap that will be filled (ish).
    """

    for i in range(n):
        Z = fill_gaps_in_2d_transect_once(Z)

    return Z

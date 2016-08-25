import copy
import numpy as np
import numpy.ma as ma
from numpy import logical_and as l_and
from numpy import cos, pi, atleast_1d
from sympy import Symbol, nsolve
from scipy.ndimage import label
from scipy.stats import mode
from MyNumpyTools import nan_or_masked


def estimate_cell_edges(x):
    """Convert one-dimensional vector x of size n into n + 1, where the input
    describes the centres of the cells, and the output is an estimate of the
    edges of the cell"""
    # centres (with extra centres padded at the ends by linear interpolation)
    dx = ma.diff(x)
    x_c = ma.hstack((x[0] - atleast_1d(dx[0]), x,
                     x[-1] + atleast_1d(dx[-1])))
    # _f is notation from MITgcm (implies faces)
    x_f = (x_c[1:] + x_c[:-1])/2
    dx_c = np.diff(x_c)

    # Catch nan or masked values and estimate edge using dx from previous or
    # next cell
    nan_before = ma.where(
        ma.logical_and(nan_or_masked(x_f[:-1]), ~nan_or_masked(x_f[1:])))[0]
    nan_after = ma.where(
        ma.logical_and(~nan_or_masked(x_f[:-1]), nan_or_masked(x_f[1:])))[0]

    x_f[nan_before] = x_f[nan_before + 1] - dx_c[nan_before + 1]
    x_f[nan_after + 1] = x_f[nan_after] + dx_c[nan_after]

    return x_f


def estimate_cell_edges_2d(X, columns=True, rows=False):
    """Runs estimate_cell_edges on every column or row as specified"""

    both_axes = columns and rows
    transpose = rows and not columns

    # Transpose input if we want to work on rows
    X = X.T if transpose else X

    edges = ma.masked_invalid(np.full((X.shape[0] + 1, X.shape[1]), np.nan))
    for i, column in enumerate(X.T):
        edges[:, i] = estimate_cell_edges(column)

    if both_axes:
        edges = ma.append(edges, np.full((edges.shape[0], 1), np.nan), axis=1)
        for i, row in enumerate(edges):
            edges[i, :] = estimate_cell_edges(row[:-1])

    # Transpose output back to original shape if necessary
    edges = edges.T if transpose else edges

    return edges


def telescope_right(dist, dx_min, dx_const, tel_frac):
    """create dx vector that expands with distance

    dx (or dy) values sum to dist
    for 0 < x < dx_const, dx is constant
    for x > dx_const, dx increases by tel_frac each time"""

    dx_sum = 0
    dx = np.array([dx_min, ])

    while dx_sum < dist:
        if dx_sum < dx_const:
            new_dx = dx_min
        else:
            new_dx = dx[-1]*tel_frac

        dx = np.append(dx, new_dx)
        dx_sum = dx_sum + new_dx

    return dx


def telescope_centre(dist, dx_min, x_centre, dx_const, tel_frac):
    """create dx vector that expands with distance each side of a central point

    dx (or dy) values sum to dist
    for (-dx_const < x - x_centre < dx_const), dx is constant
    for | x - x_centre | > dx_const, dx increases by tel_frac each time
    """

    dist_W, dist_E = x_centre, dist - x_centre
    dx_E = telescope_right(dist_E, dx_min, dx_const, tel_frac)
    dx_W = telescope_right(dist_W, dx_min, dx_const, tel_frac)

    dx = np.hstack((dx_W[::-1], dx_E))

    return dx


def get_r_and_n(dist_W, dist_E, dx_min, n_tel, init_r):
    """Numerically solve for the nW, nE, and r

    Sum of geometric sequence in standard form:
    a + a*r + a*r^2 + ... + a*r^(n-1) = a(1 - r^n)/(1 - r)
    However, I want to start at a*r^1, not a*r^0
    a*r + a*r^2 + ... + a*r^(n-1) = a((1 - r^n)/(1 - r) - 1)
    """
    #
    # Define the three unknowns
    nE = Symbol('nE')
    nW = Symbol('nW')
    r = Symbol('r')
    #
    # Equations to be solved
    eqns = [(dist_W/dx_min)*(1 - r) + r**(nW+1) - r,
            (dist_E/dx_min)*(1 - r) + r**(nE+1) - r,
            nW + nE - n_tel]

    # Put arguments in list and provide an initial guess
    args = [nW, nE, r]
    total_log_dist_sum = np.log(dist_W) + np.log(dist_E)
    nW_guess = np.log(dist_W)/total_log_dist_sum*n_tel
    nE_guess = np.log(dist_E)/total_log_dist_sum*n_tel

    # Find the solution numerically

    # The initial guess is pretty important, especially init_r
    # Want to ensure we don't get r = 1, which is the trivial but useless soln
    # This ends up being suprisingly robust

    r_found = False
    while not r_found:
        try:
            out = nsolve(eqns, args, [nW_guess, nE_guess, init_r])
            # Check for trivial soln (comes about from init_r being too low)
            if float(out[2]) == 1.0:
                # Over compensate by setting init_r far too high and starting
                # again
                init_r = 5
            else:
                r_found = True
        except ValueError:
            # Reduce r (if r = 1.5, set it to 1.25, then 1.125, etc until
            # solution is found)
            init_r = 1 + (init_r - 1)/2

    nW, nE, r = out
    nW, nE = [np.round(float(X)).astype(int) for X in [nW, nE]]
    r = float(r)
    return nW, nE, r


def telescope_centre_n(dist, dx_min, x_centre, dx_const, n, init_r=None,
                       suppress_printing=False):
    """Create dx vector that expands with distance each side of a central point

    - dx (or dy) values sum to dist
    - for (-dx_const < x - x_centre < dx_const), dx is constant
    - for | x - x_centre | > dx_const, dx increases by exponentially by a
      certain value such that the output vector has n elements

    Inputs
    ------
    dist : float
        total distance
    dx_min : float
        smallest value of dx, and value of dx inside dx_const radius
    x_centre : float
        location about which dx is symmetrical
    dx_const : float
        distance either side of x_centre to keep dx constant
    n : int
        size of output dx vector
    init_r : float
        initial guess for the value by which to telescope (e.g., for 10%
        increase, init_r would be 1.1)
        Specify a reasonable overestimate if solver does not converge
    suppress_printing : bool
        Whether to summarise result
    """
    # Number of points with constant dx
    nM = 2*dx_const//dx_min
    # Number of points total that are either side of constant dx region
    n_tel = int(n - nM)

    # Distances east and west of constant dx region
    dist_W = x_centre - dx_min*nM/2
    dist_E = dist - (x_centre + dx_min*nM/2)

    # Calculate telescoping increase (e.g. 1.02 for 2% increase per grid cell)
    # and associated nW and nE
    init_r = 2 if init_r is None else init_r
    nW, nE, r = get_r_and_n(dist_W, dist_E, dx_min, n_tel, init_r)

    # Create the pieces of the overall dx vector
    dx_W = dx_min*(r**np.r_[1:nW+1])[::-1]
    dx_E = dx_min*(r**np.r_[1:nE+1])
    dx_M = np.full(int(nM), float(dx_min))

    # Scale dx_W and dx_E to exactly fit dist
    dx_W = dx_W*(dist_W/dx_W.sum())
    dx_E = dx_E*(dist_E/dx_E.sum())

    # Put together the vector and ensure exactly the required size
    dx = np.hstack((dx_W, dx_M, dx_E))

    if not suppress_printing:
        print('Creating telescoping grid with increase of', end=' ')
        print('{0:3.2f}'.format((r-1)*100) + '%')

    return dx


def remove_unconnected_basins(depth):
    """Convert any 'lakes' to a sea level of 0

    Inputs
    ------
    depth : 2D array
        Depth below sea level (i.e. positive values are underwater)
    """
    is_masked = ma.isMaskedArray(depth)
    label_vals = label(ma.filled(depth, 0))[0]
    label_mode = mode(label_vals, axis=None)[0]
    depth[label_vals != label_mode] = 0

    if is_masked:
        depth = ma.masked_less_equal(depth, 0)
    return depth


def smooth_bump(x, H, ends=0.2, mid=0.2):
    """
    Create a smooth bump of height H with flat bits in the middle and at the
    ends. ends, and mids are fractions of x containing each of the parts
    of the bump::
        
                  .-------.
                 /         \\
        ........-           -.........
        |  end |   | mid |   |  end  |
    
    The bumps are half of a sinusoid
    """
    d = x[-1]
    # Determine indices of each of the 5 parts of the bump
    # Every element of x belongs to one and only one

    # start_inds = x <= d*ends
    lbump_inds = l_and(x > d*ends, x <= d/2 - d*mid/2)
    mid_inds = l_and(x <= d/2 + d*mid/2, x > d/2 - d*mid/2)
    rbump_inds = l_and(x <= x[-1] - d*ends, x > d/2 + d*mid/2)
    # end_inds = x >= x[-1] - d*ends

    xL = x[lbump_inds]
    xR = x[rbump_inds]
    bumpL = H/2 + H/2*cos(pi*(xL - xL[-1])/(xL[0] - xL[-1]))
    bumpR = H/2 - H/2*cos(pi*(xR - xR[-1])/(xR[-1] - xR[0]))

    # Construct the output from individual parts
    output = np.zeros(x.shape)
    output[lbump_inds] = bumpL
    output[rbump_inds] = bumpR
    output[mid_inds] = H

    return output


def dx_to_x(x, x0=0):
    return np.insert(np.cumsum(x), 0, 0) + x0


def xf_to_xc(x):
    return (x[:-1] + x[1:])/2


def x_to_xkm(x):
    return x/1000


class Grid:
    """Grid object for use with MITgcm and other models

    Inputs
    ------
    dx : 1D array
    dy : 1D array
    dz : 1D array (optional)
    m : Basemap object
    """

    def __init__(self, dx, dy, dz=1, m=None, x0=0, y0=0):
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.calc_faces(x0, y0)
        self.cell_centred()
        self.mesh()
        self.to_kms()
        if m is not None:
            self.m = m
            self.latlon(self.m)

    def __repr__(self):
        attrs = sorted(list(self.__dict__.keys()),
                       key=lambda x:(not x.islower(), x))

        def sort_attrs(dim):
            return '  '.join([str(s) for s in attrs if dim in s.lower()])

        x_attrs = sort_attrs('x')
        y_attrs = sort_attrs('y')
        z_attrs = sort_attrs('z')
        geo_attrs = sort_attrs('lon') + sort_attrs('lat')
        attr_table = '\n'.join(
            [s for s in [x_attrs, y_attrs, z_attrs, geo_attrs]])
        return 'Grid object with attributes: \n' + attr_table

    def calc_faces(self, x0, y0):
        self.xf = dx_to_x(self.dx, x0)
        self.yf = dx_to_x(self.dy, y0)
        self.zf = dx_to_x(self.dz)

    def cell_centred(self):
        self.xc = xf_to_xc(self.xf)
        self.yc = xf_to_xc(self.yf)
        self.zc = xf_to_xc(self.zf)

    def mesh(self):
        self.Xc, self.Yc = np.meshgrid(self.xc, self.yc, indexing='ij')
        self.Xf, self.Yf = np.meshgrid(self.xf, self.yf, indexing='ij')
        self.dX, self.dY = np.meshgrid(self.dx, self.dy, indexing='ij')

    def to_kms(self):
        """Hack to get all grid objects in km as well"""
        for key in list(self.__dict__.keys()):
            if 'z' in key:
                continue
            exec('self.' + key + '_km = self.' + key + '/1000')

    def latlon(self, m):
        self.LONc, self.LATc = self.m(self.Xc, self.Yc, inverse=True)
        self.LONf, self.LATf = self.m(self.Xf, self.Yf, inverse=True)

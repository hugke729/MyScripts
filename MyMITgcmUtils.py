from warnings import filterwarnings
import numpy as np
import numpy.ma as ma
import os
import sys
from subprocess import check_output
from scipy import ndimage as nd
from scipy.interpolate import RegularGridInterpolator as rgi
from MITgcmutils import rdmds
from MyGrids import Grid
from MyInterp import nan_gaussian_filter


def write_for_mitgcm(filename_in, array_in, prec=64):
    """Write binary file to use as input for MITgcm

    Inputs
    ------
    filename_in: str
        Name of binary file
    array_in: numpy array
        Array to write to binary file with shape (Nx, Ny, Nz) or subset. This
        is opposite order to output of rdmds.
    prec: int
        Only options are 32 or 64

    Notes
    -----
    * Need to confirm whether time is before x or after z
    * Output is big endian
    """

    if prec == 32:
        array_in = array_in.astype('float32')

    # Ensure no NaN or masked values
    if ma.isMaskedArray(array_in):
        array_in = ma.filled(array_in, np.nan)

    if np.any(np.isnan(array_in)):
        array_in = remove_nans_laterally(array_in)

    #  Flatten array_in, convert to Fortran order, and ensure correct byte order
    array_in = array_in.reshape(array_in.size, order='F')
    if sys.byteorder == 'little':
        array_in = array_in.byteswap()

    with open(filename_in, 'wb') as f:
        f.write(array_in)


def remove_nans_laterally(array, inverted_dimensions=False):
    """Get rid of NaNs by copying nearest adjacent value or column for 3D

    Assumes first two dimensions of array are x and y, unless
    `inverted_dimensions` is `True`, in which case last two dimensions are
    y and x.

    Idea from stackoverflow.com/questions/5551286/filling-gaps-in-a-numpy-array
    """

    if array.ndim == 1:
        nans = np.isnan(array)
        inds = nd.distance_transform_edt(
            nans, return_distances=False, return_indices=True)

        array = array[tuple(inds)]
        return array

    array = array.T if inverted_dimensions else array

    if array.ndim == 2:
        new_dim_added = True
        array = array[..., np.newaxis]
    else:
        new_dim_added = False

    for i, level in enumerate(np.rollaxis(array, 2)):
        nans = np.isnan(array[..., i])
        inds = nd.distance_transform_edt(
            nans, return_distances=False, return_indices=True)

        array[..., i] = array[..., i][tuple(inds)]

    # Ensure returned array in same form as input
    array = array.squeeze() if new_dim_added else array
    array = array.T if inverted_dimensions else array

    return array


def xc_to_xg(xc):
    """Convert cell centers to cell edges
    returns xg, which has size len(xc) + 1"""
    dx = np.diff(xc)
    xg = dx_to_xg(dx)
    return xg


def dx_to_xg(dx):
    """Convert dx to absolute x at cell edges
    returns xg, which has size len(dx) + 1"""
    xg = np.insert(np.cumsum(dx), 0, 0)
    return xg


def create_size_header(code_dir, n):
    """Create SIZE.h in ../code/ directory

    Inputs
    ------
    code_dir : str
        full path to model's code directory
    n : dict
        key, value pairs for the variable values in size.h

    SIZE_start.h and SIZE_end.h are files in ~/mitgcm/other
    They contain the start and end of SIZE.h that remains constant.
    Only need to create middle bit

    Create all the lines of the form
    '     &           nPx =    1,'
    and write to SIZE_middle.h
    """

    # Ensure code_dir ends with slash
    if not code_dir.endswith('/'):
        code_dir += '/'

    with open(code_dir + 'SIZE_middle.h', 'w') as f:
        for key, value in n.items():
            if key not in ['x', 'y', 'z']:
                line_start = '     &           '
                line_end = key.ljust(3) + ' = ' + str(value).rjust(3) + ',\n'
                f.write(line_start + line_end)

    start_end_loc = '/home/hugke729/mitgcm/other/'
    # Concatenate files_in into code/SIZE.h
    files_in = [start_end_loc + 'SIZE_start.h',
                code_dir + 'SIZE_middle.h',
                start_end_loc + 'SIZE_end.h']
    with open(code_dir + 'SIZE.h', 'w') as outfile:
        for file_i in files_in:
            with open(file_i, 'r') as infile:
                outfile.write(infile.read())

    # Remove temporary middle file
    os.remove(code_dir + 'SIZE_middle.h')


def get_run_settings(output_file):
    """Pull out important run settings from the model's standard output, which
    needs to be redirected to output_file
    Use linux binaries to find what I want. I find these easier than using
    python to work with files"""

    def remove_start_end(line, leave_start=False):
        """Remove start of line, which is '(PID.TID 0000.0001) >' or similar
        Remove end of line, which is ,\\n
        Possibly no comma, but that doesn't matter"""
        # 21 characters gets rid of PID...
        if not leave_start:
            line = line[21:]
        line = line.replace(',', '')
        line = line.replace('\n', '')
        line = line.replace(' ', '')  # Remove any whitespace
        return line

    def grep(pattern, file_in, prepend='', append=''):
        """Use grep to find the line with what I want, then get just the
        useful middle part of the line
        Changes to default pattern can be made by adding prepend and append"""
        pattern = prepend + pattern + append
        cmd = 'grep' + " '" + pattern + "' " + file_in
        outstr = check_output(cmd, shell=True).decode()
        outstr = remove_start_end(outstr, leave_start=file_in.endswith('data'))
        return outstr

    def add_to_dict(key, D, prepend='', append='', file_in=None):
        if file_in is None:
            file_in = D['file_in']
        try:
            outstr = grep(key + '=', file_in, prepend, append)
            exec(outstr)  # Create variable named key
            exec("D['" + key + "'] = " + key)
        except:
            print(key + ' did not work')

    f = output_file
    # Data file should be in same folder as output.txt
    data_file = os.path.dirname(f) + '/data'
    D = {'file_in': f}
    add_to_dict('deltaT', D)
    add_to_dict('dumpFreq', D, prepend='[^KL]')  # Ignore KLdumpFreq
    add_to_dict('nIter0', D)
    add_to_dict('nTimeSteps', D, file_in=data_file)

    # Add in value of how many iterations per dump
    D['iter_per_dump'] = int(D['dumpFreq']/D['deltaT'])
    # Add in how many dumps in total
    D['nDumps'] = int(D['nTimeSteps']*D['deltaT']/D['dumpFreq']) + 1
    D['dumpVec'] = np.linspace(D['nIter0'], D['nTimeSteps'], D['nDumps'],
                               endpoint=True).astype(int)

    # Add in some useful conversions
    D['deltaT_sec'] = D['deltaT']/1
    D['deltaT_min'] = D['deltaT']/60
    D['deltaT_hr'] = D['deltaT']/3600
    D['deltaT_day'] = D['deltaT']/86400
    D['dumpFreq_sec'] = D['dumpFreq']/1
    D['dumpFreq_min'] = D['dumpFreq']/60
    D['dumpFreq_hr'] = D['dumpFreq']/3600
    D['dumpFreq_day'] = D['dumpFreq']/86400
    D['dumpVec_hr'] = D['dumpVec']*D['deltaT']/3600
    D['dumpVec_day'] = D['dumpVec']*D['deltaT']/86400
    return D


def get_grid(run_dir, x0=0, y0=0, hFacs=True, squeeze_hfacs=True):
    """Create a Grid object from model's output grid files

    Inputs
    ------
    run_dir: str
        Full path to model's run directory
    x0, y0: floats
        Distances from origin
    hfacs: bool
        Whether to read in hFacs
    squeeze_hfacs: bool
        Whether to remove singleton dimensions from hfacs
    """
    dx = rdmds(run_dir + 'DXG*')[0, :]
    dy = rdmds(run_dir + 'DYG*')[:, 0]
    dz = rdmds(run_dir + 'DRF*')[:, 0]
    added_attrs = dict(depth=rdmds(run_dir + 'Depth*').squeeze())

    if hFacs:
        for k in ['S', 'W', 'C']:
            tmp = rdmds(run_dir + 'hFac' + k)
            added_attrs['hFac' + k] = tmp.squeeze() if squeeze_hfacs else tmp

    g = Grid(dx, dy, dz, x0=x0, y0=y0, added_attrs=added_attrs)

    return g


def mask_mitgcm_output(output):
    """Return a masked array copy of output with only values in water

    Not 100% gauranteed to work perfectly, but should work most of the time"""
    output = ma.masked_where(output == np.roll(output, -1, axis=0), output)
    return output


def add_border_values(X):
    """Add a border of values by copying the first and last lines in the
    array in each dimension and prepending/appending

    Result is a new array increased by 2 cells in each dimension

    This is useful for accounting for region between centre and face of cells
    at the edge of a domain

    Input
    -----
    X : numpy array
        Dimensions should be ordered (z, y, x)
    """
    if X.ndim == 1:
        tmp_X = np.zeros(X.size + 2)
        tmp_X[1:-1] = X
        tmp_X[0], tmp_X[-1] = X[0], X[-1]

    elif X.ndim == 2:
        shape = np.array(X.shape) + np.r_[2, 2]
        tmp_X = np.zeros(shape)
        tmp_X[1:-1, 1:-1] = X
        tmp_X[0, :], tmp_X[-1, :] = tmp_X[1, :], tmp_X[-2, :]
        tmp_X[:, 0], tmp_X[:, -1] = tmp_X[:, 1], tmp_X[:, -2]

    elif X.ndim == 3:
        shape = np.array(X.shape) + np.r_[2, 2, 2]
        tmp_X = np.zeros(shape)
        tmp_X[1:-1, 1:-1, 1:-1] = X
        tmp_X[0, :, :], tmp_X[-1, :, :] = tmp_X[1, :, :], tmp_X[-2, :, :]
        tmp_X[:, 0, :], tmp_X[:, -1, :] = tmp_X[:, 1, :], tmp_X[:, -2, :]
        tmp_X[:, :, 0], tmp_X[:, :, -1] = tmp_X[:, :, 1], tmp_X[:, :, -2]

    return tmp_X


def interpolate_output_to_new_grid(run_dir, last_iter, new_grid,
                                   get_grid_args=dict()):
    """
    Takes outputs U, V, T, S, Eta and interpolates them onto new grid

    Assumes both grids are 3D and Cartesian.

    Results for 2D aren't quite right. Interpolation falls back to nearest
    neighbour where it should be linear.

    Inputs
    ------
    run_dir: str
        Directory containing output
    last_iter: int
        The number in the output filenames that are to be interpolated
        For example, for T.00000360000.data, last_iter is 36000
    new_grid: Grid instance
        Grid from MyGrids.Grid
    get_grid_args: dict
        Arguments to pass to get_grid
        `g_in = get_grid(run_dir, squeeze_hfacs=False, **get_grid_args)`

    Returns
    -------
    all_outputs: dict
        Contains U, V, T, S, and Eta on new grid. Shape of these arrays
        are (Nz × Ny × Nx) or (Ny × Nx) for Eta

    Notes
    -----
    Not set up to work with OBCS (hFacs are not dealt with correctly)
    """
    # filterwarnings('ignore', '.*invalid value encountered in true_divide')
    # filterwarnings('ignore', '.*invalid value encountered in less_equal')

    # Helper function
    def get_coord(grid, coord_str):
        """get_coord(g, 'xc') returns g.xc or None if g has no attribute xc
        Remove extra value for xf and yf, since that's what's need in outer
        function"""
        try:
            out = getattr(grid, coord_str)
            if coord_str in ['xf', 'yf']:
                out = out[:-1]
            return out.copy()
        except TypeError:
            return None

    # Input and output grids
    run_dir = os.path.normpath(run_dir) + os.sep
    g_in = get_grid(run_dir, squeeze_hfacs=False, **get_grid_args)
    g_out = new_grid

    coord_sys = dict(
        U=('xf', 'yc', 'zc', 'hFacW'), V=('xc', 'yf', 'zc', 'hFacS'),
        T=('xc', 'yc', 'zc', 'hFacC'), S=('xc', 'yc', 'zc', 'hFacC'),
        Eta=('xc', 'yc', None, 'hFacC'))

    # Preallocate dict that is returned
    all_outputs = {}

    for k, (x, y, z, h) in coord_sys.items():
        threeD = False if k is 'Eta' else True

        # Read in all grids for current quantity
        xi, yi, zi, hi = [get_coord(g_in, q) for q in (x, y, z, h)]
        hi = hi[0, ...] if not threeD else hi

        # Read actual output
        fname = run_dir + k + '*'
        quantity = rdmds(fname, last_iter)

        # Convert zeros to NaN for values that aren't water
        # This is really important for T and S, where the average of say 35
        # and 0 is unphysical. For U and V, it's not as important but still
        # worthwhile
        quantity[hi == 0] = np.nan

        # Smooth at each depth level before interpolation. Helps reduce large
        # divergences
        # Update: 6/3/17. Try without smoothing
        gf_opts = dict(sigma=0, keep_nans=False, gf_kwargs=dict(truncate=8))
        if threeD:
            for i, level in enumerate(quantity):
                quantity[i, ...] = nan_gaussian_filter(level, **gf_opts)
        else:
            quantity = nan_gaussian_filter(quantity, **gf_opts)

        # Add a border around output to avoid problems with regions between
        # the centre of the first and last cells in a given dimension and the
        # edge of the domain
        quantity = add_border_values(quantity)

        # Add in associated values to x, y, z
        xp2 = np.r_[xi[0] - g_in.dx[0], xi, xi[-1] + g_in.dx[-1]]
        yp2 = np.r_[yi[0] - g_in.dy[0], yi, yi[-1] + g_in.dy[-1]]
        zp2 = np.r_[zi[0] - g_in.dz[0], zi, zi[-1] + g_in.dz[-1]] if threeD else None

        # Grid associated with added border
        pts_in = (zp2, yp2, xp2) if threeD else (yp2, xp2)

        # Overall interpolation will be combo of linear and nearest neighbour
        # to get the best of both
        interp_input = dict(points=pts_in, values=quantity,
                            bounds_error=False, fill_value=None)
        f_lin = rgi(method='linear', **interp_input)
        f_near = rgi(method='nearest', **interp_input)

        # Output grids
        xo, yo, zo = [get_coord(g_out, q) for q in (x, y, z)]
        if threeD:
            Zo, Yo, Xo = np.meshgrid(zo, yo, xo, indexing='ij')
        else:
            Yo, Xo = np.meshgrid(yo, xo, indexing='ij')

        pts_out = (Zo, Yo, Xo) if threeD else (Yo, Xo)

        # Linear interpolate to start with, then fill in bits near boundaries
        # with nearest neighbour interpolation, then fill everything else
        lin_out = f_lin(pts_out)
        near_out = f_near(pts_out)

        lin_out_is_nan = np.isnan(lin_out)
        lin_out[lin_out_is_nan] = near_out[lin_out_is_nan]

        # Fill any remaining gaps
        lin_out = remove_nans_laterally(lin_out, inverted_dimensions=True)

        # For completely empty levels, copy the level above
        if threeD:
            levels_to_copy = np.where(np.all(np.isnan(lin_out), axis=(1, 2)))[0]
            for level in levels_to_copy:
                lin_out[level, ...] = lin_out[level - 1, ...]

        all_outputs[k] = lin_out

    return all_outputs


def T_to_sigma(T, tAlpha=2e-4, T_0=14.9, rho_0=1026):
    """Convert model temperature to density"""
    tAlpha = 2e-4
    rho_0 = 1026
    T_0 = 14.9
    sigma_0 = rho_0 - 1000
    return sigma_0 + tAlpha*rho_0*(T_0 - T)


if __name__ == '__main__':
    run_dir = '/home/hugke729/mitgcm/test_cases/hfacs/run/'
    g = get_grid(run_dir)
    dx_new = np.ones(120)*200/2
    dz_new = np.ones(41)*5
    gout = Grid(dx_new, g.dy, dz_new)

    T = rdmds(run_dir + 'T*', [100]).squeeze()
    U = rdmds(run_dir + 'U*', [100]).squeeze()
    Eta = rdmds(run_dir + 'Eta*', [100]).squeeze()
    out = interpolate_output_to_new_grid(run_dir, 100, gout)

    fig, axs = plt.subplots(ncols=2, sharey=True, sharex=True)
    cax = axs[0].pcolormesh(g.xf, g.zf, ma.masked_equal(T, 0), cmap='jet')
    clim = cax.get_clim()
    axs[1].pcolormesh(gout.xf, gout.zf,
                      ma.masked_invalid(out['T'].squeeze()), cmap='jet',
                      vmin=clim[0], vmax=clim[1])
    # cax = axs[0].pcolormesh(g.xf, g.zf, T, cmap='jet')
    # axs[1].pcolormesh(gout.xf, gout.zf,
    #                   ma.masked_invalid(out['T'].squeeze()), cmap='jet')
    # axs[0].plot(g.xc, Eta, 'ko-')
    # axs[1].plot(gout.xc, out['Eta'].squeeze(), 'ko-')
    fig.colorbar(cax)
    flipy()

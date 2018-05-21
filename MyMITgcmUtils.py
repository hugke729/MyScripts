import re
import numpy as np
import numpy.ma as ma
import os
import sys
from subprocess import check_output
from xarray import open_dataset, Dataset
from xarray.ufuncs import fabs
from scipy import ndimage as nd
from scipy.interpolate import RegularGridInterpolator as rgi
from MITgcmutils import rdmds
from MyGrids import Grid
from MyInterp import nan_gaussian_filter
from MyNumpyTools import shift_slice


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
    * Output is big endian
    * When writing amplitude/phase files for OBCS, the input array needs
    to be Ny x num constituents or Nx x num constituents. Note lack of
    Nz
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
    """Create SIZE.h in code directory

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


def create_data_diagnostics(input_dir, code_dir, *diags, num_diag_scale=2):
    """Create data.diagnostics file and DIAGNOSTICS_SIZE.h

    Although this file is straightforward, it's easy to forget to change
    details and induce errors, so we want to write it with minimal boiler-
    plate code

    Inputs
    ------
    input_dir: str
        Input directory in which to save `data.diagnostics`
    code_dir: str
        Code directory in which to save `DIAGNOSTICS_SIZE.h`
    diags: tuple
        | First element: list of fields
        | Second element: frequency (+ve for time-average, -ve for snapshot)
        | Third element: filename
        | Fourth element: dimensions. 2 if no z dependence else 3
        | Fourth element: time-phase
    num_diag_scale: int or real number
        Factor by which to increase num_diags in DIAGNOSTICS_SIZE.h
        Allows for some lineancy for run-time changes
        Trade-off is increased memory usage, but not a big problem for
        my purposes


    Add as many diags as desired as comma-separated tuples

    Example
    -------
    (['Ebt', 'Ebc', 'uPbt', 'uEbt', 'uPbc', 'uEbc', 'Conv'], 44640,
     'energyDiags', 2, 0)
    """

    # Keep running total of number of diagnostics for DIAGNOSTICS_SIZE.h
    num_3d_diags = 0
    num_2d_diags = 0

    if not code_dir.endswith('/'):
        code_dir += '/'

    with open(input_dir + 'data.diagnostics', 'wt') as f:
        f.write(' &diagnostics_list\n')

        for i, diag in enumerate(diags, start=1):
            flds, freq, fname, dims, time_phase = diag

            # Ensure flds is list
            if type(flds) is str:
                flds = [flds]
            diag_slice = '1' if len(flds) == 1 else '1:' + str(len(flds))
            flds_str = (''.join(['fields(', diag_slice, ',', str(i), ")='"]) +
                        "','".join(flds) + "'")
            freq_str = ''.join(['frequency(', str(i), ')=', str(freq), ','])
            phase_str = ''.join(['timephase(', str(i),')=', str(time_phase)])
            fname_str = ''.join(['filename(', str(i), ")='", fname, "'"])
            f.write('#\n')
            f.write(flds_str + '\n')
            f.write(freq_str + '\n')
            f.write(phase_str + '\n')
            f.write(fname_str + '\n')

            if dims == 3:
                num_3d_diags += len(flds)
            if dims == 2:
                num_2d_diags += len(flds)

        # Not sure if statis part is needed, but no harm adding it
        end_str = '/\n\n&diag_statis_parms\n/'
        f.write(end_str)

    num_3d_diags = int(num_diag_scale*num_3d_diags)
    num_2d_diags = int(num_diag_scale*num_2d_diags)
    num_diags_str = str(num_3d_diags) + '*Nr + ' + str(num_2d_diags)

    # Create DIAGNOSTICS_SIZE.h in code directory
    diag_start_end_dir = '/home/hugke729/mitgcm/other/'
    start_file = diag_start_end_dir + 'DIAGNOSTICS_SIZE_start.h'
    end_file = diag_start_end_dir + 'DIAGNOSTICS_SIZE_end.h'
    with open(code_dir + 'DIAGNOSTICS_SIZE.h', 'w') as f:
        with open(start_file, 'r') as start:
            f.write(start.read())

        f.write('      PARAMETER( numDiags = ' + num_diags_str + ')')
        f.write('\n')
        with open(end_file, 'r') as end:
            f.write(end.read())


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


def get_grid(run_dir, grid_filename=None, x0=0, y0=0, hFacs=True,
             squeeze_hfacs=True, xslice=np.s_[0:], yslice=np.s_[0:],
             zslice=np.s_[0:]):
    """Create a Grid object from model's output grid files

    Inputs
    ------
    run_dir: str
        Full path to model's run directory without trailing slash
    grid_filename: str (optional)
        Name of grid file if it is not grid.t001.nc or grid.nc
    x0, y0: floats
        Distances from origin
    hfacs: bool
        Whether to read in hFacs
    squeeze_hfacs: bool
        Whether to remove singleton dimensions from hfacs
    xslice, yslice, zslice: slice objects
        Allow reading of part of grid
        Not implemented for mds grid

    Implementation for netcdf grid file could use work
    """

    def hfac_mnc(W_S_or_C, squeeze_hfacs=squeeze_hfacs,
                 x=xslice, y=yslice, z=zslice):
        tmp_hfac = g['HFac' + W_S_or_C].isel(Z=zslice)
        if W_S_or_C == 'W':
            tmp_hfac = tmp_hfac.isel(Xp1=xslice)
        else:
            tmp_hfac = tmp_hfac.isel(X=xslice)
        if W_S_or_C == 'S':
            tmp_hfac = tmp_hfac.isel(Yp1=yslice)
        else:
            tmp_hfac = tmp_hfac.isel(Y=yslice)

        tmp_hfac = tmp_hfac.isel(Z=zslice)

        if squeeze_hfacs:
            return tmp_hfac.squeeze().data
        else:
            return tmp_hfac.data

    try:
        # MDS approach
        dx = rdmds(run_dir + 'DXG*')[0, xslice]
        dy = rdmds(run_dir + 'DYG*')[yslice, 0]
        dz = rdmds(run_dir + 'DRF*')[:, 0]
        added_attrs = dict(depth=rdmds(run_dir + 'Depth*').squeeze())

        if hFacs:
            for k in 'SWC':
                tmp_hfac = rdmds(run_dir + 'hFac' + k)
                if squeeze_hfacs:
                    tmp_hfac = tmp_hfac.squeeze()
                added_attrs['hFac' + k] = tmp_hfac

        g = Grid(dx, dy, dz, x0=x0, y0=y0, added_attrs=added_attrs)
    except IOError:
        # NetCDF approach
        xp1_slice = shift_slice(xslice, stop=1)
        yp1_slice = shift_slice(yslice, stop=1)

        if grid_filename is not None:
            pass
        elif os.path.exists(run_dir + '/grid.t001.nc'):
            grid_filename = '/grid.t001.nc'
        elif os.path.exists(run_dir + '/grid.nc'):
            grid_filename = '/grid.nc'
        else:
            err_msg = ('\nGrid file appears to not exist.\n'
                       'Are you using arguments correctly: '
                       'run_dir, grid_filename\n'
                       'Tried:\n'
                       'grid.t001.nc\n'
                       'grid.nc\n'
                       'and grid_filename if given as argument')
            raise OSError(err_msg)
        g = open_dataset(run_dir + '/' + grid_filename)
        depth = -g.R_low.isel(X=xslice, Y=yslice).squeeze().data

        g = g.isel(X=xslice, Y=yslice, Z=zslice)

        dx, dy, dz = [g[dim].data for dim in ['dxF', 'dyF', 'drF']]
        dx, dy = dx[0, xslice], dy[:, 0]
        added_attrs = dict(depth=depth)
        if hFacs:
            # I think this needs work
            try:
                added_attrs['hFacW'] = hfac_mnc('W', x=xp1_slice, z=zslice)
            except KeyError:
                pass

            try:
                added_attrs['hFacS'] = hfac_mnc('S', y=yp1_slice, z=zslice)
            except KeyError:
                pass

            try:
                added_attrs['hFacC'] = hfac_mnc('C', z=zslice)
            except KeyError:
                pass

        g = Grid(dx, dy, dz, x0=x0, y0=y0, added_attrs=added_attrs)

    return g


def get_xgrid(run_dir, grid_filename, x0=0, y0=0, xslice=np.s_[0:],
              yslice=np.s_[0:], zslice=np.s_[0:]):
    """Read a netcdf grid file and include a few extra useful attributes

    Inputs
    ------
    run_dir: str
        Full path to model's run directory without trailing slash
    grid_filename: str (optional)
        Name of grid file if it is not grid.t001.nc or grid.nc
    x0, y0: floats
        Distances from origin
    xslice, yslice, zslice: slice objects
        Allow reading of part of grid
    """
    g = open_dataset(run_dir + grid_filename)

    xp1_slice = shift_slice(xslice, stop=1)
    yp1_slice = shift_slice(yslice, stop=1)

    g = g.isel(X=xslice, Y=yslice, Z=zslice, Xp1=xp1_slice, Yp1=yp1_slice)

    # Simplify a few naming conventions
    g['xc'], g['yc'], g['zc'] = g.X - x0, g.Y - y0, g.Z
    g['xf'], g['yf'], g['zf'] = g.Xp1 - x0, g.Yp1 - y0, g.Zp1
    g['dx'], g['dy'] = g.dxF.isel(Y=0, drop=True), g.dyF.isel(X=0, drop=True)
    g['dz'] = g.drF
    g['Xf'], g['Yf'] = g.XG - x0, g.YG - y0
    g['Xc'], g['Yc'] = g.XC - x0, g.YC - y0
    g['Nx'], g['Ny'], g['Nz'] = [g[key].data.size for key in 'XYZ']
    g['depth'] = g.Depth

    # Add convenience attributes
    g['xc_km'], g['yc_km'] = g.xc/1e3, g.yc/1e3
    g['xf_km'], g['yf_km'] = g.xf/1e3, g.yf/1e3
    g['dx_km'], g['dy_km'] = g.dx/1e3, g.dy/1e3
    g['Xf_km'], g['Yf_km'] = g.XG/1e3, g.YG/1e3
    g['Xc_km'], g['Yc_km'] = g.XC/1e3, g.YC/1e3
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
    sigma_0 = rho_0 - 1000
    return sigma_0 + tAlpha*rho_0*(T_0 - T)


def sigma_to_T(sigma, tAlpha=2e-4, T_0=14.9, rho_0=1026):
    """Convert density to temperature for use in model"""
    return T_0 - (sigma + 1000 - rho_0)/(tAlpha*rho_0)


def rho_to_T(rho, tAlpha=2e-4, T_0=14.9, rho_0=1026):
    """Convert density to temperature for use in model"""
    return T_0 - (rho - rho_0)/(tAlpha*rho_0)


def open_simulation(filename, squeeze=True, grid_filename=None, **kwargs):
    """Wrapper around xarray.open_dataset to do simple clean up things

    Clean ups
    ---------
    1. Guess file extension
    2. Squeeze by default
    3. Rename Z coordinate to Z instead of things like Zld000030
    4. Give physical coords to Z if an appropriate grid file exists
    5. Invert Z if step 4 not applicable

    Inputs
    ------
    filename: str
        Full path to MITgcm output file
    grid_filename: str (optional)
        Name of associated grid file (needed to get z coordinate correct)
    kwargs: dict
        Arguments to pass to xarray.open_dataset
    """
    try:
        ds = open_dataset(filename, **kwargs)
    except OSError:
        std_ext = '.0000000000.t001.nc'
        if filename.find(std_ext) == -1:
            filename += std_ext

        ds = open_dataset(filename, **kwargs)

    if squeeze:
        ds = ds.squeeze()

    pattern = re.compile('Z[a-z]{2}[0-9]{6}')
    for k in ds.dims:
        if pattern.match(k) is not None:
            ds.rename({k: 'Z'}, inplace=True)

    # Try default grid_filename, input grid_filename, and if neither exist,
    # give up
    try:
        g = get_grid(os.path.dirname(filename))
        ds = ds.assign_coords(Z=-g.zc)
    except OSError:
        try:
            g = get_grid(os.path.dirname(filename), grid_filename)
            ds = ds.assign_coords(Z=-g.zc)
        except OSError:
            try:
                ds['Z'] *= -1
            except KeyError:
                pass

    return ds


def plot_3d_isosurface(X, Y, Z, Q, level, is_ordered=False, fig=None,
                       **kwargs):
    """Plot a the surface of a given level value of Q

    Assumes Q approximately decreases/increases monotonically with depth

    Inputs
    ------
    X, Y, Z: 1D or 3D arrays
        The grid associated with Q
    Q: 3D array
        The quantity from which to find the isosurface
    level: float
        The value of the isosurface
    is_ordered: bool
        Set to true if Q(X, Y, Z) not Q(Z, Y, X), which is MITgcm default
    fig: figure instance
        If no figure instance is passed in, a new one is created
    kwargs: dict
        Values to pass to ?

    Output
    ------
    fig: figure instance
    surf: contour surface artist
    """
    fig = plt.figure() if fig is None else fig
    ax = fig.gca(projection='3d')

    Z_inds = np.argmin(fabs(Q - level), axis=0)
    Z_surface = Z.data[[Z_inds]].squeeze()
    print(Z_surface.shape)

    if X.ndim == 1 and Y.ndim == 1:
        X, Y = np.meshgrid(X, Y)

    # Quick hack to avoid edge effects
    inds = np.s_[5:-5, 5:-5]
    ax.plot_surface(X[inds], Y[inds], Z_surface[inds])

    return Z_inds


def calc_bc_velocities(ds, g):
    """Baroclinic velocities from UE_VEL_C and VN_VEL_C

    Inputs
    ------
    ds: dataset
        Output from open_simulation
    g: grid object
        Output from get_xgrid
    """
    try:
        hfac = ds.UE_VEL_C.isel(T=-1).copy()
        depth = ds.isel(Z=0).UE_VEL_C.isel(T=-1).copy()
    except ValueError:
        hfac = ds.UE_VEL_C.copy()
        depth = ds.isel(Z=0).UE_VEL_C.copy()

    hfac.data = g.HFacC.data.squeeze()
    depth.data = g.depth.data.squeeze()

    dz = ds.Z.copy()
    dz.data = g.dz

    Ubt = (ds.UE_VEL_C*hfac*dz).sum('Z')/depth
    ds['Ubc'] = (ds.UE_VEL_C - Ubt).where(hfac > 0)

    Vbt = (ds.VN_VEL_C*hfac*dz).sum('Z')/depth
    ds['Vbc'] = (ds.VN_VEL_C - Vbt).where(hfac > 0)
    return ds

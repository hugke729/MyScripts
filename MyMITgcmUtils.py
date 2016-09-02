import numpy as np
import numpy.ma as ma
import os
from subprocess import check_output
from scipy import ndimage as nd
from MITgcmutils import rdmds
from MyGrids import Grid


def write_for_mitgcm(filename_in, array_in):
    import sys
    # Converts array_in to column major (aka fortran) ordering and
    # big endian format.
    # Result is written on file_in

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


def remove_nans_laterally(array):
    """Get rid of NaNs by copying nearest adjacent value or column for 3D

    Assumes first two dimensions of array are x and y

    Idea from stackoverflow.com/questions/5551286/filling-gaps-in-a-numpy-array
    """
    # Nx, Ny = array.shape[:2]

    if array.ndim == 2:
        array = array[..., np.newaxis]

    nans = np.isnan(array[..., 0])

    # for xi, yi in np.argwhere(nans):
    #     array[xi, yi, :] =

    inds = nd.distance_transform_edt(
        nans, return_distances=False, return_indices=True)

    for i, level in enumerate(np.rollaxis(array, 2)):
        array[..., i] = array[..., i][tuple(inds)]

    # Remove introduced dimension if necessary
    array = array.squeeze()

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


def get_grid(run_dir, x0=0, y0=0):
    """Create a Grid object from model's output grid files

    Inputs
    ------
    run_dir : str
        Full path to model's run directory
    x0, y0 : floats
        Distances from origin
    """
    dx = rdmds(run_dir + 'DXG*')[0, :]
    dy = rdmds(run_dir + 'DYG*')[:, 0]
    dz = rdmds(run_dir + 'DRF*')[:, 0]

    g = Grid(dx, dy, dz, x0=x0, y0=y0)
    g.depth = rdmds(run_dir + 'Depth*')

    return g

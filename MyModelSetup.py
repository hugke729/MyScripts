# Functions that I use when setting up models
from warnings import filterwarnings
import numpy as np
import numpy.ma as ma
import pickle
import matplotlib.pyplot as plt
from matplotlib.path import Path
from mpl_toolkits.basemap import Basemap
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import convolve
from scipy.ndimage.filters import gaussian_filter
import xarray as xr
from seawater.eos80 import ptmp
from MyInterp import interp_weights, interpolate, get_springs, inpaint_nans
from MyNumpyTools import change_wrap, fillnan_pad, uneven_2D_convolve, cosd


def setup_map(model_name, regenerate=False):
    """Create Basemap object of fixed dimensions

    Inputs
    ------
    model_name : str
        string corresponding to the model run's name
    regenerate : bool
        whether to regenerate map or use pickled version

    Returns
    -------
    m : Basemap object
    x0, x1, y0, y1 : tuple
        Limits in metres to refine map

    Note that I create a map and then zoom in, rather than trying to make
    a map with exact dimensions I want. The former is much easier.
    """
    map_dir = '/home/hugke729/PhD/Maps/'

    if model_name == 'pilot':
        fname = map_dir + 'model_pilot.p'

        if regenerate:
            m = Basemap(
                projection='aeqd',
                lat_0=76.5,
                lon_0=260,
                width=3.4e6,
                height=2e6,
                resolution='i')
            pickle.dump(m, open(fname, 'wb'))
        else:
            m = pickle.load(open(fname, 'rb'))
    else:
        print('Try different model_name')

    x0, x1, y0, y1 = [x*1e3 for x in [750, 3400, 0, 2000]]

    return m, (x0, x1, y0, y1)


def ETOPO_interpolator():
    """Create a function to interpolate ETOPO data onto a new grid

    Returns
    -------
    f : function(lon, lat)
        A function that takes arguments and returns depths
        Positive values are below sea level
    """
    bathy_dir = '/home/hugke729/PhD/Data/Bathymetry/'

    # Read in binary file of ETOPO data with just bathymetry north of 50N
    # File goes W to E, but N to S
    # Depths at centre of grid cells
    z = np.fromfile(bathy_dir + 'etopo1_bed_c_i2.bin', 'i2')[:40*60*360*60]
    z = z.reshape(40*60, 360*60)[::-1, :].T

    # Create the corresponding vectors that associated with etopo file
    lat_f = np.r_[50:90:2401j]
    lon_f = np.r_[-180:180:21601j]
    lat_c = (lat_f[:-1] + lat_f[1:])/2
    lon_c = (lon_f[:-1] + lon_f[1:])/2

    # To ensure interpolation doesn't fail, we need lon_c and lat_c to
    # be able to capture values very close to 180W, 180E, and 90N.
    # Hence, adjust first and last values as workaround
    lon_c[0], lon_c[-1] = -180, 180
    lat_c[-1] = 90

    return RegularGridInterpolator((lon_c, lat_c), -z)


def get_penny_strait_soundings():
    """Get all bathymetry data from Penny Strait and nearby

    Data given to me by Humfrey Melling

    Returns
    -------
    lon, lat, z : 1D arrays
    """
    data_dir = '/home/hugke729/PhD/Data/Penny_Strait/'
    data_file = data_dir + 'Soundings in GEOTRACES area.csv'
    return np.loadtxt(data_file, delimiter=',', unpack=True)


def interpolate_soundings_to_grid(lon_out, lat_out):
    """Interpolate soundings in and near Penny Strait to output grid

    Returns
    -------
    z_ps_out: Nx x Ny masked array
        Depths with values only where reasonable interpolation can be made
    """

    # Get Penny Strait soundings
    ps_lon, ps_lat, ps_z = get_penny_strait_soundings()

    # Find parts of the output grid within the bounds of the soundings
    ps_lon_inds = np.logical_and(lon_out < ps_lon.max(), lon_out > ps_lon.min())
    ps_lat_inds = np.logical_and(lat_out < ps_lat.max(), lat_out > ps_lat.min())
    ps_inds = np.logical_and(ps_lon_inds, ps_lat_inds)
    ps_LON, ps_LAT = lon_out[ps_inds], lat_out[ps_inds]

    # Interpolate sounding data onto output grid
    vtx, wts = interp_weights((ps_lon, ps_lat), (ps_LON, ps_LAT))
    z_ps_out = np.full_like(lon_out, np.nan)
    z_ps_out[ps_inds] = interpolate(
        ps_z.flatten(), vtx, wts, fill_value=np.nan)
    z_ps_out = ma.masked_invalid(z_ps_out)

    return z_ps_out


def interpolate_ETOPO_to_grid(lon_out, lat_out):
    """Get the seafloor depth using ETOPO

    Inputs
    ------
    lon_out, lat_out: 2D arrays
        Locations of the centres of the grid cells for the output bathymetry

    Returns
    -------
    bathy : 2D array
        Interpolated seafloor depth (-ve values are above sea level)
    """

    f = ETOPO_interpolator()
    return f((lon_out, lat_out))


def combine_etopo_with_soundings(lon_out, lat_out, combine_method='mean'):
    """
    Inputs
    ------
    lon_out, lat_out: 2D arrays
        Locations of the centres of the grid cells for the output bathymetry
    combine_method : str
        Either mean or max

    Returns
    -------
    bathy : 2D array
        Interpolated seafloor depth
    """

    etopo_z = interpolate_ETOPO_to_grid(lon_out, lat_out)
    soundings_z = interpolate_soundings_to_grid(lon_out, lat_out)

    # Interpolation of soundings doesn't account for land, so the 'depth'
    # where the land is funky. Treat this by masking the land based on ETOPO
    soundings_z = ma.masked_where(etopo_z < 0, soundings_z)
    etopo_z = ma.masked_where(etopo_z < 0, etopo_z)

    # Make copies of ETOPO dataset (*1.0 here to ensure different IDs)
    max_z = etopo_z*1.0
    mean_z = etopo_z*1.0

    # For grid points with soundings, calculate mean and max
    inds = ~soundings_z.mask
    mean_z[inds] = (soundings_z[inds] + etopo_z[inds])/2
    max_z[inds] = np.max(np.c_[soundings_z[inds], etopo_z[inds]], axis=1)

    if combine_method == 'mean':
        return mean_z
    elif combine_method == 'max':
        return max_z


def interpolate_in_vertical(X, z_in, z_out):
    """Use linear interpolation to convert S and T at fixed depths from
    climatology to new depths given by z_out"""

    dz = np.diff(z_in)

    # Preallocate output
    Nx, Ny = X.shape[:2]
    X_out = np.zeros((Nx, Ny, len(z_out)))

    for i, z in enumerate(z_out):
        ind = np.searchsorted(z_in, z) - 1

        # searchsorted returns different answer than I want if
        # the value to find the index for is exactly equal to the array
        # it is being placed. This is a work around for that
        if z in z_out:
            ind = ind + 1

        try:
            # Weight for level above (w_a) and below (w_b)
            w_a = np.abs(z_in[ind + 1] - z)/dz[ind]
            w_b = np.abs(z_in[ind] - z)/dz[ind]
            # Sum of weights should equal one, but there could be edge cases
            X_out[..., i] = (w_a*X[..., ind] + w_b*X[..., ind + 1])/(w_a + w_b)
        except IndexError:
            # Deeper than KG climatology
            X_out[..., i] = X[..., -1]

    return X_out


def project_quarter_degree_climatology(
        LON_c, LAT_c, z_out, update_progress=False):
    """
    Inputs
    ------
    LON_c, LAT_c : 2D arrays
        Locations of the centres of the grid cells for the output hydrography
    z_out : 1D array
        Depth vector

    Returns
    -------
    S, T : 3D arrays
        S and T projected onto output grid
    """

    # Preallocate intermediate result
    Nx, Ny = LON_c.shape
    Nz_qd = 57  # QD climatology has 57 levels
    S_xy = np.zeros((Nx, Ny, 57))
    T_xy = np.zeros((Nx, Ny, 57))

    if update_progress:
        print('Projected quarter-degree climatology for level', end=' ')
    # Reproject qd climatology to new X-Y grid, but keep original z vector
    for i in range(Nz_qd):
        S_interpolator = quarter_degree_climatology_interpolator('S', i)
        T_interpolator = quarter_degree_climatology_interpolator('T', i)

        S_xy[..., i] = S_interpolator((LAT_c, LON_c))
        T_xy[..., i] = T_interpolator((LAT_c, LON_c))

        if update_progress:
            print(i, flush=True, end=', ')

    # Reproject to z_out
    depth_fname = ('/home/hugke729/PhD/Data/TS_climatology/' +
                   'Arctic Regional Climatology/nc/depths.txt')
    z_in = np.genfromtxt(depth_fname, delimiter=',')
    S = interpolate_in_vertical(S_xy, z_in, z_out)
    T = interpolate_in_vertical(T_xy, z_in, z_out)

    # Convert in situ temperature to potential
    T = ptmp(S, T, z_out[np.newaxis, np.newaxis, :])

    return S, T


def quarter_degree_climatology_interpolator(T_or_S, z_i):
    """Create a function to interpolate TS data onto a new grid

    Inputs
    ------
    T_or_S : str
        Either 'T' or 'S'
    z : float
        Depth (m)

    Returns
    -------
    f : function((lon, lat))
        A function that takes arguments and returns temperature or salinity
    """
    data_dir = ('/home/hugke729/PhD/Data/TS_climatology/' +
                'Arctic Regional Climatology/nc/')

    if T_or_S.lower() == 't':
        var = 't'
    elif T_or_S.lower() == 's':
        var = 's'
    else:
        print('Specify T_or_S correctly')

    fname = data_dir + var + '09_04.nc'
    D = xr.open_dataset(fname, decode_times=False)

    if T_or_S.lower() == 't':
        quantity = D.t_an
    elif T_or_S.lower() == 's':
        quantity = D.s_an

    quantity = np.array(quantity[0, z_i, ...])

    lon_c, lat_c = np.array(D.lon), np.array(D.lat)

    # To ensure interpolation doesn't fail, we need lon_c and lat_c to
    # be able to capture values very close to 180W, 180E, and 90N.
    # Hence, adjust first and last values as workaround
    lon_c[0], lon_c[-1] = -180, 180
    lat_c[-1] = 90

    return RegularGridInterpolator((lat_c, lon_c), quantity)


def project_kliem_greenberg_climatology(X_c, Y_c, m, z_out):
    """
    Inputs
    ------
    X_c, Y_c : 2D arrays
        Locations of the centres of the grid cells for the output hydrography
    m : basemap object
        Basemap associated with X_c and Y_c
    z_out : 1D array
        Depth vector

    Returns
    -------
    S, T : 3D arrays
        S and T projected onto output grid
    """

    # Temperature and salinity on original grid
    def get_T_S(X):
        """Read in T or S from Kliem and Greenberg fields

        X is either 'T' or 'S'"""
        out = np.genfromtxt(kg_data_loc + X + '.lst', skip_header=4)
        # Rearrange to Nz x Nx with surface at top
        out = out.reshape((out.size//Nz_kg, Nz_kg))[:, ::-1]
        return out

    def project_to_new_xy(vtx, wts, S, T):
        # Preallocate outputs interpolated to new xy
        S_xy = np.zeros((Nx, Ny, Nz_kg))
        T_xy = S_xy.copy()

        # Project onto model grid (but keeping KG depths for now)
        for i in range(Nz_kg):
            S_i = interpolate(S[:, i], vtx, wts, np.nan)
            T_i = interpolate(T[:, i], vtx, wts, np.nan)
            S_xy[:, :, i] = S_i.reshape(Nx, Ny)
            T_xy[:, :, i] = T_i.reshape(Nx, Ny)

        return S_xy, T_xy

    # Main part of function
    phd_dir = '/home/hugke729/PhD/'
    kg_data_loc = phd_dir + 'Data/TS_climatology/Kliem_Greenberg/fields/'
    Nx, Ny = X_c.shape
    Nz_kg = 25  # Number of levels in Kliem Greenberg fields

    node_no, lon_kg, lat_kg = np.genfromtxt(kg_data_loc + 'arctic4.nod').T
    X_kg, Y_kg = m(lon_kg, lat_kg)
    # nodes = np.genfromtxt(kg_data_loc + 'arctic4.ele').T.astype(int)[1::, :] - 1

    vtx, wts = interp_weights((X_kg, Y_kg), (X_c, Y_c))
    S = get_T_S('S')
    T = get_T_S('T')
    S_xy, T_xy = project_to_new_xy(vtx, wts, S, T)

    # Depths used by Kliem and Greenberg
    z_kg = np.genfromtxt(kg_data_loc + 'depths.txt')[::-1]

    S_out = interpolate_in_vertical(S_xy, z_kg, z_out)
    T_out = interpolate_in_vertical(T_xy, z_kg, z_out)

    return S_out, T_out


def combine_qd_and_kg_climatology(X_c, Y_c, m, z_out, ST_kg=None, ST_qd=None,
                                  filter_sigma=10):
    """Combine the two climatologies

    Method
    ------
    Calculate an array of weights from 0 to 1 by findings points within the
    Kliem and Greenberg mesh. Then smooth around the edges with a Gaussian

    Inputs
    ------
    X_c, Y_c : 2D arrays
        Locations of the centres of the grid cells for the output hydrography
    m : basemap object
        Basemap associated with X_c and Y_c
    ST_kg : 2-tuple
        Two 3D arrays of salinity and temp from Kliem + Greenberg
        Vertical grid must be z_out
    ST_qd : 2-tuple
        Two 3D arrays of salinity and temp from quarter-degree climatology
        Vertical grid must be z_out
    z_out : 1D array
        Depth vector

    Returns
    -------
    S, T : 3D arrays
    """

    # Read in the boundary describing the KG mesh
    kg_bnd_lon, kg_bnd_lat = np.genfromtxt(
        '/home/hugke729/PhD/Data/TS_climatology/Kliem_Greenberg/derived/' +
        'approx_inner_boundary.txt', delimiter=',', skip_header=1)
    bnd_x, bnd_y = m(kg_bnd_lon, kg_bnd_lat)

    # Convert to matplotlib Path object
    kg_path = Path(np.c_[bnd_x, bnd_y])
    grid_pts = np.c_[X_c.flatten(), Y_c.flatten()]

    # Find points inside
    inside = kg_path.contains_points(grid_pts).reshape(X_c.shape)
    weights = gaussian_filter(inside.astype(float), filter_sigma)
    kg_wts, qd_wts = weights[..., np.newaxis], 1 - weights[..., np.newaxis]

    # Project each climatology to new grid unless specified as inputs
    LON_c, LAT_c = m(X_c, Y_c, inverse=True)
    if ST_kg is None:
        S_kg, T_kg = project_kliem_greenberg_climatology(X_c, Y_c, m, z_out)
    else:
        S_kg, T_kg = ST_kg
    if ST_qd is None:
        S_qd, T_qd = project_quarter_degree_climatology(LON_c, LAT_c, z_out)
    else:
        S_qd, T_qd = ST_qd

    # Ensure arrays are masked
    S_kg, T_kg, S_qd, T_qd = map(ma.masked_invalid, [S_kg, T_kg, S_qd, T_qd])

    # Weighted average of the two datasets
    # Weights are effectively 0 or 1 well away from the boundary path
    wts_total = kg_wts*~S_kg.mask + qd_wts*~S_qd.mask
    S = (kg_wts*ma.filled(S_kg, 0) + qd_wts*ma.filled(S_qd, 0))/wts_total
    T = (kg_wts*ma.filled(T_kg, 0) + qd_wts*ma.filled(T_qd, 0))/wts_total

    # Zeros and NaNs are introduced by the previous step, so remove these
    # Will leave zeros for temperature just in case
    S = ma.masked_equal(ma.masked_invalid(S), 0)
    T = ma.masked_invalid(T)

    return S, T


def deepen_nares_st(m, X_c, Y_c, depth):
    """Artificially deepen Nares Strait

    Then it can act as if it is Fram Strait and allow SSH difference to be
    set up

    Inputs
    ------
    m : Basemap object
    X_c, Y_c : 2D arrays
        Location of cell centres
    depth : 2D array
        Bathymetry to be converted

    Returns
    -------
    new_depth : 2D array
        Same as input depth, but with Nares Strait deepened
    """

    # Polygon describing new, widened Nares Strait
    lons = np.r_[-79.916, -75.896, -65.234, -64.456, -71.386, -75.576, -76.378,
                 -75.9, -72.876, -70.765, -67.603, -66.75, -66.547, -62.286,
                 -53.185, -49.028, -48.363, -59.415, -64.639]
    lats = np.r_[84.144, 83.861, 82.821, 81.974, 80.079, 79.182, 77.987,
                 76.732, 75.43, 75.475, 75.067, 75.991, 77.473, 79.089,
                 80.679, 81.63, 82.559, 84.01, 84.43]

    # Convert to map units, then matplotlib Path
    xs, ys = m(lons, lats)
    p = Path(np.c_[xs, ys])

    # Find all grid points within box
    inside_inds = p.contains_points(np.c_[X_c.flatten(), Y_c.flatten()])
    inside_inds = inside_inds.reshape(X_c.shape)

    # Convolve inside_inds with square box to get an array of wts that is
    # 1 in the middle of the box, 0 well outside and 0 to 1 near edges

    # Using a kernel with specified width
    smooth_kernel = 100  # Distance in kilometres
    x, y = X_c[:, 0], Y_c[0, :]
    x_km, y_km = x.max()/1e3, y.max()/1e3

    # Set Nx and Ny for convolution (scale down from number of kms for speed)
    # Note that smooth_kernel is dependent on Nx and Ny so scale this
    # by the same amount
    fac = 10
    Nx, Ny, smooth_kernel = [int(a/fac) for a in [x_km, y_km, smooth_kernel]]

    # Create 2D kernel
    kernel = np.ones((smooth_kernel, smooth_kernel))
    wts = uneven_2D_convolve(
        inside_inds, x, y, kernel, Nx=Nx, Ny=Ny)/kernel.size

    # Calculate new depth by setting 600 m as depth of Nares St and smoothly
    # converting back to real bathymetry
    to_change = wts > 0.05
    new_depth = depth.copy()
    new_depth[to_change] = 600*wts[to_change] + (new_depth*(1-wts))[to_change]

    return new_depth


def project_tide(X_c, Y_c, m, constituents, return_velocity=False):
    """
    Inputs
    ------
    X_c, Y_c : 2D arrays
        Locations of the centres of the grid cells for the output hydrography
    m : basemap object
        Basemap associated with X_c and Y_c
    constituent : str or list of strings
        Any of S2, 01, N2, M4, K1
    return_velocity : bool
        Whether to return barotropic velocities

    Returns
    -------
    amp : 2D array or list of 2D arrays
        Tidal amplitude for given constituents (metres)
    phase : 2D array or list of 2D arrays
        Tidal phase in degrees for given constituents
    U, V : 2D arrays
        Velocities for given constituents
    """

    tide_data_dir = '/home/hugke729/Programs/WebTide/data/arctic9/'

    # Read in mesh information
    node_file = tide_data_dir + 'arctic9.nod'
    node_no, lon_tide, lat_tide = np.genfromtxt(node_file, unpack=True)
    node_no = node_no.astype('int')
    X_tide, Y_tide = m(lon_tide, lat_tide)

    # Calculate vertices and weights to project to model grid
    vtx, wts = interp_weights((X_tide, Y_tide), (X_c, Y_c))

    # Create blank list to append results to
    amp = []
    phase = []
    u, u_phase = [], []
    v, v_phase = [], []

    # Ensure constituents is a list
    if type(constituents) is str:
        constituents = [constituents]

    # Read in information for given tide
    for const in constituents:
        constituent_file = tide_data_dir + const.upper() + '.barotropic.s2c'
        amp_node, phase_node = np.genfromtxt(
            constituent_file, skip_header=3, usecols=[1, 2], unpack=True)

        if return_velocity:
            vel_file = constituent_file.replace('s2c', 'v2c')
            no, u_a, u_p, v_a, v_p = np.genfromtxt(
                vel_file, skip_header=3, unpack=True)

        def tmp_interpolate(X):
            return interpolate(X, vtx, wts, np.nan).reshape(X_c.shape)

        def interpolate_phase(X):
            """
            Do phase interp multiple times with different wrapping points.
            Then convert back to original wrapping.
            Then take median
            """
            filterwarnings('ignore', 'All-NaN slice*.')
            wraps = np.r_[0:360:7j][1:]
            phases = np.zeros((X_c.shape[0], X_c.shape[1], len(wraps)))
            for i, wrap_out in enumerate(wraps):
                phase_nodes = change_wrap(X, 180, wrap_out)
                phases_i = tmp_interpolate(phase_nodes)
                phases[..., i] = change_wrap(phases_i, wrap_out, 180)

            return np.nanmedian(phases, axis=-1)

        if return_velocity:
            u += [tmp_interpolate(u_a)]
            u_phase += [interpolate_phase(u_p)]
            v += [tmp_interpolate(v_a)]
            v_phase += [interpolate_phase(v_p)]

        # Project to model grid
        amp += [tmp_interpolate(amp_node)]

        # Add current constituent to list
        phase += [interpolate_phase(phase_node)]

    # Sum up velocities to get U0 and V0
    U0, V0 = np.zeros_like(X_c), np.zeros_like(X_c)
    for u_i, u_i_phase, v_i, v_i_phase in zip(u, u_phase, v, v_phase):
        # Using cos here as that is what's used in SSHR package
        U0 += u_i*cosd(u_i_phase)
        V0 += v_i*cosd(v_i_phase)

    # Convert list to array if appropriate
    if len(amp) == 1:
        amp, phase = amp[0], phase[0]

    if return_velocity:
        return amp, phase, U0, V0
    else:
        return amp, phase


def project_and_extrapolate_tide(
        X_c, Y_c, m, constituents, leave_mask=None, update_progress=1,
        return_velocity=False):
    """
    Project tidal amplitude and phases for given constituents onto new grid

    Then extrapolate empty region using inpaint_nans

    Inputs
    ------
    X_c, Y_c : 2D arrays
        Locations of the centres of the grid cells for the output hydrography
    m : basemap object
        Basemap associated with X_c and Y_c
    constituents : str or list of strings
        Any of S2, 01, N2, M4, K1
    leave_mask : 2D boolean array
        Argument passed to inpaint_nans
    update_progress : (0, 1, or 2)
        Level of updates printed
        0 = no updates
        1 = update after each constituent
        2 = update all steps of inpaint_nans
    return_velocity : bool
        Whether to return velocities corresponding to time 0

    Returns
    -------
    amp : 2D array or list of 2D arrays
        Tidal amplitude for given constituent (metres)
    phase : 2D array or list of 2D arrays
        Tidal phase in degrees for given constituent
    U0, V0 : 2D arrays
        Barotropic tidal current velocities
    """
    A, P, U0, V0 = project_tide(X_c, Y_c, m, constituents,
                                return_velocity=True)

    # Ensure A and P are within a list even if only one constituent
    if type(A) is not list:
        one_constituent = True
        A, P = [[X] for X in [A, P]]
    else:
        one_constituent = False

    springs = get_springs(A[0], leave_mask)

    inpaint_opts = dict(
        springs=springs, update_progress=update_progress,
        leave_mask=leave_mask)
    if update_progress == 1:
        # If 0, it is already False. If it is 1, we don't want inpaint_nans
        # to update the progress
        inpaint_opts['update_progress'] = False

    for i, (A_i, P_i) in enumerate(zip(A, P)):
        C = constituents if one_constituent else constituents[i]
        if update_progress >= 1:
            print('Calculating ' + C + ' tide', flush=True)
        A[i] = inpaint_nans(A_i, (X_c, Y_c), **inpaint_opts)
        P[i] = inpaint_nans(P_i, (X_c, Y_c), **inpaint_opts)

    if return_velocity:
        U0 = inpaint_nans(U0, (X_c, Y_c), **inpaint_opts)
        V0 = inpaint_nans(V0, (X_c, Y_c), **inpaint_opts)

    # Return A and P to array if only one constituent given
    if len(A) == 1:
        A, P = A[0], P[0]

    if return_velocity:
        return A, P, U0, V0
    else:
        return A, P

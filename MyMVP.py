# All post-processing of MVP should be here.
# Plots in ./MVP_plots.py
# Scripts adjusted from Rowan and Jody's work
from operator import itemgetter as get
import numpy as np
import numpy.ma as ma
import scipy.signal as signal
import re
from time import strptime
from scipy.interpolate import interp1d
from scipy.stats import binned_statistic, mode
from scipy.ndimage.measurements import label
import pickle
import matplotlib.pyplot as plt
import seawater.eos80 as sw
from seawater.constants import c3515
from warnings import filterwarnings
from MyMapFunctions import haversines
from MyFunctions import (scalar_projection, angle, central_diff_gradient,
                         get_contour)
from MySysUtils import preall
from MyInterp import smooth1d_with_holes as smooth
from MyFilters import butter_lowpass_filter as blf
from MyFilters import butter_highpass_filter as bhf
from MyInterp import interp_weights, interpolate
from MyGrids import estimate_cell_edges
from MyNumpyTools import (nan_or_masked, nan_or_masked_to_value, logical_all,
                          minmax, arange_like, normalize)
from MyOceanography import intermediate_density_profile
from vertmodes import vertModes

# Overview of functions

# concatenate_binned_arrays
#   |-loadMVP_m1
#   |  |- cast_no_to_filename
#   |  |- header_info
#   |  |- m1_to_dict
#   |  |- smooth_p_and_z
#   |  |- lag_temperature
#   |  |- calc_density
#   |  |- calc_N2
#   |  |- potential_temp
#   |  |- calc_eps (removed eps references (11 Dec 2017)
#   |  |   |- calc_Lt
#   |  |- calc_eps_avg
#   |  |- calc_modes
#   |  |- bin_fields
#   |      |- select_downcast
#   |-interp_missing_latlon
#   |-flatten_to_line
#   |-two_layer_treatment
#   |-combine_MVP_ADCP
#
# create_timeline (stand-alone function used only once)


def loadMVP_m1(cast_no, z_bins=None, bin_data=True, lagT=True,
               mask_nans=False):
    """Load .m1 file and bin data into z_bins if desired

    Inputs
    ------
    cast_no : integer
        Cast number associated with the .m1 file
    z_bins : array
        Depth values to bin results into
    bin_data : bool
        Whether to bin data (influences what the function returns)
    lagT : bool
        Whether or not to lag temperature (set to False in obtain_phase_lag.py to see how the dt comes about)
    mask_nans : bool
        Whether to return binned values as masked arrays

    Returns
    -------
    xyt : dict
        Time and location information
    data : dict
        Vectors of length N_raw for each of the parameters in field
    binned : dict (returned only if bin_data is True)
        As for data, but binned into z_bins with results that are
        len(z_bins) - 1
    """

    filename = cast_no_to_filename(cast_no)
    # Get header information
    with open(filename, 'r') as f:
        # read in 5000 bytes to ensure all header is read
        header = f.read(5000)
        h = header_info(header)

    # Time and place
    xyt = {}
    xyt['lat'] = h.coords('lat')
    xyt['lon'] = h.coords('lon')
    xyt['time'] = h.date_time('time')
    xyt['date'] = h.date_time('date')
    xyt['cast'] = h.cast_number()
    xyt['bottom'] = h.bottom()

    # Hydrography
    fields = ['p', 'z', 'SV', 'T', 'C', 'S', 'rho', 'ANGL1', 'ANLG2', 'ANLG3']
    data = m1_to_dict(filename, fields)

    data['p_raw'], data['z_raw'] = data['p'].copy(), data['z'].copy()
    data['p'], data['z'] = smooth_p_and_z(*get('p_raw', 'z_raw')(data))

    if lagT:
        # Having option to not lag temperature is helpful to show working of
        # how I got the lag
        data['T_unlagged'] = data['T'].copy()
        data['S_unlagged'] = data['S'].copy()
        data['S'], data['T'] = lag_temperature(*get('C', 'T', 'p')(data))

    # Note: I think I have covered my bases to ensure arrays aren't modified
    # in place, but I might have missed some. Anyways, I calculate dissipation
    # early on, just to be safe, since the functions after that include
    # smoothing procedures, which would ruin diss calculation
    data['prho'], data['rho'] = calc_density(*get('S', 'T', 'p')(data))
    # data['eps'], data['L_T'] = calc_eps(*get('p_raw', 'prho', 'z')(data))
    # data['eps_zavg'], data['eps_z_integral'] = calc_eps_avg(
    #     data['eps'], data['z'], xyt['bottom'])
    data['theta'] = potential_temp(*get('S', 'T', 'p')(data))
    data['N2'] = calc_N2(*get('p', 'prho', 'z')(data))

    if bin_data:
        z_bins = np.arange(0, 250) if z_bins is None else z_bins
        binned = bin_fields(data, z_bins, mask_nans)
        binned['z_bins'] = z_bins
        hori, vert, c = calc_modes(binned['N2'], xyt['bottom'], z_bins)
        binned['hori_0'], binned['hori_1'], binned['hori_2'] = hori.T
        binned['vert_0'], binned['vert_1'], binned['vert_2'] = vert.T
        binned['c0'], binned['c1'], binned['c2'] = c

        return xyt, data, binned
    else:
        return xyt, data


def cast_no_to_filename(cast_no):
    file_base = '/home/hugke729/PhD/Data/Shipboard/MVP/raw/1503001_'
    file_name = file_base + str(cast_no).zfill(4) + '.m1'
    return file_name


def concatenate_binned_arrays(
        start_n, end_n, z_bins=np.arange(0, 251), flatten_transect=False,
        print_cast=False):
    """Join single grids returned by loadMVP_m1 into one overall grid

    Inputs
    ------
    start_n, end_n : ints
        Cast numbers for start and end of transect
    z_bins : 1D array
        Edges of bins used for binning data
    flatten_transect : bool
        Whether to flatten transect to line (set to True for long-sound but
        not cross-sound transects)

    Returns
    -------
    grid_all : dict
        Various arrays with oceanographic, location, and grid information
    """

    # Preallocate the dictionary to hold concatenated grids with NaNs
    grid_all = {}
    fields = ['p', 'z', 'SV', 'T', 'C', 'S', 'rho', 'prho', 'theta',
              'ANGL1', 'ANLG2', 'ANLG3', 'N2',
              'hori_0', 'hori_1', 'hori_2', 'vert_0', 'vert_1', 'vert_2']
    scalar_fields = ['c0', 'c1', 'c2']
    xyt_fields = ['bottom', 'cast', 'date', 'lat', 'lon', 'time']

    Nx, Nz = end_n - start_n + 1, len(z_bins) - 1

    for field in fields:
        grid_all[field] = np.full((Nx, Nz), np.nan)
    for field in scalar_fields:
        grid_all[field] = np.full(Nx, np.nan)
    for field in xyt_fields:
        grid_all[field] = []

    # For each individual m1 file, load in the data and get a grid by using
    # loadMVP_m1, then add it to the overall grid
    bad_casts = []  # Keep list of bad casts
    for i, cast_no in enumerate(range(start_n, end_n + 1)):
        print(cast_no, end=' ') if print_cast else None

        # Include try/except to account for profiles that didn't work
        try:
            xyt_i, _, grid_i = loadMVP_m1(cast_no, z_bins)
            for field in fields:
                grid_all[field][i, :] = grid_i[field]

            for field in scalar_fields:
                grid_all[field][i] = grid_i[field]

            # Include time and position information
            for field in xyt_fields:
                grid_all[field] += [xyt_i[field]]

        except (ValueError, IndexError):
            # Skip this cast. If the try suite didn't work, it's very likely
            # the cast had something wrong, such as no downcast data
            bad_casts += [i]
            print(' Bad cast: ' + str(cast_no), flush=True)

    # Remove empty spaces left by bad casts
    for field in (fields + scalar_fields):
        grid_all[field] = np.delete(grid_all[field], bad_casts, axis=0)

    # Fix coordinates for a few files that have missing information
    if np.any(np.array(grid_all['lon']) == 0):
        grid_all = interp_missing_latlon(grid_all)

    # Calculate distances between sites
    grid_all['dist'], grid_all['bearing'] = haversines(
        grid_all['lon'], grid_all['lat'])

    # Ensure that dist increases from west to east for cross-sections
    if np.mean(np.diff(grid_all['lon'])) < 0 and not flatten_transect:
        grid_all['dist'] = grid_all['dist'].max() - grid_all['dist']

    # Mask NaNs
    for field in (fields + scalar_fields):
        grid_all[field] = ma.masked_invalid(grid_all[field])

    if flatten_transect:
        # Put all repeat transects in Maury Channel onto a single line
        grid_all['dist_flat'] = flatten_to_line(
            grid_all['lon'], grid_all['lat'])/1000  # kilometres

        # Then, reorder so that distances increase monotonically
        dist_order = np.argsort(grid_all['dist_flat'])
        for key, value in grid_all.items():
            if type(value) is list:
                grid_all[key] = [value[i] for i in dist_order]
            else:
                # If not list, it will be numpy array
                grid_all[key] = value[dist_order, ...]

    # Make a 2D distance arrays (for pcolor plots) of size n x m and
    # n + 1 x m + 1, where n is number of casts and m is number of z bins
    grid_all['dist2d_c'] = np.outer(grid_all.get('dist_flat', grid_all['dist']),
                                    np.ones(len(z_bins) - 1))
    dist_np1 = estimate_cell_edges(grid_all.get('dist_flat', grid_all['dist']))
    grid_all['dist2d_f'] = np.outer(dist_np1, np.ones(len(z_bins)))

    # Add in z_bins of size m + 1
    # using z_f notation from MITgcm
    grid_all['z_f'] = z_bins*1.0
    grid_all['z_c'] = (z_bins[1:] + z_bins[:-1])/2
    grid_all['z_f2d'] = np.outer(np.ones(dist_np1.size), z_bins)

    # Analyse flow in two layers. Do separately for each part of sill,
    # which I call sill_1 and sill_2 funnily enough
    # Keep all results in a dict within the overall dict
    if flatten_transect:
        sill_names = 'sill_1', 'sill_2'
        x_sills_both = (48, 58), (75, 90)
        x_ranges_both = (0, 75), (0, 200)
        for sill_name, x_sills, x_ranges in zip(
                sill_names, x_sills_both, x_ranges_both):

            rho_interface, gprime, z_interface = two_layer_treatment(
                grid_all, x_sills, x_ranges)

            grid_all[sill_name] = dict()
            grid_all[sill_name]['x_sills'] = x_sills
            grid_all[sill_name]['x_ranges'] = x_ranges
            grid_all[sill_name]['gprime'] = gprime
            grid_all[sill_name]['rho_interface'] = rho_interface
            grid_all[sill_name]['interface_depth'] = z_interface

    return grid_all


def bin_data(data_to_bin, z, z_bins):
    """Bin data into z_bins by taking mean within the bins"""
    data_to_bin = ma.masked_invalid(data_to_bin)
    binned_data, _, _ = binned_statistic(
        z, data_to_bin, bins=z_bins, statistic='mean')
    return binned_data


class header_info:

    def __init__(self, header):
        self.header = header

    def coords(self, lat_or_lon):
        """
        Pull out coordinates from header

        | Coordinates in header file are in the form
        | LAT ( ddmm.mmmmmmm,N):  7435.0958200,N

        Inputs
        ------
        header: str from open(file).read()
            Long string containing header time and location information
        lat_or_lon: str
            either `'lat'` or `'lon'`

        Output
        ------
        Coordinate in decimal minutes
        """
        # Search for line starting with LAT or LON and ending with N, E, W, or S
        inds = re.search(lat_or_lon + '.*[NEWS]', self.header, re.I).span()
        coord_line = self.header[inds[0]:inds[1]]
        digit_inds = re.search('[0-9].*' + '\.' + '[0-9]*', coord_line).span()
        coord = coord_line[digit_inds[0]:digit_inds[1]]
        split = 2
        if lat_or_lon is 'lon':
            split = 3
        coord_dd = float(coord[:split])
        coord_dm = float(coord[split:])
        coord = coord_dd + coord_dm/60

        # Adjust sign
        if coord_line[-1] in ['W', 'S']:
            coord = -coord
        return coord

    def date_time(self, time_or_date):
        """
        Pull out time and date

        | Example of date and time in header:
        | ``Time (hh|mm|ss.s): 13:24:57.0``
        | ``Date (dd/mm/yyyy): 06/09/2015``
        """
        inds = re.search(time_or_date + '.*[0-9]', self.header, re.I).span()
        t_or_d_line = self.header[inds[0]:inds[1]]
        t_or_d_inds = re.search('[0-9].*[0-9]', t_or_d_line).span()
        t_or_d = t_or_d_line[t_or_d_inds[0]:t_or_d_inds[1]]
        return t_or_d

    def cast_number(self):
        """Pull out cast number"""
        inds = re.search('Index.*[0-9]', self.header).span()
        index_line = self.header[inds[0]:inds[1]]
        index_inds = re.search(4*'[0-9]', index_line).span()
        index = int(index_line[index_inds[0]:index_inds[1]])
        return index

    def bottom(self):
        """Pull out bottom depth"""
        inds = re.search('Bottom.*[0-9]', self.header).span()
        bottom_line = self.header[inds[0]:inds[1]]
        bottom_inds = re.search('[0-9].*', bottom_line).span()
        bottom = float(bottom_line[bottom_inds[0]:bottom_inds[1]])
        return bottom


def m1_to_dict(filename, fields):
    """Convert .m1 file to python dictionary"""
    data = {}
    data_array = np.loadtxt(filename, skiprows=49, delimiter=',', unpack=True)
    for i, key in enumerate(fields):
        data[key] = data_array[i, :]
    return data


def select_downcast(pressure, for_overturn_calcs=False):
    """Find indices of the downcast part of data"""
    # Take the derivative of the pressure profile
    dp = np.diff(pressure)
    # Constants for the filter
    B, A = signal.butter(2, 0.01, output='ba')
    # Filter the pressure derivative, to smooth out the curve
    dp_smooth = signal.filtfilt(B, A, dp)
    # Make the arrays the same size
    dp_smooth = np.append(dp_smooth, [0])
    # Find the indices where the descent rate is more than 0.05
    falling_inds = dp_smooth > 0.05

    if for_overturn_calcs:
        # For overturns, we want fall to be smooth.
        # Therefore, we want to exclude portions near the surface where
        # fall rate may drop below 0.05. In such cases without the code below
        # we would end up with discontinuous pieces of the profile
        inds_label = label(falling_inds)[0]
        inds_label_mode = mode(inds_label[falling_inds])[0]
        falling_inds[inds_label != inds_label_mode] = False

    falling_inds = np.where(falling_inds)[0]

    return falling_inds


def lag_temperature(C, T, p, lag_n=-3.2):
    """Lag temperature by a fixed number of samples"""
    # See /home/hugke729/PhD/Python/MVP/obtain_phase_lag.py
    # Lag is negative, so it is actually conductivity lagging, but leave
    # names as is
    samples = np.arange(len(T))
    T = np.interp(samples, samples + lag_n, T)

    # Recalculate salinity
    S = sw.salt(C/c3515, T, p)
    return S, T


def calc_density(S, T, p):
    args = S, T, p
    prho = sw.pden(*args)
    rho = sw.dens(*args)
    return prho, rho


def calc_N2(p, prho, z):
    """Calculate buoyancy frequency in a 5 step process

    1) Sort p. density and monotonically increasing vectors
    2) Smooth this vector
    3) Get central difference of smooth pden and smoothed z
    4) Smooth the derivative
    Yip, that's smoothing of a quantity calculated from smoothed quantities
    5) Convert to buoyancy frequency
    6) Interpolate N2 back onto original z vector
    """
    # Don't want to modify in place (otherwise over smooth things)
    p = p.copy()
    prho = prho.copy()
    z = z.copy()

    # For most raw quantities, I'm leaving upcast and downcast data unchanged
    # However, for N2 I specifically want to work with only downcast data
    down_inds = select_downcast(p)
    not_down_inds = np.setdiff1d(np.arange(z.size), down_inds)

    prho[not_down_inds] = np.nan
    z[not_down_inds] = np.nan

    # Step 1
    prho[down_inds] = np.sort(prho[down_inds])

    # Step 2
    # Smooth p. density over 12 values based on inspection of smoothed vs raw
    # for various smoothing values (equals 2Hz)
    smooth_n = 12
    prho_smooth = smooth(prho, smooth_n)

    # Step 3
    # Ignore warning arising from dz == 0 in central_diff_gradient
    filterwarnings('ignore', '.*invalid value encountered in true_divide*.')
    z_smooth = smooth(z, smooth_n)
    dprho_dz = central_diff_gradient(prho_smooth, z_smooth)

    # Convert obviously incorrect values arising from MVP not falling to NaN
    # A value of dprho_dz greater than 10 is far larger than I expect
    bad_inds = np.abs(nan_or_masked_to_value(dprho_dz, 11)) > 10

    dprho_dz[bad_inds] = np.nan

    # Step 4
    dprho_dz = smooth(dprho_dz, smooth_n)

    # Step 5
    N2 = 9.81*ma.filled(dprho_dz, np.nan)/np.array(prho)

    # Step 6
    N2 = np.interp(z, z_smooth, N2)

    return N2


def potential_temp(S, T, p):
    theta = sw.ptmp(S, T, p)
    return theta


def smooth_p_and_z(p, z):
    """Smooth the pressure and depth records.

    Raw data are jumpy because the resolutions are 0.1 dbar and 0.1m"""

    # Smoothing over 25 samples because records at 25 Hz
    p = smooth(p, 25)
    z = smooth(z, 25)
    return p, z


def calc_Lt(prho, z, n_smooth_rho=8, plot_overturns=False, min_drho=5E-4):
    """Calculate Thorpe scale

    Inputs are not allowed NaNs

    Get N2 estimate for overturn at same time

    n_smooth_rho defaults to 8 samples (approx 3 Hz). This is estimated from
    coherence analysis of temperature and conductivity. It is a reasonable
    comprimise between smoothing overturns and picking up noise
    """
    prho_raw = prho.copy()
    prho = prho.copy()
    z = z.copy()
    z_raw = z.copy()

    # Doubly ensure z is monotonically increasing and last prho value is max
    dz = central_diff_gradient(z)
    if np.any(dz < 0):
        prho = prho[dz > 0]
        z = z[dz > 0]
    prho[-1] = np.nanmax(prho) + 1E-6

    # Ensure no overturns involve first point (or cumsum wont have zeros)
    prho[0] = prho.min() - 0.02

    # Smooth prho
    prho = smooth(prho, n_smooth_rho)

    # Be extra cautious and calculate intermediate profile following
    # Gargett and Garner (2008)
    prho = intermediate_density_profile(prho, min_drho=min_drho)

    # Sort potential density
    inds = np.argsort(prho, kind='mergesort')

    # Find the displacements needed to sort to density profile
    thorpe_disp = z[inds] - z

    # Index displacements
    inds_disp = inds - np.arange(0, len(inds))

    # Overturn bounds where cumulative sum is zero (ignore first)
    overturn_inds = np.argwhere(np.cumsum(inds_disp) == 0).squeeze()
    overturn_starts = np.setdiff1d(overturn_inds[:-1], overturn_inds - 1)
    overturn_ends = np.setdiff1d(overturn_inds[1:], overturn_inds + 1)

    # Currenty, overturn_starts includes a stable point above the overturn
    # Remove this simply by adding one
    overturn_starts += 1

    min_dens_range = 1E-3
    min_N_samples = 4  # approx 1m in vertical

    starts_to_rm = []
    ends_to_rm = []

    # Plot where overturns start and end
    if plot_overturns:
        fig, ax = plt.subplots(
            ncols=5, sharey=True,
            gridspec_kw=dict(width_ratios=(3, 1, 1, 1, 1)))
        ax[0].set_ylim(z.max(), 0)
        ax[1].set(xlabel='Density range\nin overturn (10$^{-3}$ kg/m3)')
        ax[2].set(xlabel='R_o')
        ax[3].set(xlabel='log dissipation')
        ax[4].plot(thorpe_disp, z)

    for start_i, end_i in zip(overturn_starts, overturn_ends):
        inds_i = np.s_[start_i:end_i]
        inds_ip1 = np.s_[start_i:end_i+1]
        inds_wide = np.s_[np.max((0, start_i - 10)):end_i+11]
        density_range = np.ptp(prho[inds_i])
        # Approximation of Ro given in Gargett and Garner (2008)
        # Assumes constant profiling speed (good approx over size
        # of overturn)
        tdi = thorpe_disp[inds_ip1]
        Ro = min([(tdi < 0).sum(), (tdi > 0).sum()])/tdi.size

        if plot_overturns:
            line, = ax[0].plot(prho[inds_i], z[inds_i], 'r', zorder=5)
            col = 'r' if density_range < min_dens_range else 'k'
            ax[1].plot(2*(density_range*1E3, ), minmax(z[inds_i]), color=col)

            col = 'r' if Ro < 0.2 else 'k'
            ax[2].plot(2*(Ro, ), minmax(z[inds_i]), color=col)

            ax[4].plot(tdi, z[inds_ip1], 'r')
            ax[4].set(xlabel='Thorpe displacement (m)')

        # Noise checks
        # Ignore overturn if
        #     its density range is less than 0.001 kg/m3
        #     if only over distance of 4 samples (~1m)
        #     badly conditioned (as quantified by Ro)
        if density_range < min_dens_range or Ro < 0.2 or len(tdi) <= min_N_samples:
            starts_to_rm += [start_i]
            ends_to_rm += [end_i]
        else:
            # If it passes the test, plot enlarged version
            if plot_overturns:
                norm_prho = normalize(prho[inds_ip1])
                enlarged_prho = prho[inds_ip1][-1] + 0.2*norm_prho
                ax[0].plot(enlarged_prho + 0.02, z[inds_ip1], 'k')

                # T, C, S = [blf(data[key], 3, 25) for key in 'TCS']
                # norm_C = normalize(C[inds_wide])
                # norm_T = normalize(T[inds_wide])
                # norm_S = normalize(S[inds_wide])
                # enlarged_C = prho[inds_ip1][-1] + 0.2*norm_C - 0.22
                # enlarged_S = prho[inds_ip1][-1] + 0.2*norm_S - 0.22
                # enlarged_T = z[inds_ip1][0] + norm_T*np.ptp(z[inds_ip1])
                # ax[0].plot(enlarged_C, enlarged_T, 'grey')
                # ax[0].plot(enlarged_C[10:-10], enlarged_T[10:-10], 'k')
                # ax[0].plot(enlarged_S, enlarged_T, 'r')

                # rho_std = np.std(np.diff(prho_raw[inds_ip1]))
                # rho_ptp = np.ptp(prho[inds_ip1])
                # ratio = rho_ptp/rho_std
                # res_str = '{0:.1f},  {1:.1f},  {2:.1f}'
                # res_str = res_str.format(1e3*rho_std, 1e3*rho_ptp, ratio)
                # xt, yt = prho[inds_ip1].mean() - 0.02, z[inds_ip1].mean()
                # ax[0].text(xt, yt, res_str, ha='right')

    overturn_starts = np.setdiff1d(overturn_starts, starts_to_rm)
    overturn_ends = np.setdiff1d(overturn_ends, ends_to_rm)

    if plot_overturns:
        ax[0].plot(np.sort(prho), z, 'g')
        ax[0].plot(prho_raw, z_raw, 'grey')
        ax[0].plot(prho, z, 'k')
        ax[0].plot(prho[overturn_starts], z[overturn_starts], 'r_')
        ax[0].plot(prho[overturn_ends], z[overturn_ends], 'b_')

    thorpe_scales = np.zeros_like(prho)
    N2 = np.zeros_like(prho)
    for start, end in zip(overturn_starts, overturn_ends):

        prho_range = np.ptp(prho[start:end])  # Range over overturn

        zrms = np.sqrt(np.sum((thorpe_disp[start:end])**2)/(end-start))
        thorpe_scales[start:end] = zrms
        dprho_dz = prho_range/np.ptp(z[start:end])
        N2_in_overturn = 9.81/np.mean(prho[start:end])*dprho_dz
        N2[start:end] = N2_in_overturn

        if plot_overturns:
            log_eps = np.log10(zrms**2*N2_in_overturn**(3/2))
            ax[3].plot(np.ones_like(z[start:end])*log_eps, z[start:end], 'k')
            ax[4].plot(thorpe_scales[start:end], z[start:end], 'k')
            ax[4].plot(np.zeros(end-start), z[start:end], 'k--')

    return thorpe_scales, N2


def calc_eps(p, prho, z, plot_overturns=False, n_smooth_rho=8,
             already_downcast=False, min_drho=0.5e-3):
    """Calculate dissipation using Thorpe scale"""

    if already_downcast:
        # This allows calc_eps to be used for CTD data
        down_inds = arange_like(p)
    else:
        down_inds = select_downcast(p, for_overturn_calcs=True)

    finite_inds = np.argwhere(~nan_or_masked(prho)).squeeze()
    inds = np.intersect1d(finite_inds, down_inds)

    # Preallocate thorpe scale array and associated N2 array
    L_T = np.full_like(prho, np.nan)
    N2 = np.full_like(prho, np.nan)

    # Calc L_T and derive dissipation from parameterisation
    L_T[inds], N2[inds] = calc_Lt(
        prho[inds], z[inds], plot_overturns=plot_overturns,
        n_smooth_rho=n_smooth_rho, min_drho=min_drho)
    eps = (0.8*L_T)**2*N2**(3/2)

    return eps, L_T


def calc_eps_avg(eps, z, depth):
    """Depth-averaged dissipation rate"""

    # Total effective depth for depth average
    # ie ignore regions above and below where MVP went
    depth_total = np.ptp(z[np.isfinite(z)])
    dz = np.insert(np.diff(z), 0, 0)
    eps_zavg = np.nansum(eps*dz)/depth_total

    eps_z_integral = eps_zavg*depth

    return eps_zavg, eps_z_integral


def calc_modes(N2, bottom_depth, z_bins):
    """Wave velocity and structure of first three modes"""

    dz = np.mean(np.diff(z_bins))

    # Truncate N2 to appropriate length based on depth and dz
    Nz = (bottom_depth/dz).astype(int)
    N2 = N2[:Nz]

    # Find indices of start and end of finite values
    finite_vals = nan_or_masked(N2) == 0
    labels = label(finite_vals)[0]
    main_data = np.where(labels == mode(labels[finite_vals]))[1]
    start_ind, end_ind = main_data[0], main_data[-1]

    # Fill in NaN values with start or end values
    N2[:start_ind] = N2[start_ind]
    N2[end_ind + 1:] = N2[end_ind]

    # Preallocate arrays for horizontal and vertical structure
    hori = np.full((len(z_bins) - 1, 3), np.nan)
    vert = hori.copy()

    hori[:len(N2), :], vert[:len(N2), :], c, _ = vertModes(N2, dz, 3)

    return hori, vert, c[:3]


def bin_fields(D, z_bins, mask_nans=False):
    """Bin data contained in python dictionary"""
    grid = {}
    fields = D.keys()
    down = select_downcast(D['p'])

    for key in fields:
        # Scalar fields don't need binning
        if key in ['eps_zavg', 'eps_z_integral']:
            grid[key] = D[key]

        else:
            # Work around for extra masked values in dissipation
            down_inds = down[~(D[key][down].mask)] if key is 'eps' else down

            grid[key] = bin_data(
                D[key][down_inds], D['z'][down_inds], z_bins)
        if mask_nans:
            grid[key] = ma.masked_invalid(grid[key])

    return grid


def interp_missing_latlon(grid):
    """Three files have missing lat/lon information
    Fix by using linear interpolation"""
    missing_coords = np.where(np.array(grid['lat']) == 0)[0]
    for i in missing_coords:
        grid['lat'][i] = (grid['lat'][i-1] + grid['lat'][i+1])/2
        grid['lon'][i] = (grid['lon'][i-1] + grid['lon'][i+1])/2
    return grid


def create_timeline():
    """Pull out just time and cast number and save for other uses

    Saves output to output_file (see below)
    """
    # Really inefficient, since I'm loading/reading way more than I need to
    # but I'm doing this only once
    output_file = '/home/hugke729/PhD/Data/Shipboard/doc/time_cast_lat_lon.txt'
    f = open(output_file, 'wt')

    # for each file, load data, get time, date (convert to decimal day), lat,
    # and lon, then print to file
    for cast_no in range(37, 1151):
        xyt_i, _ = loadMVP_m1(cast_no, bin_data=False)
        ts = strptime(xyt_i['date'] + ' ' + xyt_i['time'], '%m-%d-%Y %H:%M:%S')
        dday = ts.tm_yday + (ts.tm_hour + ts.tm_min/60 + ts.tm_sec/(60*60))/24
        print(dday, xyt_i['cast'], xyt_i['lat'], xyt_i['lon'], file=f)
        print(cast_no, end=' ')  # update on progress

    f.close()


def flatten_to_line(lons, lats):
    """Convert lons, lats into a single distance coordinate"""
    lons, lats = map(np.asanyarray, [lons, lats])

    # Create a map so that we can convert lat/lon to x/y
    m = pickle.load(open('/home/hugke729/PhD/Maps/penny_strait.pickle', 'rb'))

    # Read in data describing transect
    lon_line, lat_line = np.genfromtxt(
        '/home/hugke729/PhD/Data/Shipboard/doc/repeat_transect_coords.csv',
        skip_header=1, delimiter=',', unpack=True)

    # Distances along line describing transect
    x_line, y_line = m(lon_line, lat_line)
    line_dists = np.cumsum(np.hypot(np.diff(x_line), np.diff(y_line)))
    line_dists = np.insert(line_dists, 0, 0)

    # Repeat x_line, y_line such that their shape is len(x_line), len(lons)
    x_line2d = np.outer(x_line, np.ones(lons.shape))
    y_line2d = np.outer(y_line, np.ones(lons.shape))

    # Calculate distances between every input coordinate and every point along
    # the line describing the transect
    x_in, y_in = m(lons, lats)
    dists = np.hypot(x_in - x_line2d, y_in - y_line2d)

    def get_line_inds(x_in, y_in, x_line, y_line, distances):
        """Get the indices of which parts of 'line' to use for scalar
        projection"""
        # index and coordinates of closest pt
        closest = np.argmin(distances)
        x_c, y_c = x_line[closest], y_line[closest]

        # coordinates of point before and after
        x_c_im1, y_c_im1 = x_line[closest - 1], y_line[closest - 1]
        x_c_ip1, y_c_ip1 = x_line[closest + 1], y_line[closest + 1]

        # angles between (x_in, y_in), (x_c, y_c), and point before or after
        # along line
        angle_im1 = angle(x_in, y_in, x_c_im1, y_c_im1, x_c, y_c)
        angle_ip1 = angle(x_in, y_in, x_c_ip1, y_c_ip1, x_c, y_c)

        # index of which other point to go with closest point
        # (based on maximising the angle between lines)
        second_ind = closest + (1 if angle_ip1 > angle_im1 else -1)

        # Edge case for points very near end of transect
        if closest == 0 and second_ind == -1:
            closest, second_ind = 0, 1

        return sorted((closest, second_ind))

    # For every input point:
    # Find the two closest points along the transect
    # Project distance along the line
    # Add projection distance to the distance up until the first point
    total_dist = np.zeros(len(lons))
    for j, (x, y) in enumerate(zip(x_in, y_in)):
        i0, i1 = get_line_inds(x, y, x_line, y_line, dists[:, j])
        x0, y0 = x_line[i0], y_line[i0]
        x1, y1 = x_line[i1], y_line[i1]

        dist_from_last_pt = scalar_projection(x0, y0, x1, y1, x, y)
        total_dist[j] = dist_from_last_pt + line_dists[i0]

    return total_dist


def two_layer_treatment(mvp_dict, x_sills, x_ranges):
    """Estimate g-prime and interface depth

    Inputs
    ------
    mvp_dict: dict
        Used throughout this file
    x_sills: 2-tuple of floats
        x position of the start and end of the sill in kilometres
    x_ranges: 2-tuple of floats
        x position of location where output is applicable

    For my transect:
        first sill: x_sills = (48, 58), x_ranges = (0, 80)
        second sill: x_sills = (75, 90), x_ranges = (65, 200)

    Returns
    -------
    gprime : 1D array
        g * delta_rho/rho
    interface_depth : 1D array
        Depth in metres of "two-layer" interface
    """

    # Simplify names of commonly used variables
    prho = mvp_dict['prho'].copy()
    z_c = mvp_dict['z_c'].copy()
    x = mvp_dict['dist_flat']

    bottom = np.array(mvp_dict['bottom'])
    sill_inds = np.where(np.logical_and(x > x_sills[0], x < x_sills[1]))[0]

    # Find interface of mode-1 wave for each profile in sill_inds
    rho_interfaces = np.zeros_like(sill_inds, dtype='float')
    for i, ind in enumerate(sill_inds):
        # Interpolate horizontal mode structure against density
        f = interp1d(mvp_dict['hori_0'][ind, :], prho[ind, :])
        # Find density of zero crossing of horizontal structure
        rho_interfaces[i] = f(0)

    rho_interface = np.nanmean(rho_interfaces)

    # Find depth of rho_interface along transect
    x_in, y_in, z_in = x, z_c, prho
    interface_depth = get_contour(x_in, y_in, z_in, rho_interface)

    # Preallocate results to keep
    gprime = np.full_like(x_in, np.nan)

    # Find average density in each layer using linear fit
    for i, rho_i in enumerate(mvp_dict['prho']):
        top_layer_inds = z_c <= interface_depth[i]
        bot_layer_inds = z_c > interface_depth[i]
        top_z = z_c[top_layer_inds]
        bot_z = z_c[bot_layer_inds]
        top_rho = prho[i, :][top_layer_inds]
        bot_rho = prho[i, :][bot_layer_inds]

        if (~top_rho.mask).sum() < 3 or (~bot_rho.mask).sum() < 3:
            # Not enough values to do linear fit
            continue

        p_top = ma.polyfit(top_z, top_rho, 1)
        p_bot = ma.polyfit(bot_z, bot_rho, 1)

        top_rho_avg = np.polyval(p_top, interface_depth[i]/2)
        bot_rho_avg = np.polyval(p_bot, (bottom[i] + interface_depth[i])/2)

        gprime[i] = 9.81*(bot_rho_avg - top_rho_avg)/bot_rho_avg

    # Remove output where inappropriate
    inappropriate_inds = np.logical_or(x < x_ranges[0], x > x_ranges[1])
    gprime[inappropriate_inds] = np.nan
    interface_depth[inappropriate_inds] = np.nan

    return rho_interface, gprime, interface_depth


def calc_froude_number(along_u, gprime, interface_depth, adcp_z, seafloor):
    """Calculate composite Froude number G"""
    G, top_u, bot_u, Fr_top_sq, Fr_bot_sq = preall(
        gprime.shape, copies=5, initial_value=np.nan)

    for i, u_i in enumerate(along_u.T):
        top_layer_inds = adcp_z <= interface_depth[i]
        bot_layer_inds = adcp_z > interface_depth[i]

        if 0 in [top_layer_inds.sum(), bot_layer_inds.sum()]:
            # Insufficient data to calculate Froude number
            continue

        top_u[i] = ma.mean(u_i[top_layer_inds])
        bot_u[i] = ma.mean(u_i[bot_layer_inds])

        Fr_top_sq[i] = top_u[i]**2/(gprime[i]*interface_depth[i])
        Fr_bot_sq[i] = bot_u[i]**2/(gprime[i]*(seafloor[i] - interface_depth[i]))

        G_squared = (Fr_top_sq[i] + Fr_bot_sq[i])

        G[i] = np.sqrt(G_squared)

    # fig, ax = plt.subplots(nrows=2, sharex=True)
    # ax[0].plot(top_u, 'b')
    # ax[0].plot(bot_u, 'r')
    # ax[1].plot(seafloor)
    return G, Fr_top_sq, Fr_bot_sq


def combine_MVP_ADCP(transect_name):
    """Put MVP and ADCP data on the same grid for individual transects

    This is useful for quantities that require velocity and density
    information such as mass fluxes or Froude numbers

    Inputs
    ------
    transect_name: str
        One of 'full_long', 'maury_repeat_X'

    Returns
    -------

    Caveats
    -------
    Not the best near edge points
    """

    mvp_processed_dir = '/home/hugke729/PhD/Data/Shipboard/MVP/transects/'
    adcp_processed_dir = '/home/hugke729/PhD/Data/Shipboard/ADCP/processed/'

    mvp_file = mvp_processed_dir + transect_name + '.p'
    adcp_file = adcp_processed_dir + transect_name + '.p'

    mvp_dict = pickle.load(open(mvp_file, 'rb'))
    adcp_dict = pickle.load(open(adcp_file, 'rb'))

    # x coordinate based on where we have MVP data
    # I would rather interpolate ADCP data to MVP locations than vice versa
    # MVP distances in km
    x = mvp_dict['dist_flat']*1000

    # z coordinate based on ADCP
    z = adcp_dict['dep']

    # Grids used for the MVP and ADCP
    mvp_X, mvp_Z = np.meshgrid(x, mvp_dict['z_c'])
    adcp_X, adcp_Z = np.meshgrid(adcp_dict['dist_s'], z)

    # Output grid
    X, Z = np.meshgrid(x, z)

    # Get vertices and weights for interpolating onto the new grid (X, Z)
    # Obviously, these wts and vtx are different for the different datasets
    vtx_mvp, wts_mvp = interp_weights((mvp_X, mvp_Z), (X, Z))
    vtx_adcp, wts_adcp = interp_weights((adcp_X, adcp_Z), (X, Z))

    def interpolate_to_new_grid(values, dataset):
        """Interpolate values on to X, Z with correct weights and vertices"""

        # Convert masked values to NaN to avoid any unexpected problems
        # values = ma.filled(values, np.nan)

        # Transpose values (otherwise interpolation will not work)
        values = values.T

        # Pick which weights to use
        if dataset == 'mvp':
            vtx, wts = vtx_mvp, wts_mvp
        elif dataset == 'adcp':
            vtx, wts = vtx_adcp, wts_adcp

        # Interpolate, reshape and return masked array
        interped_1d = interpolate(values, vtx, wts)
        interped = interped_1d.reshape(X.shape)
        return ma.masked_invalid(interped)

    u_along = interpolate_to_new_grid(adcp_dict['along_vel_s'], 'adcp')

    filterwarnings('ignore', 'invalid value*.')
    filterwarnings('ignore', '.*converting a masked element*.')

    G = dict()
    Fr_top_sq = dict()
    Fr_bot_sq = dict()

    for sill in ['sill_1', 'sill_2']:
        G[sill], Fr_top_sq[sill], Fr_bot_sq[sill] = calc_froude_number(
            u_along, mvp_dict[sill]['gprime'],
            mvp_dict[sill]['interface_depth'], z, mvp_dict['bottom'])

    out_dict = dict(X=X, x=X[0, :], z=Z[:, 0], Z=Z, G=G)
    out_dir = '/home/hugke729/PhD/Data/Shipboard/combined_mvp_adcp/'
    pickle.dump(out_dict, open(out_dir + transect_name + '.p', 'wb'))

    return G, Fr_top_sq, Fr_bot_sq


def quantify_effect_smoothing_freq():
    """Test how much total dissipation changes by smoothing density to
    2, 3, 4, and 6 Hz"""
    casts = np.r_[45:1150:3]
    eps_all = np.full((4, casts.size), np.nan)
    for i, cast in enumerate(casts):
        try:
            print(cast, end=' ')
            _, data = loadMVP_m1(cast, bin_data=False)

            for j, n_smooth_rho in enumerate([4, 6, 8, 12]):
                eps, Lt = calc_eps(
                    data['p_raw'], data['prho'], data['z'],
                    plot_overturns=False, n_smooth_rho=n_smooth_rho)
                eps_all[j, i] = eps.sum()
        except IndexError:
            pass

    print('\n\nCompared to low-passing at 3Hz, filtering at a different freq\n'
          'produces dissipation values of the following relative magnitude:')
    relative_mags = [ma.mean(ma.masked_invalid(eps_all[i]/eps_all[2]))
                     for i in [0, 1, 3]]
    print("""
    6 Hz: {0:2.2f}
    4 Hz: {1:2.2f}
    2 Hz: {2:2.2f}

    """.format(*relative_mags))
    return eps_all


if __name__ == '__main__':
    for cast in np.r_[372]:
        try:
            xyt, data, binned = loadMVP_m1(cast, bin_data=True)
            # plt.plot(data['prho'], data['z'])
        except IndexError:
            pass

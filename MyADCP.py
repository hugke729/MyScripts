# Script to process ADCP data
# For summary of steps, see the function process ADCP data
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pickle
import glob
import numpy as np
import numpy.ma as ma
from pycurrents.adcp.rdiraw import Multiread
from MyPlotFunctions import flipy, fix_colorbar
from MyInterp import smooth1d_with_holes
from MyMVP import flatten_to_line
from MyInteractive import disp_latlon
from MyMapFunctions import haversines
from MyNumpyTools import cosd, ma_mad

# Overview of functions

# concatenate_binned_arrays
#   |- process_adcp
#   |  |- read_data
#   |  |- truncate_data
#   |  |- subtract_vessel_velocity
#   |  |- get_latlon
#   |  |- mask_seafloor
#   |  |- along_component
#   |  |- smooth_output
#   |  |- create_dist_coord
#   |  |- clean_data_dict
#   |  |- save_data


def read_data(use_pickle=True):
    """Convert .ENX files to python dict using CODAS' Multiread function"""
    if use_pickle:
        # Marginally faster to just open from pickle rather than use
        # Multiread function again
        data = pickle.load(open('/home/hugke729/PhD/Data/Shipboard/ADCP/raw/' +
                                'all_raw_data.pickle', 'rb'))
    else:
        data_loc = '/home/hugke729/PhD/Data/Shipboard/ADCP/raw/'
        file_base = 'ArcticNet_1503__009_'
        # Files ending in 0000XX.ENX where XX is >= 20 are what I want
        files_in = sorted(glob.glob(data_loc + file_base + '0000[2-3]*.ENX'))
        m = Multiread(files_in, 'os')
        data = m.read(step=1)
        pickle.dump(data, open('/home/hugke729/PhD/Data/Shipboard/ADCP/raw/' +
                               'all_raw_data.pickle', 'wb'))
    return data


def save_data(data, name='temp'):
    """Save data to pickle"""
    work_folder = '/home/hugke729/PhD/Data/Shipboard/ADCP/processed/'
    with open(work_folder + name + '.p','wb') as f:
        pickle.dump(data, f)


def get_dn(data, dtSmooth=600):
    """Determine the number of points to smooth over"""
    # dt = (data.dday[-1] - data.dday[0])*86400/(data.dday.shape)
    dt = np.array(9.7)  # seconds between measurements (calculated from above)
    nSmooth = (dtSmooth/dt).astype(int)
    return nSmooth


def subtract_vessel_velocity(raw_vel, bt_vel):
    """Subtract bottom track velocity from absolute velocity"""
    vel = raw_vel - bt_vel[:, np.newaxis, :]
    return vel


def along_component(vel_array, heading, lat, is_long_section=True):
    """Convert velocities to along-transect componento

    Currently only computed for long-sound transects
    """

    U, V = vel_array[:, :, :2].T
    vel_mag = np.hypot(U, V)

    # Heading is like bearing, but with range -180 to 180
    # Convert velocity direction to this same coordinate system
    vel_heading = (90 - np.rad2deg(np.arctan2(V, U)))*np.sign(U)
    cos_theta = cosd(vel_heading - heading)

    if np.mean(np.diff(lat)) > 0:
        # Transect is northward
        cos_theta = -cos_theta

    along_vel = (vel_mag*cos_theta).T

    if not is_long_section:
        # Along-transect velocity needs work for cross-sections
        # Fill it with NaNs to ensure it isn't used incorrectly
        along_vel = np.full(along_vel.shape, np.nan)

    return along_vel


def smooth_output(data, downsample_n=True):
    """Smooth output with with moving average

    Smoothed values have _s appended to their name
    If true, downsample_n will take all smoothed values and return only two
    values every nSmooth elements"""
    nSmooth = get_dn(data)

    def smooth_func(arr):
        return smooth1d_with_holes(arr, nSmooth)

    # Pre-allocate smoothed results
    data['vel_s'] = np.full(data['vel'].shape, np.nan)
    data['along_vel_s'] = np.full(data['along_vel'].shape, np.nan)
    data['depth_s'] = np.full(data['bt_depth'].shape[0], np.nan)

    # Do the actual smoothing.
    # Treat U and V separately, and each depth bin seperately
    for comp, lev in np.ndindex(2, data.nbins):
        raw_vel = data['vel'][:, lev, comp]
        data['vel_s'][:, lev, comp] = smooth_func(raw_vel)

    # Smooth along-channel velocity
    for lev in range(data.nbins):
        along_vel_i = data['along_vel'][:, lev]
        data['along_vel_s'][:, lev] = smooth_func(along_vel_i)

    # Smooth depth and heading
    depth = np.mean(data['bt_depth'], axis=1)  # mean from four pingers
    data['depth_s'], data['heading_s'] = map(
        smooth_func, [depth, data['nav_heading']])

    # Smooth and then downsample lat/lon
    data['lat_s'], data['lon_s'] = map(smooth_func, [data['lat'], data['lon']])

    # return masked results and downsample if required
    for key in [key for key in data.keys() if key.endswith('_s')]:

        data[key] = ma.masked_invalid(data[key])
        if downsample_n:
            data[key] = data[key][::(nSmooth//2), ...]

    return data


def depth_averaged_velocity(along_vel_s):
    """
    Inputs
    ------
    along_vel_s : 2D Masked Array
    """
    return ma.mean(along_vel_s, axis=1)


def get_latlon(data):
    """Get longitude and latitude of coordinates of pings"""
    lat = 0.5*(data['nav_end_txy'][:, 2] + data['nav_start_txy'][:, 2])
    lon = 0.5*(data['nav_end_txy'][:, 1] + data['nav_start_txy'][:, 1])
    return lat, lon


def cast_number_to_month_day(cast_no):
    """Convert MVP cast number to day of September 2015

    Makes it easy to compare with ADCP"""
    assert 37 <= cast_no <= 1150, 'Cast number out of range'
    file_name = 'time_cast_lat_lon.txt'
    dday_v_cast_file = '/home/hugke729/PhD/Data/Shipboard/doc/' + file_name
    dday, cast = np.loadtxt(dday_v_cast_file, unpack=True)[:2]
    ind = np.where(cast == cast_no)[0]
    sep1 = 243.
    return dday[ind] - sep1


def ctd_cast_number_to_month_day(cast_no):
    """As for cast_number_to_month_day but using ctd cast numbers"""
    assert 93 <= cast_no <= 122, 'Cast number out of range'
    file_name = 'ctd_time_v_cast.txt'
    dday_v_cast_file = '/home/hugke729/PhD/Data/Shipboard/doc/' + file_name
    dday, cast = np.loadtxt(dday_v_cast_file, unpack=True)
    ind = np.where(cast == cast_no)[0]
    sep1 = 243.
    return dday[ind] - sep1


def map_plot(ax, lon, lat):
    """Map showing transect"""
    m = pickle.load(open('/home/hugke729/PhD/Maps/penny_strait.pickle', 'rb'))
    m.plot(lon, lat, ax=ax, latlon=True)
    m.fillcontinents(ax=ax)
    return m


def truncate_data(data, start, end, lim_type='cast'):
    """Get only the transect data we currently want

    Inputs
    ------
    start and end can be either
        day of September 2015 (lim_type = 'month_day')
        MVP cast number (lim_type = 'cast')
        CTD cast number (lim_type = 'ctd')
    """
    sep1 = 243.
    monthDays = data.dday - sep1 + 1  # data.dday is zero-based (I think)

    if lim_type == 'cast':
        # Convert cast numbers to day of September
        start, end = map(cast_number_to_month_day, [start, end])
    elif lim_type == 'ctd':
        start, end = map(ctd_cast_number_to_month_day, [start, end])

    # truncate
    inds = np.where(np.logical_and(monthDays > start, monthDays < end))[0]
    for key, value in data.items():
        try:
            data[key] = value[inds, ...]
        except (IndexError, AttributeError, TypeError):
            pass

    return data


def create_dist_coord(data, is_long_section=False):
    """Flatten long sections to single distance coordinate as per MVP
    Or convert lat, lon to distance if cross section"""

    # data['dist_s'] will be in metres
    if is_long_section:
        data['dist_s'] = flatten_to_line(data['lon_s'], data['lat_s'])
    else:
        data['dist_s'] = haversines(data['lon_s'], data['lat_s'])[0]*1e3

    # Ensure that dist increases from west to east  for cross-sections
    if np.mean(np.diff(data['lon_s'])) < 0 and not is_long_section:
        data['dist_s'] = data['dist_s'].max() - data['dist_s']

    dist_inds = np.argsort(data['dist_s'])
    for key, value in data.items():
        if key.endswith('_s'):
            data[key] = value[dist_inds]
    return data


def mask_seafloor(data):
    """Mask velocities below the seafloor and bad values above it

    For velocities above seafloor, check for bad data using median and
    median absolute deviation and distance above seafloor as parameters
    """

    # Below seafloor
    seafloor = np.mean(data['bt_depth'], axis=1)
    dist_from_seafloor = data['dep'] - seafloor[:, np.newaxis]
    mask = dist_from_seafloor > 0
    mask3d = mask[..., np.newaxis]*np.ones(mask.shape + (4,))

    data['vel'] = ma.masked_where(mask3d, data['vel'])

    # Remove values that appear incorrect as a result of being close to the
    # seafloor
    u_mag = np.hypot(data['vel'][..., 0], data['vel'][..., 1])
    median_vel = ma.median(u_mag, axis=1)[..., np.newaxis]
    mad_vel = ma_mad(u_mag, axis=1)[..., np.newaxis]
    badness_score = np.abs(u_mag - median_vel)/mad_vel

    # Scale distance of 8m corresponds to ~5 cells above seafloor that stand
    # a chance of being masked
    badness_score *= np.exp(-np.abs(dist_from_seafloor)/8)

    # Normalise
    badness_score = badness_score/np.sum(badness_score, axis=1)[:, np.newaxis]

    # Don't mask anything more than 30m above seafloor
    seafloor_mask = badness_score > 0.1
    seafloor_mask.data[dist_from_seafloor > 30] = False

    # Total mask
    mask = np.logical_or(mask, seafloor_mask)

    # Apply mask
    data['vel'].mask = mask[..., np.newaxis]*ma.ones((1, 1, 4))

    return data


def clean_data_dict(data):
    """Remove parts of data I won't ever use in order to keep things clean"""

    rm_list = ['Bin1Dist', 'Blank', 'FL', 'NBeams', 'NPings', 'Pulse', 'VL',
               'amp', 'amp1', 'amp2', 'amp3', 'amp4', 'cor', 'cor1', 'cor2',
               'cor3', 'cor4', 'ens_num', 'nav_PC_minus_UTC', 'nprofs',
               'pg', 'pg1', 'pg2', 'pg3', 'pg4', 'pingtype', 'pitch', 'rVL',
               'rawnav', 'roll', 'sysconfig', 'temperature', 'trans',
               'vel1', 'vel2', 'vel3', 'vel4', 'yearbase']
    for key in rm_list:
        try:
            data.pop(key)
        except KeyError:
            pass

    return data


def plot_section(data, is_long_section=False):
    """Plot smoothed ADCP data together with map"""
    fig, axs = plt.subplots(ncols=2, figsize=(10, 4))

    adcp_pcolor(data, is_long_section, ax=axs[0])

    # Show transect on map
    m = map_plot(axs[1], data['lon_s'], data['lat_s'])
    disp_latlon(axs[1], m)
    m.scatter(data['lon_s'][0], data['lat_s'][0], 10)

    plt.show()
    return fig, axs


def process_adcp(start, end, lim_type='cast', is_long_section=False,
                 save_name=None):
    data = read_data(False)
    data = truncate_data(data, start, end, lim_type=lim_type)
    data['vel'] = subtract_vessel_velocity(data['vel'], data['bt_vel'])
    data['lat'], data['lon'] = get_latlon(data)
    data = mask_seafloor(data)
    data['along_vel'] = along_component(
        data['vel'], data['heading'], data['lat'], is_long_section)
    data = smooth_output(data)
    data = create_dist_coord(data, is_long_section)
    data = clean_data_dict(data)
    data['u_along_zavg'] = depth_averaged_velocity(data['along_vel_s'])
    if save_name is not None:
        save_data(data, save_name)
    return data


# if __name__ == '__main__':
    # data = process_adcp(78, 371, lim_type='cast', is_long_section=True)
    # plot_section(data, is_long_section=True)

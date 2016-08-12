import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from seawater.eos80 import ptmp
from MyNumpyTools import minmax
from MySysUtils import preall
from MyMapFunctions import haversines
from MyFunctions import central_diff_gradient
from MyInterp import smooth1d_with_holes as smooth


def cast_number_to_filename(cast_number):
    data_loc = '/home/hugke729/PhD/Data/Shipboard/CTD/avg/'
    file_name_base = '1503'
    file_in = data_loc + file_name_base + str(cast_number).zfill(3) + 'avg.cnv'
    return file_in


def read_cnv(cast_number):
    """Take cnv file for specified cast and return only the outputs I want

    Output
    ------
    Pressure (db)
    Temperature (deg C)
    Salinity (PSU)
    sigma-t (kg/m^3)"""
    file_in = cast_number_to_filename(cast_number)

    # Data always starts on line 295
    p, T, S, sigma, lat, lon = np.genfromtxt(
        file_in, skip_header=294, unpack=True, usecols=[0, 1, 2, 13, 15, 16])

    return p, T, S, sigma


def get_xyt(cast_numbers):
    """Get position, location, and depth information from summary

    Returns lat, lon, utc_time, and maximum pressure without having to read
    all the data in"""

    data_loc = '/home/hugke729/PhD/Data/Shipboard/CTD/processed/'
    D = pickle.load(open(data_loc + 'xyt_summary.pickle', 'rb'))

    # Return tuple or list of types depending on what cast_numbers is
    try:
        return D[cast_numbers]
    except TypeError:
        return [D[i] for i in cast_numbers]


def ctd_map():
    m = pickle.load(open('/home/hugke729/PhD/Maps/penny_strait.pickle', 'rb'))
    m.fillcontinents(color='grey')
    casts = np.r_[92:123]
    xyts = get_xyt(casts)
    for i, (lat, lon, _, _) in enumerate(xyts):
        plt.annotate(xy=m(lon, lat), s=str(casts[i] % 100).zfill(2),
                     ha='center', fontsize=9)


def p_to_z(p, sigma):
    """Make the minor change from pressure (dbar) to depth (m)"""
    dbar_to_pa = 10000
    z = np.cumsum(1/((sigma+1000)*9.81)*central_diff_gradient(p))*dbar_to_pa
    # ...avg.cnv files all start at p = 2 db
    # for reasons I can't think how to explain, need to add 1 to z
    # To do with constant of integration in z = int dp/ρg
    z = z + 1
    return z


def T_to_theta(S, T, p):
    """Make the minor change from in situ temperature to potential temp"""
    return ptmp(S, T, p)


def get_theta_S_sigma_z(cast_number):
    """Read .cnv file, and convert p to z and T to theta"""
    p, T, S, sigma = read_cnv(cast_number)
    z = p_to_z(p, sigma)
    theta = T_to_theta(S, T, p)
    return theta, S, sigma, z


def calc_N2(sigma, z):
    """Get buoyancy frequency squared from sigma"""
    N2 = (1000+sigma)/np.mean(sigma)*central_diff_gradient(sigma, z)
    N2 = smooth(N2, 6)
    return N2


def grid_section(cast_list):
    """Put all casts in cast_list into grids for each quantity"""
    lats, lons, pmaxs = preall(len(cast_list), copies=3)
    # Create vector of positions and convert to distance
    for i, (lat, lon, _, pmax) in enumerate(get_xyt(cast_list)):
        lats[i], lons[i], pmaxs[i] = lat, lon, pmax

    # Maximum depth among all casts
    pmax_all = np.ceil(max(pmaxs)).astype(int)

    # Create 2d depth array
    dist = haversines(lons, lats)[0]
    dist = np.outer(np.ones(pmax_all), dist)

    # g for grid
    pg, Tg, Sg, sigmag = preall((pmax_all, len(cast_list)),
                                copies=4, initial_value=np.nan)

    for i, cast_number in enumerate(cast_list):
        p, T, S, sigma = read_cnv(cast_number)
        x = len(p)
        pg[:x, i], Tg[:x, i], Sg[:x, i], sigmag[:x, i] = p, T, S, sigma

    return pg, Tg, Sg, sigmag, dist


def get_velocity_profile(cast_number):
    """Read in preprocessed (I believe) lowered-ADCP data"""
    fname = ('/home/hugke729/PhD/Data/Shipboard/CTD/LADCP/Traitement/stn' +
             str(cast_number).zfill(3) + '/processed/stn' +
             str(cast_number).zfill(3) + '.mat')
    D = loadmat(fname)
    # Not sure why the indexing or the key being dr actually means,
    # but these values correspond to the plots in the CTD folder
    u = D['dr']['u'][0][0][:, 0]
    v = D['dr']['v'][0][0][:, 0]
    z = D['dr']['z'][0][0][:, 0]
    return u, v, z


def plot_cross_section(cast_list):
    pg, Tg, Sg, sigmag, dist = grid_section(cast_list)
    fig, ax = plt.subplots()
    cax = plt.contourf(dist, pg, sigmag, levels=np.arange(24.5, 27, 0.25),
                       cmap='afmhot_r')
    cbar = plt.colorbar(cax)
    cbar.ax.invert_yaxis()
    ax.set(xlim=(-1, np.max(dist) + 1), ylim=minmax(pg)[::-1])
    for d in dist[0, :]:
        ax.plot((d,  d), (2, np.nanmax(pg)), color='grey', zorder=5)
        plt.draw()

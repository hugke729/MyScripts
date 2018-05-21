import pickle
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy.stats import mode
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.ndimage.filters import gaussian_filter
from MyMapFunctions import haversines
from MyMITgcmUtils import remove_nans_laterally as interp_nans
from MyColormaps import truncate_colormap
from MyMVP import flatten_to_line
from MyInterp import nan_gaussian_filter
from MyFunctions import get_contour


def load_parsed_data(file_in, freq):
    """
    Load EK60 data that has been parsed but otherwise not post processed

    Input
    -----
    file_in: str
        Of the form D201509dd-Thhmmss
    freq: int
        Frequency in kHz. One of 38, 120, or 200

    Returns
    -------
    D: numpy record array
        Relevant fields: range, Sv, power, latitude, longitude
    """
    base_dir = '/home/hugke729/PhD/Data/Shipboard/EK60/parsed/'
    data_dir = base_dir + str(freq) + 'kHz/'
    D = loadmat(data_dir + file_in + '.mat', squeeze_me=True)['x']
    return D


def rec_array_to_dict(D):
    """
    Convert data from numpy record array to standard arrays
    """

    # Some files change the resolution part way through.
    # Reasonable simplification (albeit losing small bit of data) is to
    # take the biggest chunk where resolution is unchanging
    N_per_profile = [x.size for x in D['range']]
    inds = N_per_profile == mode(N_per_profile)[0][0]

    # 1D arrays
    lat, lon = [D[key][inds] for key in ['latitude', 'longitude']]
    lat, lon = [interp_nans(arr.astype('float')) for arr in [lat, lon]]
    z = D['range'][inds][0]
    time = D['time'][inds]

    # 2D arrays
    Sv = np.column_stack(D['Sv'][inds])
    power = np.column_stack(D['power'][inds])

    return dict(z=z, lat=lat, lon=lon, power=power, Sv=Sv, time=time)


def fix_latlon(D):
    """Fix raw GPS data that imply that ship's position doesn't change for
    periods of time. Easily identifiable as times when difference in lat and
    lon are identically zero

    Needs extra work for repeat 4
    """
    lat, lon = D['lat'], D['lon']
    N = len(lat)
    dlat = np.diff(lat)
    changing = np.insert(dlat != 0, 0, True)
    D['lat'] = np.interp(np.r_[:N], np.where(changing)[0], lat[changing])
    D['lon'] = np.interp(np.r_[:N], np.where(changing)[0], lon[changing])
    return D


def calc_dist(D):
    if np.any(D['lat'] > 75.7):
        # This is crude check to see if it is along-channel transect
        try:
            dist = flatten_to_line(D['lon'], D['lat'])/1000
        except IndexError:
            # Some files have points beyond line used in 'flatten_to_line'.
            # These don't currently work, but could with some effort
            dist = haversines(D['lon'], D['lat'])[0]
            print('Distance coordinate not quite right')
    else:
        dist = haversines(D['lon'], D['lat'])[0]
        if D['lon'][0] - D['lon'][-1] > 0:
            dist = dist[::-1]
    return dist


def mask_seafloor(D):
    """
    Use known bathymetry to mask Sv and power

    Bit too chunky without more work
    """

    # Load in gridded bathymetry
    bathy_file = '/home/hugke729/PhD/Data/Penny_Strait/Penny_strait.mat'
    Dz = loadmat(bathy_file)
    lon, lat, z = [Dz['topo' + key] for key in ['lon', 'lat', 'depth']]
    z_interp = rgi((lon[:, 0], lat[0, :]), z)

    z_along_line = z_interp(np.c_[D['lon'], D['lat']])

    Z = D['z'][:, np.newaxis]*np.ones_like(D['Sv'])
    z_below_bot = Z > z_along_line

    D['power'] = ma.masked_where(z_below_bot, D['power'])
    D['Sv'] = ma.masked_where(z_below_bot, D['Sv'])

    return D


def plot_seafloor(ax, transect):
    """Overlays gray polygon to mask anything below seafloor.

    Only currently works for along-channel transects"""
    if 'repeat' in transect or 'long' in transect:
        d_vs_z_fname = '/home/hugke729/PhD/Data/Shipboard/MVP/transect_depth.txt'
        dist, depth = np.genfromtxt(d_vs_z_fname, unpack=True, delimiter=',')
        ax.autoscale(False)
        ax.fill_between(dist, 250, depth, color='grey', zorder=10)


def get_Dmvp(transect):
    mvp_fname = '/home/hugke729/PhD/Data/Shipboard/MVP/transects/'
    mvp_fname += transect + '.p'
    Dmvp = pickle.load(open(mvp_fname, 'rb'))
    return Dmvp


def overlay_density_contours(ax, transect):
    """Only currently works properly for along-channel transects"""
    # Before smoothing, change to evenly spaced x
    Dmvp = get_Dmvp(transect)
    x = Dmvp['dist_flat'] if 'dist_flat' in Dmvp else Dmvp['dist']
    y, Z = Dmvp['z_c'], Dmvp['prho'].T - 1000
    ax.plot(x, 5*np.ones_like(x), 'w|', zorder=5)

    if x[-1] - x[0] < 0:
        inds = np.argsort(x)
        x = x[inds]
        Z = Z[:, inds]

    # Standard contour method is poor in comparison to more manual method here
    # x_out = np.arange(x.min(), x.max(), 1)
    # Xo, Yo = np.meshgrid(x_out, y, indexing='ij')
    # sigma = rgi((x, y), Z.T)((Xo, Yo)).T

    # cons = get_contour(x_out, y, sigma, levels=np.r_[26:26.75:0.25])
    # for line in cons.T:
    #     ax.plot(x_out, nan_gaussian_filter(line, 0), c='k')

    ax.contour(x, y, Z, levels=np.r_[26:26.75:0.1], colors='k')


def plot_summary(D, transect, pcolor=False, power=False, sigma_smooth=2,
                 quick_pcolor=True):
    map_file = '/home/hugke729/PhD/Python/Maps/penny_strait.pickle'
    m = pickle.load(open(map_file, 'rb'))

    fig, axs = plt.subplots(ncols=2, gridspec_kw=dict(width_ratios=(2, 1)),
                            figsize=(8.6, 2.5))
    cmap = truncate_colormap('afmhot_r', minval=0.2)

    arr = D['power'] if power else D['Sv']
    if np.diff(D['lat'][[-1, 0]]) < 0:
        arr = arr[:, ::-1]
        D['lon'], D['lat'] = D['lon'][::-1], D['lat'][::-1]
    arr = gaussian_filter(arr, sigma_smooth)
    if power:
        color_opts = dict(vmin=-160, vmax=-110, cmap=cmap)
    else:
        color_opts = dict(vmin=-90, vmax=-65, cmap=cmap)

    # D['dist'] = calc_dist(D)
    # New distance calc not thoroughly tested
    D['dist'] = EK60_distance_from_time_matching(transect)

    if pcolor:
        if quick_pcolor:
            x, y, Z = D['dist'][::5], D['z'][::5], arr[::5, ::5]
        else:
            x, y, Z = D['dist'], D['z'], arr

        cax = axs[0].pcolormesh(x, y, Z, **color_opts)
        axs[0].set(ylim=(250, 0))
        plot_seafloor(axs[0], transect)
        overlay_density_contours(axs[0], transect)
    else:
        axs[0].invert_yaxis()
        cax = axs[0].imshow(arr, aspect=4, **color_opts)
    m.fillcontinents(ax=axs[1])

    m.plot(D['lon'], D['lat'], latlon=True)
    m.plot(D['lon'][0], D['lat'][0], latlon=True, marker='o')
    cbar = fig.colorbar(cax)
    cbar_lab = 'Power' if power else 'Volume backscattering'
    cbar.set_label(cbar_lab)

    return fig, axs, cax


def define_files_for_transect():
    """Define which files are needed to cover each transect"""
    files_for_transect = dict(
        wellington_cross_1=[
            'D20150926-T070020'],
        full_long=[
            'D20150926-T070020',
            'D20150926-T134201',
            'D20150926-T203334',
            'D20150927-T032659',
            'D20150927-T165823'],
        maury_repeat_1=[
            'D20150927-T202303',
            'D20150928-T001538',
            'D20150928-T051735'],
        maury_repeat_2=[
            'D20150928-T051735',
            'D20150928-T143835',
            'D20150928-T143936'],
        maury_repeat_3=[
            'D20150928-T143835',
            'D20150928-T143936',
            'D20150929-T170801',
            'D20150928-T171001'],
        maury_repeat_4=[
            'D20150928-T171001',
            'D20150928-T193711',
            'D20150928-T220714'],
        maury_repeat_5=[
            'D20150928-T220714',
            'D20150929-T003718',
            'D20150929-T030545'],
        wellington_cross_2=[
            'D20150929-T030545',
            'D20150929-T170826'],
        wellington_cross_3=[
            'D20150929-T170826',
            'D20150929-T201341'])

    files_for_transect = dict(
        wellington_cross_1=[
            ('D20150926-T070020', np.s_[270:3300])],
        full_long=[
            ('D20150926-T070020', np.s_[3280:]),
            ('D20150926-T134201', np.s_[:6256]),
            ('D20150926-T203334', np.s_[472:5620])],
        maury_repeat_1=[
            ('D20150927-T202303', np.s_[515:]),
            ('D20150928-T001538', np.s_[:]),
            ('D20150928-T051735', np.s_[:1087])],
        # Repeat 2 missing data
        maury_repeat_2=[
            ('D20150928-T051735', np.s_[1087:])],
        maury_repeat_3=[
            ('D20150928-T143936', np.s_[:]),
            ('D20150928-T171001', np.s_[:1498])],
        maury_repeat_4=[
            ('D20150928-T171001', np.s_[2740:]),
            ('D20150928-T193711', np.s_[:]),
            ('D20150928-T220714', np.s_[:774])],
        maury_repeat_5=[
            ('D20150928-T220714', np.s_[830:]),
            ('D20150929-T003718', np.s_[:1621])],
        wellington_cross_2=[
            ('D20150929-T170826', np.s_[:2097])],
        wellington_cross_3=[
            ('D20150929-T170826', np.s_[2097:]),
            ('D20150929-T201341', np.s_[:2316])],
        lancaster_sound=[
            ('D20150930-T022442', np.s_[2230:]),
            ('D20150930-T053003', np.s_[:]),
            ('D20150930-T083528', np.s_[:2445])])
    return files_for_transect


def cat_files_to_transect(freq):
    files_for_transect = define_files_for_transect()
    print('Working on: ')
    for transect, v in files_for_transect.items():
        print(transect)
        for i, (file_in, inds) in enumerate(np.array(v)):
            D = load_parsed_data(file_in, freq)
            D = rec_array_to_dict(D)
            D = fix_latlon(D)

            for k in ['power', 'Sv']:
                D[k] = D[k][:, inds]
            for k in ['lat', 'lon']:
                D[k] = D[k][inds]
            for k in ['time']:
                D[k] = D[k][inds]

            # Join files
            if i == 0:
                D_all = D
            else:
                for k, v in D.items():
                    if k == 'z':
                        continue
                    if v.ndim == 1:
                        D_all[k] = np.r_[D_all[k], v]
                    else:
                        D_all[k] = np.c_[D_all[k], v]

        out_dir = '/home/hugke729/PhD/Data/Shipboard/EK60/transects/'
        fname = out_dir + str(freq) + 'kHz/' + transect
        savemat(fname, D_all)


def all_processing():
    for freq in [38, 120, 200]:
        cat_files_to_transect(freq)


def load_processed_data(transect_name, freq):
    base_dir = '/home/hugke729/PhD/Data/Shipboard/EK60/transects/'
    base_dir += str(freq) + 'kHz/'
    fname = base_dir + transect_name + '.mat'
    return loadmat(fname, squeeze_me=True)


def MVP_date_time_to_timeval(Dmvp):
    """Convert MVP's date and time strings to decimal days after Sep 1, 2015

    Needed to compare with echosounder time

    Dmvp['date'] in form 'MM-DD-YYYY'
    Dmvp['time'] in form 'HH:MM:SS'
    """
    day = np.array([np.float(d[3:5]) for d in Dmvp['date']])
    hr, mn, sc = [np.array([np.float(d[i]) for d in Dmvp['time']])
                  for i in (np.s_[:2], np.s_[3:5], np.s_[6:])]
    t = day + hr/24 + mn/(60*24) + sc/(60**2*24)
    return t


def EK60_date_time_to_timeval(D):
    """Calculate equivalent time value returned by counterpart MVP function
    above

    D['time'] in form 'DD-MMM-YYYY HH:MM:SS.ssss...'
    """
    day, hr, mn, sc = [
        np.array([np.float(d[i]) for d in D['time'].flatten()])
        for i in (np.s_[:2], np.s_[12:14], np.s_[15:17], np.s_[18:])]

    t = day + hr/24 + mn/(60*24) + sc/(60**2*24)
    return t


def EK60_distance_from_time_matching(transect):
    """Instead of getting distance from lat/lon, interpolate time vectors
    using MVP distance vector as starting point
    """
    Dmvp = get_Dmvp(transect)
    dist_m = Dmvp['dist_flat']

    tm = MVP_date_time_to_timeval(Dmvp)
    te = EK60_date_time_to_timeval(D)

    flipy_dist_e = False
    if np.diff(tm[[0, -1]]) < 0:
        # Ensure np.interp works
        tm, dist_m = tm[::-1], dist_m[::-1]
        flipy_dist_e = True

    dist_e = np.interp(te, tm, dist_m)
    if flipy_dist_e:
        dist_e = dist_e[::-1]
    return dist_e


if __name__ == '__main__':
    for transect in ['full_long', 'maury_repeat_1']:
    # for transect in ['maury_repeat_1']:
        D = load_processed_data(transect, 120)
        fig, axs, cax = plot_summary(
            D, transect, pcolor=True, power=False, sigma_smooth=(0, 2),
            quick_pcolor=False)
        axs[0].set_title(transect)
        axs[0].set(xlim=(50, 110), ylim=(150, 0), yticks=np.r_[0:201:50])
        # raster_and_save(transect + '.svg', cax, dpi=300)

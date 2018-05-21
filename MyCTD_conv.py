# Script to process .cnv CTD data
# Except for processing for archiving, ...
# All I really want from this is the depth-averaged dissipation rates
# I use the overturn code from MyMVP since I originally wrote the code for
# the MVP. I checked by comparing the results to the Matlab code in
# https://github.com/OceanMixingCommunity/Standard-Mixing-Routines
# Results are in reasonable agreement given the different tests used.
# Also I'm only reporting values to 1sf.
import numpy as np
from scipy.signal import csd, coherence
from seawater.eos80 import salt, pden
from seawater.constants import c3515
from MyFilters import butter_lowpass_filter
from MyPlotFunctions import get_xy_line_data
from MyMVP import calc_eps

data_dir = '/home/hugke729/PhD/Data/Shipboard/CTD/converted/'


def select_down_inds(p, cast_num):
    psm3 = butter_lowpass_filter(p, 3, 24)
    inds = np.gradient(psm3) > 0.01
    inds[len(inds)//2:] = False

    return inds


def select_up_inds(p, cast_num):
    # Checking upcast useful for troubleshooting
    psm3 = butter_lowpass_filter(p, 3, 24)
    inds = np.gradient(psm3) < -0.01

    inds[:np.argmax(psm3)] = False

    return inds


def read_cast(cast_num, downcast=True):
        fname = data_dir + '1503' + str(cast_num).zfill(3) + '.cnv'
        data = np.genfromtxt(
            fname, skip_header=263, usecols=[0, 1, 6, 7, 8, 10], unpack=True)
        p = data[3]

        if downcast:
            inds = select_down_inds(p, cast_num)
        else:
            inds = select_up_inds(p, cast_num)

        C, sigma, theta, p, S, T = [row[inds] for row in data]
        C *= 10  # Convert to same units reported by MVP
        return C, sigma, theta, p, S, T


def get_shift(regenerate=False):
    """Possibly needs work on robust linear fit, but close enough

    Shift given as number of samples"""
    if regenerate:
        plt.figure()
        for cast_num in np.r_[92:123]:
            C, sigma, theta, p, S, T = read_cast(cast_num)
            # If uncommented, the line below shifts temperature, and so
            # output graph should have data along horizontal line
            # S, T, theta = shift_T(C, theta, T, p)
            f, csd_f = csd(C, theta, fs=24, nperseg=256)
            plt.plot(f, np.angle(csd_f),
                     color='k', alpha=0.7, marker='+', ls='none')

        # For checking shift
        fs, angles = get_xy_line_data(plt.gca(), sort_x=True)
        wts = (12.5 - fs)/12.5
        p = np.polyfit(fs, angles, 1, w=wts)
        shift = p[0]
        plt.plot(fs, angles, 'k+')
        plt.plot(np.r_[0, 12.5], np.polyval(p, np.r_[0, 12.5]))

        print('To find shift, get slope of best fit line, then ×24/2pi')
        print('Linear fits do not work well enough here. Need an iterative' +
              'procedure')
        return shift*24/(2*np.pi)
    else:
        return -0.42


def plot_coherence():
    casts = np.r_[92:123]
    # 33 comes from nperseg of 64
    coheres = np.zeros((len(casts), 33))
    for i, cast_num in enumerate(casts):
        C, sigma, theta, p, S, T = read_cast(cast_num)
        f, cohere = coherence(C, T, fs=24, nperseg=64)
        coheres[i] = cohere

    cohere_bins = np.r_[0:1:11j]
    cohere_f = np.zeros((f.size, cohere_bins.size - 1))
    for i, cohere_i in enumerate(coheres.T):
        cohere_f[i] = np.histogram(cohere_i, cohere_bins, density=True)[0]

    # Exact details need work. Mostly wanted to see at what frequency
    # coherence clearly starts to decrease
    # I'd say 4Hz
    plt.pcolormesh(f, cohere_bins, cohere_f.T, cmap='afmhot_r', vmax=4)
    return coheres


def shift_T(T, theta):
    lag = get_shift()
    samples = np.arange(len(T))
    T = np.interp(samples, samples + lag, T)
    theta = np.interp(samples, samples + lag, theta)
    return T, theta


def smooth_params(C, theta, T, p):
    C, T, theta, p = [butter_lowpass_filter(X, 4, 24)
                      for X in [C, T, theta, p]]
    return C, T, theta, p


def recalc_S_and_sigma(C, T, p):
    S = salt(C/c3515, T, p)
    sigma = pden(S, T, p) - 1000
    return S, sigma


def remove_pressure_reversals(p_raw, C, sigma, theta, p, S, T):
    """Filter pressure at 4 times scan rate (recommended by Seabird I believe)
    and check for when pressure is increasing on downcast

    Do this after other processing makes most sense I think

    Simply removing anything that is going up. Good enough approximation here.
    """
    psm6 = butter_lowpass_filter(p_raw, 6, 24)
    reversal = psm6 < np.roll(psm6, 1)
    reversal[0] = False
    C, sigma, theta, p, S, T = [
        X[~reversal] for X in [C, sigma, theta, p, S, T]]

    return C, sigma, theta, p, S, T


def below_surface_influence(cast_num, p):
    """Return True if below the influence of the surface

    The surface influence determined subjectively by looking at vertical
    profiles. This was primarily implemented for sake of dissipation calcs
    """
    region = {92:16, 93:10, 94:10, 95:5,96:9,97:10,98:7,99:16,
              100:0, 101:18, 102:26, 103:18, 104:20, 105:11, 106:23, 107:9,
              108:15, 109:16, 110:20, 111:18, 112:14, 113:23, 114:18, 115:28,
              116:23, 117:20, 118:11, 119:8, 120:11, 121:14, 122:15}
    below_inds = p > region[cast_num]
    return below_inds


def process_cast(cast_num, downcast=True):
    C, sigma, theta, p_r, S, T = read_cast(cast_num, downcast=downcast)
    T, theta = shift_T(T, theta)
    C_r, T_r, theta_r = C.copy(), T.copy(), theta.copy()
    C, T, theta, p = smooth_params(C, T, theta, p_r)
    S, sigma = recalc_S_and_sigma(C, T, p)
    C, sigma, theta, p, S, T = remove_pressure_reversals(
        p_r, C, sigma, theta, p, S, T)

    S_r, sigma_r = recalc_S_and_sigma(C_r, T_r, p_r)
    raw = dict(C=C_r, T=T_r, theta=theta_r, p=p_r, sigma=sigma_r)

    return C, S, theta, T, sigma, p, raw


def calc_eps_ctd(p, sigma, below_inds, plot_overturns=False):

    # Ignore near-surface values in overturn calcs
    sigma = sigma.copy()
    sigma[~below_inds] = np.sort(sigma[~below_inds])

    eps = calc_eps(p, sigma + 1000, p, plot_overturns=plot_overturns,
                   n_smooth_rho=1, already_downcast=True, min_drho=5e-4)[0]

    # Assume p in dbar = z in m
    eps_zavg = (np.gradient(p)*eps).sum()/np.ptp(p[below_inds])
    return eps, eps_zavg


def print_depth_averaged_diss():
    casts = np.r_[92:117]
    eps_all_zavg = np.zeros_like(casts).astype(float)
    for i, cast in enumerate(casts):
        C, S, theta, T, sigma, p, raw = process_cast(cast)
        below = below_surface_influence(cast, p)
        eps, eps_zavg = calc_eps_ctd(p, sigma, below)
        eps_all_zavg[i] = eps_zavg
        print('Cast: {0},  eps: {1:1.4g}'.format(cast, eps_zavg))
    return eps_all_zavg


# if __name__ == '__main__':
    # eps_all_zavg = print_depth_averaged_diss()

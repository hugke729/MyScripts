from seawater.eos80 import dens0
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv


def add_sigma_contours(ax, levels=np.arange(20, 30, 0.5), n=100, color='k',
                       fmt=r'%r', label_levels=None, contour_kw=dict()):
    """Add sigma contours to an existing T/S diagram"""
    Slim = ax.get_xlim()
    Tlim = ax.get_ylim()
    S, T = np.meshgrid(np.linspace(Slim[0], Slim[1], n),
                       np.linspace(Tlim[0], Tlim[1], n))
    sigma = dens0(S, T) - 1000
    cax = ax.contour(S, T, sigma, levels, colors=color, zorder=-5,
                     **contour_kw)
    label_levels = levels[::2] if label_levels is None else label_levels
    plt.clabel(cax, label_levels, inline=True, fmt=fmt, zorder=5)
    plt.draw()


def intermediate_density_profile(rho, min_drho=1E-3, return_up_down=False):
    """Intermediate density profile following Gargett and Garner 2008

    doi:10.1175/2008JTECHO541.1"""

    rho_down = np.zeros_like(rho)
    rho_up = np.zeros_like(rho)

    rho_down[0] = rho[0]
    rho_up[0] = rho[-1]

    for i, (rho_i, rho_ip1) in enumerate(zip(rho, rho[1:])):
        if abs(rho_ip1 - rho_down[i]) > min_drho:
            rho_down[i+1] = rho_i
        else:
            rho_down[i+1] = rho_down[i]

    # Same but from bottom up
    rho_inv = np.flipud(rho)
    for i, (rho_i, rho_ip1) in enumerate(zip(rho_inv, rho_inv[1:])):
        if abs(rho_ip1 - rho_up[i]) > min_drho:
            rho_up[i+1] = rho_ip1
        else:
            rho_up[i+1] = rho_up[i]

    # Flip rho_up now that it has been calculated
    rho_up = np.flipud(rho_up)

    rho_intermediate = np.mean(np.c_[rho_up, rho_down], axis=1)

    if return_up_down:
        # Odd problem with views into array means I need to copy output
        return rho_intermediate, rho_up.copy(), rho_down.copy()
    else:
        return rho_intermediate


def read_tide_csv(tide_file):
    """Read tide height csv file output from tbone.biol.sc.edu/tide

    See examples in /home/hugke729/PhD/Data/Shipboard/doc/

    Input
    -----
    tide_file: str
        Name of csv file

    Returns
    -------
    dates: list
        List of datetime.datetime objects
    table: pandas Dataframe
        Composed of date and tidal height
    """
    table = read_csv(
        tide_file, sep=',', skiprows=5, nrows=630,
        engine='python', infer_datetime_format=True,
        header=None, names=('Date', 'Height'), parse_dates=['Date'])

    # Unable to figure out how to do this correctly
    dates = [d.to_pydatetime() for d in table.Date]

    return dates, table

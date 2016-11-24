from seawater.eos80 import dens0
import numpy as np
import matplotlib.pyplot as plt


def add_sigma_contours(ax, levels=np.arange(20, 30, 0.5), n=100, color='k',
                       fmt=r'%r', label_levels=None):
    """Add sigma contours to an existing T/S diagram"""
    Slim = ax.get_xlim()
    Tlim = ax.get_ylim()
    S, T = np.meshgrid(np.linspace(Slim[0], Slim[1], n),
                       np.linspace(Tlim[0], Tlim[1], n))
    sigma = dens0(S, T) - 1000
    cax = ax.contour(S, T, sigma, levels, colors=color, zorder=-5)
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
            rho_down[i+1] = rho_ip1
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
        return rho_intermediate, rho_up, rho_down
    else:
        return rho_intermediate

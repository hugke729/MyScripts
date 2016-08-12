from seawater.eos80 import dens0
import numpy as np
import matplotlib.pyplot as plt


def add_sigma_contours(ax, levels=np.arange(20, 30, 0.5), n=100):
    """Add sigma contours to an existing T/S diagram"""
    Slim = ax.get_xlim()
    Tlim = ax.get_ylim()
    S, T = np.meshgrid(np.linspace(Slim[0], Slim[1], n),
                       np.linspace(Tlim[0], Tlim[1], n))
    sigma = dens0(S, T) - 1000
    cax = ax.contour(S, T, sigma, levels, colors='k')
    plt.clabel(cax, levels[::2], inline=True, fmt=r'%r', zorder=5)
    plt.draw()

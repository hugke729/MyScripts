from scipy.integrate import simps
from scipy.optimize import fsolve
from scipy.special import eval_legendre
from seawater.eos80 import dens0, salt
from seawater.constants import c3515
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from MyNumpyTools import rms
from MyGrids import estimate_cell_edges
from MyFunctions import calc_power_law


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


def read_tide_csv(tide_file, nrows=630):
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
        tide_file, sep=',', skiprows=5, nrows=nrows,
        engine='python', infer_datetime_format=True,
        header=None, names=('Date', 'Height'), parse_dates=['Date'])

    # Unable to figure out how to do this correctly
    dates = [d.to_pydatetime() for d in table.Date]

    return dates, table


def C_from_S_and_T(S, T, p=0):
    """Find conductivity based on salinity and temperature

    This inverts the standard direction of the equations where S(C, T, p)
    """

    def func(R):
        return salt(R, T, p) - S

    R0 = 26.6/c3515  # 26.6 is typical conductivity for CAA
    R_out = fsolve(func, R0)[0]
    C_out = R_out*c3515

    return C_out


def gk96_xi(S, T, print_both_xi=False):
    """Calculate the xi parameter defined in Galbraith and Kelley

    Their method is defined on page 694

    Inputs
    ------
    S, T: array-like
        Values of salinity and temperature within the overturn
    print_both_xi: bool
        If print, give both xi_T and xi_S

    This has been checked by digitizing to my best ability Figure 12 of
    GK96 (both top panels) and ensuring I get comparable result
    """
    rho = dens0(S, T)

    b_S, a_S = np.polyfit(S, rho, 1)
    b_T, a_T = np.polyfit(T, rho, 1)

    rho_S = a_S + b_S*S
    rho_T = a_T + b_T*T

    N_times_rms_thorpe_sq = np.sum((rho - np.sort(rho))**2)
    rms_thorpe = ((1/S.size)*N_times_rms_thorpe_sq)**0.5
    xi_S = rms(rho_S - rho)/rms_thorpe
    xi_T = rms(rho_T - rho)/rms_thorpe

    if print_both_xi:
        print('ξ_S: {0:2.2g}'.format(xi_S))
        print('ξ_T: {0:2.2g}'.format(xi_T))

    return np.max(np.r_[xi_S, xi_T])


def calc_mode_coeffs(signal, z, sines=True, nmax=5, normed=True):
    """Assuming constant stratification, determine contribution of modes.

    Inputs
    ------
    signal: array-like
        Anomaly about zero: rho', T', u' etc
    z: array-like
        Depth vector same shape as signal
    sines: bool
        If true, basis functions are sines (e.g., rho').
        If False, then cosines (e.g., u')
    nmax: int
        Total number of modes to use
    normed: bool
        If true, return coefficients whose absolute values sum to 1

    Returns
    -------
    coeffs: array-like
        nmax + 1 values of coefficients where first value is 0th mode
    recreated_signal: array-like
        The input signal created from nmax basis functions
    """
    signal, z = [np.asarray(arr) for arr in [signal, z]]
    zf = estimate_cell_edges(z)

    fn = np.sin if sines else np.cos

    # Remove depth-mean from signal
    # signal -= signal.mean()

    recreated_signal = np.zeros_like(signal)
    coeffs = np.zeros(nmax)
    for n in np.r_[1:nmax + 1]:
        vec_n = fn(n*np.pi*z/np.ptp(zf))
        coeffs[n-1] = simps(vec_n*signal)/np.ptp(zf)*2

        recreated_signal += coeffs[n-1]*vec_n

    if normed:
        coeffs /= np.abs(coeffs).sum()

    return coeffs, recreated_signal


def tidal_diss_supercritical_topo(Uvec, b, h, N):
    """
    Evaluated the dissipation predicted by Klymak et al. 2010 for constant N

    Inputs
    ------
    Uvec: array or list
        Deep-water tidal velocities
    b: float
        Obstacle height (m)
    h: float
        Depth in deep water (m)
    N: float
        Constant buoyancy frequency in rad/s

    Returns
    -------
    D: array
        Dissipation (W/m) for each U in Uvec
    """
    def eval_nc(U, N, b, h):
        """Critical mode number. Eq 6 of Klymak et al. (2010)"""
        d = h-b
        nc = N*d/(np.pi*U)
        nc = np.ceil(nc).astype(int)
        return nc

    def F_LY03(U, N, b, h):
        """Flux in W/m given in abstract of Llewelyn Smith and Young (2003)"""
        rho0 = 1000
        M = eval_M(b, h)
        F = (np.pi/4)*rho0*b**2*U**2*N*M
        return F

    def eval_M(b, h):
        """Interpolate the curve in Figure 4 of Llewelyn Smith and Young (2003)

        This is a quick and dirty alternative to doing it properly"""
        assert b/h <= 0.9, 'b/h exceeds LY03 Figure 4. Cannot interpolate'
        xgraph = np.sort(np.r_[0:0.91:0.1, 0.85])
        ygraph = 1 + 0.01*np.r_[0, 1, 3, 5, 8, 13, 20, 32, 49, 64, 87]
        M = np.interp(b/h, xgraph, ygraph)
        return M

    def eval_P_k(k, b, h):
        B = np.pi*b/h
        cosB = np.cos(B)
        P_k = eval_legendre(k-1, cosB) - eval_legendre(k, cosB)
        return P_k

    def flux_per_mode(F, b, h):
        ninf = 20  # Approximation of infinity
        kvec = np.r_[1:ninf]
        P_vec = np.r_[[eval_P_k(k, b, h) for k in kvec]]
        denom = np.sum([(1/k)*P_vec[k-1]**2 for k in kvec])
        F_n = np.r_[[(F/n)*P_vec[n-1]**2/denom for n in kvec]]
        # Add unused value at beginning to allow correct nc indexing
        F_n = np.insert(F_n, 0, np.nan)
        return F_n

    Uvec = np.atleast_1d(Uvec)
    D = np.zeros_like(Uvec)
    for i, U in enumerate(Uvec):
        nc = eval_nc(U, N, b, h)
        F = F_LY03(U, N, b, h)
        F_n = flux_per_mode(F, b, h)
        D[i] = np.sum(F_n[nc:])
    return D


if __name__ == '__main__':
    U = np.r_[1, 2, 4, 8, 12, 16, 24]/100
    D = tidal_dissipation_supercritical_topography(U, b=200, h=250, N=6.2E-3)
    calc_power_law(U, D, True)

# These plotting functions are intended for use with IPython notebook,
# specifically ~/PhD/Python/MVP/Amundsen_Sep_2015_CAA_data.ipynb
import numpy as np
import numpy.ma as ma
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import warnings
import pickle
from collections import OrderedDict
from matplotlib.patches import Polygon
from warnings import filterwarnings
from MyPlotFunctions import flipx, flipy, padded_lims, fix_colorbar, pos
from MyOceanography import add_sigma_contours
from MyNumpyTools import minmax
from MyColormaps import (cmap_cold, get_n_colors, get_n_colors2,
                         red_yellow_grey_cyan_blue)
import colormaps
from MySysUtils import merge_dicts, preall
from MyInteractive import disp_latlon
from IPython.display import Markdown, display
from MyMVP import loadMVP_m1, combine_MVP_ADCP
from MyCTD import get_xyt, get_theta_S_sigma_z, get_velocity_profile, calc_N2
from MyInterp import inpaint_nans
from MyMapFunctions import haversines
from MyInterp import smooth1d_with_holes as smooth


# Versatile contour plot
def contourZ(ax, X, Y, Z, sigma, depth, clim, label,
             cmap=cmap_cold(reverse=True), fine_rho=False, direc=1,
             extrapolate=False):
    """Create a contour plot of Z(X, Y) overlaid with density contours

    Inputs
    ------
    X: N x 1 vector of distance for each transect (typically km)
    Y: M x 1 vector of depths (i.e., depths used for binning data)
    Z: M x N masked array of quantity to contour

    sigma: M x N masked array of density - 1000 (kg/m^3)
    depth: N x 1 vector of depths
    clim: 2-tuple of colour limits for contour plot
    label: label for the colour axis
    cmap: which colourmap to use (defaults to cold (dark blue to grey))
    fine_rho: add 0.1 kg/m3 contours if True
    direc: flip x axis if direc is -1
    extrapolate: whether to extrapolate missing density and 'Z' values
    """

    # Fill any masked values with NaNs
    Z[Z.mask] = np.nan
    sigma[sigma.mask] = np.nan

    if extrapolate:
        leave_mask = Y[:, np.newaxis] > np.array(depth)[np.newaxis, :]
        Z = inpaint_nans(Z, leave_mask=leave_mask)
        sigma = inpaint_nans(sigma, leave_mask=leave_mask)

    # Mask NaNs now that interpolation is completed
    X, Y, Z, sigma = map(ma.masked_invalid, [X, Y, Z, sigma])

    # Contour plot options
    options = {'levels': np.linspace(clim[0], clim[1], 10), 'cmap': cmap}

    # Plot using pcolor if doing dissipation
    if 'dissipation' in label.lower():
        cax = ax.pcolormesh(X, Y, Z, cmap='jet', vmin=-8, vmax=-5)
        print(X.shape)
    else:
        # Allow Z to input the wrong way around but still work
        try:
            cax = ax.contourf(X, Y, Z, **options)
        except TypeError:
            cax = ax.contourf(X, Y, Z.T, **options)

    # Add colorbar
    cb = plt.colorbar(cax, ax=ax, format='%2.2f', fraction=0.08,
                      orientation='vertical', shrink=1,
                      ticks=np.linspace(clim[0], clim[1], 3))
    cb.set_label(label)

    # Add 0.1 kg/m^3 density contours if desired
    if fine_rho:
        options = {'colors': 'yellow', 'linewidth': 0.3, 'alpha': 0.5}
        contour_levels = np.arange(22, 30, 0.1)

        # Allow sigma to input the wrong way around but still work
        try:
            ax.contour(X, Y, sigma, contour_levels, **options)
        except TypeError:
            ax.contour(X, Y, sigma.T, contour_levels, **options)

    # Add 0.25 kg/m^3 density contours
    main_levels = np.arange(22, 30.01, 0.25)
    co = ax.contour(X, Y, sigma, main_levels, colors='k')
    plt.clabel(co, main_levels[::2], colors='k', fmt='%1.1f', fontsize=7,
               use_clabeltext=False)

    # Add seafloor depth
    ax.plot(X, depth, 'k', linewidth=2)

    # Add markers to show where casts
    ax.plot(X, 2 + np.zeros_like(X), 'k+')

    # Adjust limits
    curr_ylims = ax.get_ylim()
    ax.set(ylabel='Depth', xlabel='Distance (km)',
           ylim=sorted(curr_ylims, reverse=False))

    # Flip x axis if required
    if direc == -1:
        flipx(ax)
    return cax, cb


# Pcolor plot for ADCP data
def adcp_pcolor(data, is_long_section=False, ax=None, cross_vel='V',
                add_quiver=False, extrapolate=False, **pcolor_kwargs):
    """Plot smoothed ADCP data

    Inputs
    ------
    data: dict containing smoothed (denoted _s) ADCP data
    is_long_section: True if along channel and False if cross channel
    ax: axis in which to add the plot
    cross_vel: component to plot for cross sound transects ('U' or 'V')
    pcolor_kwargs: dict containing other options for pcolor plot
    add_quiver: bool
    extrapolate: bool
    """
    # Get current axis unless axis is given
    ax = plt.gca() if ax is None else ax

    # Velocity to plot:
    #     along_vel for long-sound transects
    #     V for cross-sound transects since cross-sections all run east-west
    #     unless specified that we want to plot U
    if is_long_section:
        vel = -data['along_vel_s'].T
        label = 'Along-transect velocity\n(toward NW)'
    elif (not is_long_section) and cross_vel.upper() == 'V':
        vel = data['vel_s'][..., 1].T
        label = 'Northward velocity'
    elif (not is_long_section) and cross_vel.upper() == 'U':
        vel = data['vel_s'][..., 0].T
        label = 'Eastward velocity'

    if extrapolate:
        vel[vel.mask] = np.nan
        # For some reason depths are 12.37, 16.37, 20.37, ...
        dep = np.insert(data['dep'], 0, (np.r_[0, 4, 8] + 0.37))
        Nx = len(data['depth_s'])
        vel = np.insert(vel, np.r_[0, 1, 2], np.nan, axis=0)
        leave_mask = dep[:, np.newaxis] > data['depth_s'][np.newaxis, :]
        vel = inpaint_nans(vel, leave_mask=leave_mask)
        vel = ma.masked_invalid(vel)
    else:
        dep = data['dep']

    cmap = red_yellow_grey_cyan_blue()
    pcol_opts = dict(rasterized=True, cmap=cmap, vmin=-0.5, vmax=0.5)
    pcol_opts = merge_dicts(pcol_opts, pcolor_kwargs)
    print(pcol_opts)
    pcm = ax.pcolormesh(data['dist_s'] / 1e3, dep, vel, **pcol_opts)
    cbar = plt.colorbar(pcm, ax=ax, format='%2.2f', orientation='vertical',
                        fraction=0.08, shrink=1)
    fix_colorbar(cbar)
    # Draw seafloor
    ax.plot(data['dist_s'] / 1e3, data['depth_s'], color='k')

    if add_quiver:
        filterwarnings('ignore', '.*Mean of empty slice*.')
        filterwarnings('ignore', '.*converting a masked element*.')
        x_step, y_step = 5, 5
        Nx, Ny = len(data['dist_s'])//x_step, len(data['dep'])//y_step

        x, y, u, v = preall((Nx - 1, Ny - 1), 4)
        U_all, V_all = data['vel_s'][..., :2].T
        i_ranges = np.linspace(0, len(data['dist_s']) - 1, Nx).astype(int)
        j_ranges = np.linspace(0, len(data['dep']) - 1, Ny).astype(int)
        for i, j in np.ndindex(Nx - 1, Ny - 1):
            i_inds = np.r_[i_ranges[i]:i_ranges[i+1]]
            j_inds = np.r_[j_ranges[j]:j_ranges[j+1]]
            J_inds, I_inds = np.meshgrid(j_inds, i_inds)
            x[i, j] = np.nanmean(data['dist_s'][i_inds])/1e3
            y[i, j] = np.nanmean(data['dep'][j_inds])
            u[i, j] = np.nanmean(U_all[J_inds, I_inds])
            v[i, j] = np.nanmean(V_all[J_inds, I_inds])

        # Reduce any particularly large velocities
        std_dev_max = 1.2
        u[np.isnan(u)], v[np.isnan(v)] = 0, 0
        u_inds = np.abs(u - np.nanmean(u)) > std_dev_max*np.std(u)
        v_inds = np.abs(v - np.nanmean(v)) > std_dev_max*np.std(v)
        u[u_inds] = np.sign(u[u_inds] - np.nanmean(u))*std_dev_max*np.std(u)
        v[v_inds] = np.sign(v[v_inds] - np.nanmean(v))*std_dev_max*np.std(v)

        ax.quiver(x, y, u, v)

    # Put sea surface at top
    flipy(ax)

    # Label plots
    clabel = label + ' (m/s)'
    ax.set(xlabel='Distance (km)', ylabel='Depth (m)')
    cbar.set_label(clabel)
    plt.draw()
    return pcm


def TS_cmap(direc):
    plt.register_cmap(name='inferno', cmap=colormaps.inferno)
    plt.register_cmap(name='inferno_r', cmap=colormaps.inferno)
    plt.register_cmap(name='viridis', cmap=colormaps.viridis_r)
    plt.register_cmap(name='viridis_r', cmap=colormaps.viridis_r)
    cmap = 'inferno'
    if direc == -1:
        cmap = 'inferno_r'
    return cmap


def add_water_masses(ax, water_mass_list=[], show_markers=True, alpha=0.5):
    """Add patches and scatter points showing different water masses

    Water masses derived from Kliem and Greenberg (2003)

    If water_mass_list is a non-empty list, only plot those on the list.
    Possible water masses are:
        'baffin_bay', 'northwest_CAA', 'canada_basin', 'n_viscount_melville',
        'n_lancaster_sound', 'nares_st'
    """
    defn_file = ('/home/hugke729/PhD/Data/TS_climatology/Kliem_Greenberg/' +
                 'derived/water_mass_definitions.p')
    D = pickle.load(open(defn_file, 'rb'))

    # Create seperate dicts with median values and patches
    D_medians = OrderedDict()
    D_polys = OrderedDict()
    for k in D:
        if k.endswith('_medians'):
            new_key = k.replace('_medians', '')
            D_medians[new_key] = D[k]
        elif k.endswith('_poly'):
            new_key = k.replace('_poly', '')
            D_polys[new_key] = D[k]

    # Add each water mass as a patch, which are already defined as patches
    # in the loaded dict
    names, patches = [], []
    cols = get_n_colors(8)
    for i, (k, v) in enumerate(D_polys.items()):
        if (water_mass_list != [] and k not in water_mass_list):
            # Don't plot this water mass
            continue
        names += [k]
        poly = Polygon(v, color=cols[i], zorder=1, alpha=alpha)
        patches += [ax.add_patch(poly)]

        # D_medians has same keys as D_polys
        xs, ys = D_medians[k].T
        xs, ys = [ma.filled(x, np.nan) for x in [xs, ys]]
        markers = 2*['o', 'v', 'D', 's', 'p', '*', '+', '<', 'x', 'v']
        for x, y, marker in zip(xs, ys, markers):
            if show_markers:
                ax.scatter(x, y, c=cols[i], s=45, marker=marker, zorder=5)
        else:
            # Work around to get autoscaling to work
            # Easiest to just plot line of zero width
            ax.plot(x, y, lw=0)

    # Make names nicer to read
    names = [name.replace('_', ' ').title() for name in names]
    names = [name.replace('Caa', 'CAA') for name in names]

    # Add legend below plot
    L, B, W, H = pos(ax)
    ax_leg = ax.get_figure().add_axes((L, 0.01, W, 0.06))
    ax_leg.legend(patches, names, ncol=2, frameon=False)
    ax_leg.axis('off')

    plt.draw()


def TS_diagram(ax, S, T, dist2d, direc=1, vmin=None, vmax=None,
               Slim=None, Tlim=None):
    """Temperature-salinity diagram coloured by distance along transect

    Inputs
    ------
    ax: axis on which to add the plot
    S, T, dist2d: Nx x Nz arrays
    direc: flip x axis if direc is -1
    vmin, vmax: limits for colour axis
    Slim, Tlim: limits for S and T (2-tuples)
    """

    # Colour limits are based current distance data, unless specified
    vmin = dist2d.min() if vmin is None else vmin
    vmax = dist2d.max() if vmax is None else vmax

    # T and S limits are based on current data (with some padding), unless
    # they are otherwise specified
    Slim = padded_lims(S) if Slim is None else Slim
    Tlim = padded_lims(T) if Tlim is None else Tlim

    # Indices with distances between vmin and vmax
    inds = np.logical_and(dist2d > vmin, dist2d < vmax)

    # T and S scatter with colour based on distance
    cax = ax.scatter(S[inds], T[inds], c=dist2d[inds], s=5,
                     cmap=TS_cmap(direc), vmin=vmin, vmax=vmax)

    # Specify labels, ticks, and limits
    ax.set(xlabel='Salinity', ylabel='Pot. Temperature (°C)',
           xticks=np.arange(int(Slim[0]), int(Slim[1]) + 1),
           xlim=Slim, ylim=Tlim)

    # Overlay density contours
    add_sigma_contours(ax)

    # Underlay water mass definitions
    add_water_masses(ax)

    # Add colorbar for distance along transect
    cbar = plt.colorbar(
        cax, ax=ax, orientation='vertical', fraction=0.08,
        ticks=np.linspace(*minmax(dist2d, astype='int'), num=2))
    cbar.set_label('Distance (km)')
    cbar.set_ticks((int(vmin) + 1, int(vmax) - 1))

    fix_colorbar(cbar)  # Only necessary if exporting figure

    return cbar, cax


def map_plot(ax, lon, lat, direc=1, dist=None, vmin=None, vmax=None):
    """Map showing transect

    Inputs should be self-explanatory
    """

    # Using pickled basemap is much quicker than starting from scratch
    m = pickle.load(open('/home/hugke729/PhD/Maps/penny_strait.pickle', 'rb'))
    m.fillcontinents(ax=ax)

    # vmin and vmax should equal those used in the T-S diagram
    vmin = dist.min() if vmin is None else vmin
    vmax = dist.max() if vmax is None else vmax

    # Indices with distances between vmin and vmax
    inds = np.logical_and(dist > vmin, dist < vmax)

    # Ensure coordinates are numpy arrays
    lon, lat = map(np.array, [lon, lat])

    # Scatter points outside the contour plot but on the transect
    cax_grey = m.scatter(lon[~inds], lat[~inds], c='grey', ax=ax, latlon=True,
                         cmap=TS_cmap(direc))

    # Scatter points coloured by distance along transect
    cax_col = m.scatter(lon[inds], lat[inds], c=dist[inds], ax=ax,
                        latlon=True, cmap=TS_cmap(direc))

    # Add the distance in km at each end of the transect
    ax.annotate(s=dist[0].astype(int), xy=m(lon[0], lat[0]))
    ax.annotate(s=dist[-1].astype(int), xy=m(lon[-1], lat[-1]))

    # For interactive use (may or may not work with notebook)
    disp_latlon(ax, m)

    return cax_col, cax_grey


def create_subplot(figsize=None):
    figsize = (15, 10) if figsize is None else figsize
    fig, axs = plt.subplots(
        nrows=2, ncols=2, gridspec_kw={'width_ratios': (3, 2)},
        **{'figsize': figsize})
    return fig, axs[0][0], axs[0][1], axs[1][0], axs[1][1]


def remove_bad_casts(grid_dict, nmin=10):
    """Remove bad casts from all components of dictionary

    nmin is Minimum number of bins needed to be considered a good cast"""
    good_casts = grid_dict['T'].count(axis=1) > nmin
    for key, value in grid_dict.items():
        if hasattr(value, 'ndim'):
            if key in ['z_c', 'z_f']:
                continue
            elif key in ['dist2d_f', 'z_f2d']:
                grid_dict[key] = value[np.append(good_casts, True), :]
            else:
                grid_dict[key] = value[good_casts, ...]
        else:
            # Lists are a little bit tricky compared to Numpy arrays
            grid_dict[key] = [x for i, x in enumerate(value) if good_casts[i]]

    return grid_dict


def plot_profiles(cast_nos, ctd_or_mvp='mvp', raw=False, figsize=None,
                  invert_color_order=False, TS_axes=None, vel_type='mag',
                  rho_or_N='N', show_water_masses=False):
    """
    Plot profiles of temperature, salinity, and density for specified casts

    Includes a map, to identify sites plotted

    Inputs
    ------

    cast_nos : list, array or, int
        casts to plot (each gets its own colour)
    ctd_or_mvp : str, or list of strings same length as cast_nos
        whether cast_nos refers to ctd or mvp profiles
    raw : bool
        if raw is True, plot raw data, not binned data (for MVP only)
    figsize : 2-tuple
        figsize in inches
    invert_color_order : bool
        If true, use blue to red, rather than red to blue
    TS_axes : None or tuple
        (S_min, S_max), (T_min, T_max)
    vel_type : str
        The component of velocity to plot. One of 'u', 'v', or 'mag'
    rho_or_N : str
        Which quantity to plot in top right corner
        If rho, then plot sigma
        If N, then plot N2
    show_water_masses : bool
        Whether to add water masses from Kliem and Greenberg
    """

    # Specific z values to use
    z_bins = np.r_[:251]

    # Ensure cast_nos is iterable, even if only one item
    cast_nos = np.atleast_1d(cast_nos)

    # Set up subplots
    figsize = (15, 10) if figsize is None else figsize
    fig = plt.figure(figsize=figsize)

    # Top row of plots
    axT = plt.subplot2grid((2, 6), (0, 0), colspan=2)
    axS = plt.subplot2grid((2, 6), (0, 2), colspan=2, sharey=axT)
    ax_N2 = plt.subplot2grid((2, 6), (0, 4), colspan=2, sharey=axT)
    [plt.setp(ax.get_yticklabels(), visible=False) for ax in [axS, ax_N2]]

    # Bottom row of plots
    ax_TS = plt.subplot2grid((2, 6), (1, 0), colspan=2)
    ax_uv = plt.subplot2grid((2, 6), (1, 2), colspan=2, sharey=axT)
    ax_map = plt.subplot2grid((2, 6), (1, 4), colspan=2)

    # We aren't worried about ax_map not working with tight_layout
    warnings.filterwarnings('ignore', '.*not compatible with tight_layout*.')

    # Invert axis to put sea surface at top
    # axS and ax_N2 change as well since they're linked
    axT.invert_yaxis()

    # Read in pickled basemap object, and plot the land
    m = pickle.load(open('/home/hugke729/PhD/Maps/penny_strait.pickle', 'rb'))
    m.fillcontinents(ax=ax_map)

    # Define colours to use
    colors = get_n_colors2(len(cast_nos))

    # Invert color order if desired
    colors = colors[::-1] if invert_color_order else colors

    # For each chosen, read in data from appropriate place, then plot profiles
    for i, cast in enumerate(cast_nos):

        # Decide what kind of cast we're working with
        if type(ctd_or_mvp) is str:
            cast_type = ctd_or_mvp.lower()
        elif type(ctd_or_mvp) is list:
            assert len(ctd_or_mvp) == len(cast_nos), 'Inputs not same length'
            cast_type = ctd_or_mvp[i].lower()

        if cast_type == 'mvp':
            xyt, data, binned = loadMVP_m1(cast, z_bins, mask_nans=True)
            # Rename binned to data, unless we're plotting raw data
            # Note that data and binned contain the same keys
            data = data if raw else binned

            z = data['z']
            theta, S, prho = data['theta'], data['S'], data['prho']
            sigma = prho - 1000
            N2 = data['N2']
            lon, lat = xyt['lon'], xyt['lat']

        elif cast_type == 'ctd':
            theta, S, sigma, z = get_theta_S_sigma_z(cast)
            N2 = calc_N2(sigma, z)
            lat, lon = get_xyt(cast)[:2]
            u, v, z_vel = get_velocity_profile(cast)

        # For current cast, do all plots in same colour and add scatter to
        line_props = {'color': colors[i], 'lw': 1}
        axT.plot(theta, z, **line_props)
        axS.plot(S, z, **line_props)
        if rho_or_N == 'N':
            ax_N2.plot(1e4*smooth(N2, 20), z, **line_props)
        elif rho_or_N == 'rho':
            ax_N2.plot(sigma, z, **line_props)
        ax_TS.plot(S, theta, color=colors[i], lw=2)

        # Type of velocity plot depends on keyword arg
        if cast_type == 'ctd':
            vels = dict(u=u, v=v, mag=np.hypot(u, v))
            ax_uv.plot(100*vels[vel_type], z_vel, color=colors[i])

        # map axis to show position of profile
        m.scatter(lon, lat, s=40, color=colors[i], latlon=True)

        # If correct backend, display lat lon and distances on map's cursor
        if matplotlib.get_backend().lower() in ['qt4agg', 'nbagg']:
            disp_latlon(ax_map, m)

    TS_axes = ((30.5, 33.5), (-1.5, 1.5)) if TS_axes is None else TS_axes
    ax_TS.set(xlim=TS_axes[0], ylim=TS_axes[1])

    add_sigma_contours(ax_TS, levels=np.r_[24:28:0.25])
    if show_water_masses:
        add_water_masses(ax_TS)
    # Add labels to plots
    axT.set(xlabel='Pot. temperature (°C)')
    axS.set(xlabel='Salinity')

    if rho_or_N == 'N':
        ax_N2.set(xlabel='N2 (×10^−4)')
    elif rho_or_N == 'rho':
        ax_N2.set(xlabel='Sigma (kg/m$^3$)')
    ax_uv.set(xlabel='Velocity ' + vel_type + ' (cm/s)')

    # Finish off by using space as best we can
    fig.tight_layout()
    plt.show()


def plot_section(name, limits, mvp_quantity='theta', figsize=None,
                 cross_vel='V', add_quiver=False, extrapolate=False):
    """Put together T contour, velocity, T-S diagram and map plots

    mvp_quantity can be 'theta' or 'eps'
    cross_vel can be 'U' or 'V' (takes effect only for cross-sound transects)
    if add_quiver is True, arrows are plotted to show velocity bearing
    """
    warnings.filterwarnings('ignore', 'elementwise comparison')
    plt.ioff()

    # Get limits for current transect
    clim, xlim, ylim, direc_str, direc = limits[name]
    vmin, vmax = sorted(xlim)

    # Set up the subplot
    fig, ax_mvp, ax_map, ax_adcp, ax_ts = create_subplot(figsize)

    if ('long' in name) or ('repeat' in name):
        is_long_section = True
        # S and T limits for all long-sound transects
        # Slim, Tlim = (31.5, 33.2), (-1.5, 0.2)
        Slim, Tlim = (30.5, 33.5), (-1.5, 2.0)
    else:
        is_long_section = False
        Slim, Tlim = (30.5, 33.5), (-1.5, 2.0)

    # Read in processed data saved in pickle
    transect_loc = '/home/hugke729/PhD/Data/Shipboard/MVP/transects/'
    transect_name = (transect_loc + name + '.p')
    D = pickle.load(open(transect_name, 'rb'))

    # Remove bad casts for plotting
    D = remove_bad_casts(D)

    # Unwrap commonly used parts of the dict D
    X, Z, theta = D.get('dist_flat', D['dist']), D['z_c'], D['theta'].T
    eps = D['eps'].T
    sigma = D['prho'].T - 1000

    # Decide what to plot in top left corner
    if mvp_quantity == 'theta':
        quantity = theta
        ylabel = 'Pot. Temperature (°C)'
    elif mvp_quantity == 'eps':
        quantity = np.log10(ma.masked_equal(eps, 0))
        clim = -8, -5
        ylabel = 'Dissipation rate (W/kg)'

    # Plot temperature contour
    cax, cbar = contourZ(
        ax_mvp, X, Z, quantity, sigma, D['bottom'], clim, ylabel,
        fine_rho=False, direc=direc, extrapolate=extrapolate)
    ax_mvp.set(xlim=xlim, ylim=ylim)

    # T-S plot
    cbar_ts, cax_ts = TS_diagram(
        ax_ts, D['S'], D['theta'], D['dist2d_c'], direc=direc,
        vmin=vmin, vmax=vmax, Slim=Slim, Tlim=Tlim)

    # Map
    cax_col, cax_grey = map_plot(
        ax_map, D['lon'], D['lat'], direc=direc,
        dist=D.get('dist_flat', D['dist']), vmin=vmin, vmax=vmax)

    # ADCP plot
    data = pickle.load(open('/home/hugke729/PhD/Data/Shipboard/ADCP/processed/' +
                            name + '.p', 'rb'))
    adcp_pcolor(data, is_long_section=is_long_section, ax=ax_adcp,
                cross_vel=cross_vel, add_quiver=add_quiver,
                extrapolate=extrapolate)
    ax_adcp.set(ylim=ylim, xlim=xlim)
    ax_adcp.contour(X, Z, sigma, np.arange(22, 30, 0.25), linewidths=0.75,
                    colors='grey')

    # Adjust plot without worrying about legend axis
    warnings.filterwarnings('ignore', '.*figure includes Axes that are not*.')
    plt.tight_layout()

    # Adjust TS diagram to make space for legend
    pos1 = ax_ts.get_position()  # get the original position
    nudge_up = 0.05
    pos2 = [pos1.x0, pos1.y0 + nudge_up, pos1.width, pos1.height - nudge_up]
    ax_ts.set_position(pos2)

    # For some reason, I get two figures by default. Hence, this close command
    # actually gives me the desired result (one figure)
    plt.close(fig)
    title = name.replace('_', ' ')
    display(Markdown('## ' + title.capitalize()))
    return fig


def plot_ctd_as_section(cast_nos):
    data_dir = '/home/hugke729/PhD/Data/Shipboard/CTD/processed/'
    xyt_file = data_dir + 'xyt_summary.pickle'
    xyt = pickle.load(open(xyt_file, 'rb'))

    # Preallocate
    lat, lon, depth = preall(len(cast_nos), copies=3)
    S, theta, sigma = preall((len(cast_nos), 300), 3, np.nan)
    z = np.r_[2:302]

    # Ensure west to east
    if xyt[cast_nos[0]][1] > xyt[cast_nos[-1]][1]:
        cast_nos = cast_nos[::-1]

    for i, cast in enumerate(cast_nos):
        lat[i] = xyt[cast][0]
        lon[i] = xyt[cast][1]
        depth[i] = xyt[cast][3]
        theta_i, S_i, sigma_i, _ = get_theta_S_sigma_z(cast)
        N = len(theta_i)

        theta[i, :N] = theta_i
        S[i, :N] = S_i
        sigma[i, :N] = sigma_i

    # Calculate dist
    dist = haversines(lon, lat)[0]

    fig, ax = plt.subplots()

    cax = ax.contourf(
        dist, z, theta.T,
        cmap=cmap_cold(reverse=True), levels=np.r_[-1.6:1.61:0.2])

    rho_levels = [25, 25.5, 26, 26.25, 26.5]
    co = ax.contour(dist, z, sigma.T, rho_levels, colors='r')
    ax.clabel(co, rho_levels[1:], fmt='%3.2f')
    ax.plot(dist, depth, 'k')
    cbar = fig.colorbar(cax)
    cbar.set_label('Potential temperature (C)')
    ax.set(ylim=(250, 0), ylabel='Depth (m)', xlabel='Distance eastward (km)')

    ax.set(title='CTD casts: ' + str(cast_nos[0]) + '--' + str(cast_nos[-1]))

    return ax


def froude_plots():
    """Plot composite Froude number for all sections"""
    fig_opts = dict(nrows=6, ncols=1, sharex=True, sharey=True)
    fig_G, ax_G = plt.subplots(**fig_opts)
    fig_rho, ax_rho = plt.subplots(**fig_opts)
    i = 0

    d_vs_z_fname = '/home/hugke729/PhD/Data/Shipboard/MVP/transect_depth.txt'
    dist, depth = np.genfromtxt(d_vs_z_fname, unpack=True, delimiter=',')
    for j, k in enumerate(transects):
        if 'repeat' not in k and 'full' not in k:
            continue
        data_dir = '/home/hugke729/PhD/Data/Shipboard/'
        mvp_fname = data_dir + 'MVP/transects/' + k + '.p'
        adcp_fname = data_dir + 'ADCP/processed/' + k + '.p'
        mvp_dict = pickle.load(open(mvp_fname, 'rb'))
        adcp_dict = pickle.load(open(adcp_fname, 'rb'))
        G = combine_MVP_ADCP(mvp_dict, adcp_dict)
        ax_rho[i].plot(dist, depth, 'k')
        ax_rho[i].plot(mvp_dict['dist_flat'], mvp_dict['interface_depth'], 'k')
        cax = ax_rho[i].contourf(
            mvp_dict['dist_flat'], mvp_dict['z_c'], mvp_dict['prho'].T - 1000,
            cmap='afmhot_r', levels=np.r_[25.2:27:0.2])

        ax_rho[i].set_ylabel(k)
        flipy(ax_rho[i])
        ax_G[i].plot(mvp_dict['dist_flat'], G, 'k')
        ax_G[i].plot(mvp_dict['dist_flat'], np.ones_like(G), 'r')
        i += 1

    cbar = fig_rho.colorbar(cax, ax=list(ax_rho))
    cbar.ax.invert_yaxis()

    ax_G[0].set_ylim(0, 5)

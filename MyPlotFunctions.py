from matplotlib.lines import Line2D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from MyNumpyTools import minmax
from MyFunctions import flatten2Dlist


def rm_x_axis(ax=None):
    ax = plt.gca() if ax is None else ax
    ax.get_xaxis().set_ticklabels([])


def rm_y_axis(ax=None):
    ax = plt.gca() if ax is None else ax
    ax.get_yaxis().set_ticklabels([])


def dash_x_zero(ax=None):
    ax = plt.gca() if ax is None else ax
    x0, x1 = ax.get_xlim()
    l = Line2D([x0, x1], [0, 0], linestyle='--', zorder=-1, color='k')
    ax.add_line(l)


def flipx(ax=None):
    ax = plt.gca() if ax is None else ax
    curr_x_lim = ax.get_xlim()
    ax.set_xlim((max(curr_x_lim), min(curr_x_lim)))
    plt.draw()


def flipy(ax=None):
    ax = plt.gca() if ax is None else ax
    curr_y_lim = ax.get_ylim()
    ax.set_ylim((max(curr_y_lim), min(curr_y_lim)))
    plt.draw()


def flip(ax):
    flipy(ax)


def flipys():
    fig_list = plt.get_fignums()
    for i in fig_list:
        plt.figure(i)
        for ax in plt.gcf().get_axes():
            flipy(ax)


def rm_offset(ax=None):
    ax = plt.gca() if ax is None else ax
    y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(y_formatter)
    plt.draw()


def pos(ax=None):
    """Get the [L, B, W, H] of ax"""
    ax = plt.gca() if ax is None else ax
    pos_array = np.array(ax.get_position()).flatten()
    # Convert from [L, B, R, T] to [L, B, W, H]
    pos_array[[2, 3]] = pos_array[[2, 3]] - pos_array[[0, 1]]
    return pos_array


def replace_axis(fig, ax):
    """Delete an axis and add a new one

    A work around to not being able to unlink shared axes (AFAIK)
    """
    position = pos(ax)
    fig.delaxes(ax)
    new_ax = fig.add_axes(position)
    return new_ax


def label_power_of_2sf(x, pos):
    # http://stackoverflow.com/questions/25983218/
    # scientific-notation-colorbar-in-matplotlib
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    fmt = r'${} \times 10^{{{}}}$'.format(a, b)
    # fmt = ticker.FuncFormatter(fmt)
    return fmt


def note(string):
    """Add note to bottom left of figure"""
    a = plt.annotate(string, xy=(5, 5), xycoords='figure points')
    return a


def fix_colorbar(cbar):
    """Fix problems with colobar output to PDF (and probably other formats)"""
    # http://stackoverflow.com/questions/15003353/
    # why-does-my-colorbar-have-lines-in-it
    cbar.solids.set_rasterized(True)
    cbar.solids.set_edgecolor("face")


def zerolines(ax, x=False, y=False, linestyle='--', color=3*[0.5]):
    """Draw lines at x = 0 and/or y=0"""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if x:
        ax.plot((xlim[0], xlim[1]), (0, 0), linestyle=linestyle, color=color)
    if y:
        ax.plot((0, 0), (ylim[0], ylim[1]), linestyle=linestyle, color=color)
    # Ensure the lines don't alter the axes
    ax.set(xlim=xlim, ylim=ylim)


def padded_lims(X, pad=0.1):
    """limits based on input data x with padding fraction added to each side"""
    lims = minmax(X)
    lims = lims[0] - np.diff(lims)*pad, lims[1] + np.diff(lims)*pad
    return lims


def move_ax_vert(ax, dist):
    """Move ax vertically on figure by dist (fraction of figure)"""
    ax_pos = pos(ax)
    ax_pos[1] = ax_pos[1] + dist
    ax.set_position(ax_pos)
    plt.draw()


def pull_ax_vert(ax, dist):
    """Stretch ax vertically on figure by dist (fraction of figure)"""
    ax_pos = pos(ax)
    ax_pos[1] = ax_pos[1] + dist
    ax_pos[3] = ax_pos[3] - dist
    ax.set_position(ax_pos)
    plt.draw()


def subplot_cbar(fig, ax, cax):
    """Add a colorbar to right of panel of a suplot with rows

    Inputs
    ------
    fig: figure instance
    ax: axis on which to add subplot
    cax: the contour/pcolor/etc instance associated with colorbar"""
    r = 0.8
    b = 0.1
    fig.subplots_adjust(right=r, bottom=b)
    h = pos(ax)[3]
    l = 0.82
    w = 0.03
    cbar_ax = fig.add_axes([l, b, w, h])
    cbar = plt.colorbar(cax, cax=cbar_ax)
    return cbar


def create_scaled_subplot(xlims, ylims, bottom=0.1, left=0.1, figsize=None):
    """Create suplot grid that has axes with sizes that scale with limits"""
    # http://stackoverflow.com/questions/9879944/
    # adjust-one-subplots-height-in-absolute-way-not-relative-in-matplotlib
    xlims, ylims = map(np.array, [xlims, ylims])
    M, N = len(xlims), len(ylims)
    lpad, bpad = left, bottom
    rpad, tpad = 0.02, 0.02
    hpad, vpad = lpad + M*rpad, bpad + N*tpad

    rel_widths = xlims/sum(xlims)*(1 - hpad)
    l = np.insert(lpad + np.cumsum(rel_widths[:-1]), 0, lpad) + rpad*np.arange(M)

    rel_heights = (ylims/sum(ylims)*(1 - vpad))[::-1]
    b = (np.insert(bpad + np.cumsum(rel_heights[:-1]), 0, bpad) +
         tpad*np.arange(N))

    # Reverse b and rel_heights so that axs comes out in the expected order
    b, rel_heights = b[::-1], rel_heights[::-1]

    axs = []
    fig = plt.figure(figsize=figsize)
    for i, h in enumerate(rel_heights):
        axi = []
        for j, w in enumerate(rel_widths):
            ax = fig.add_axes([l[j], b[i], w, h])
            if i != N - 1:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])

            axi += [ax]
        axs += [axi]

    # Set correct limits
    for j, ax_row in enumerate(axs):
        for i, ax in enumerate(ax_row):
            ax.set_ylim(ylims[j], 0)
            ax.set_xlim(0, xlims[i])

    return fig, axs


def pcolor_variable_y(x, Y, C, ax=None, **kwargs):
    """Just like pcolor, but with the quadrilaterals forced to be rectangles

    This represents how an ocean model usually treats cells when hFacs are used

    The rectangles are defined by
    >>> (X[i  ], Y[i,   j])
    >>> (X[i  ], Y[i, j+1])
    >>> (X[i+1], Y[i+1, j])
    >>> (X[i+1], Y[i+1, j])
    Note the repetition of third and fourth points

    NEEDS WORK
    """
    ax = plt.gca() if ax is None else ax
    caxs = []
    for i, x_i in enumerate(x[:-1]):
        X_in, Y_in = np.meshgrid(np.array([x[i], x[i+1]]), Y[:, i])
        caxs += [ax.pcolormesh(X_in, Y_in, np.atleast_2d(C[:, i]).T)]

    # For every column, set all the desired properties
    # The following three have to be the same for all columns
    # So, make a record if they aren't specified
    vmin, vmax, norm = [], [], []
    for cax in caxs:
        for k, v in kwargs:
            cax.set(k=v)
            vmin += [cax.get_vmin()]
            vmax += [cax.get_vmax()]
            # norm += caxs[-1].get_norm()

        cax.set(vmin=min(vmin), vmax=max(vmax))

    # Each element of caxs is pretty much the same, so just return the first one
    return caxs[0]


def get_xy_line_data(ax):
    """Get all data on an axis corresponding to line plots"""
    all_lines = [x for x in ax.get_children() if 'Line2D' in str(x)]
    xdata = np.hstack([line.get_xdata() for line in all_lines])
    ydata = np.hstack([line.get_ydata() for line in all_lines])

    return xdata, ydata


def get_all_handles_labels(axs):
    """Get all handles and labels of lines with a label property

    Useful for custom legend"""

    handles = np.array(flatten2Dlist(
        [ax.get_legend_handles_labels()[0] for ax in axs]))
    labels = np.array(flatten2Dlist(
        [ax.get_legend_handles_labels()[1] for ax in axs]))

    labels, inds = np.unique(labels, return_index=True)
    handles = list(handles[inds])
    labels = list(labels)

    return handles, labels


def add_colorbar_to_subplot(cax, fig):
    """Adds a colorbar as per normal, but ensures subplots are appropriately
    adjusted"""
    axs = fig.get_axes()
    cbar_to_rm = fig.colorbar(cax, ax=axs)
    cbar = fig.colorbar(cax)
    cbar_to_rm.remove()
    plt.draw()
    return cbar

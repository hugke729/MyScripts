import warnings
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import pickle
from cycler import cycler
plt.rc('axes', prop_cycle=cycler('color', ['r', 'g', 'y', 'm']))

m = pickle.load(open('/home/hugke729/PhD/Python/Maps/penny_strait.pickle', 'rb'))


class LineBuilder:

    """Draw line on a figure with mouse
    Left click to add point
    Right click to remove last point
    Middle click (or left + right) to finish

    x, y data are saved to '~/.tmp/' + name + '.txt'"""
    # http://matplotlib.org/users/event_handling.html

    def __init__(self, line, name='temp'):
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.name = name

    def __call__(self, event):
        mode = plt.get_current_fig_manager().toolbar.mode
        if mode in ['zoom rect', 'pan/zoom']:
            # Zooming or panning, so don't count as clicks
            return
        if event.inaxes != self.line.axes:
            # Outside axes
            return
        if event.button == 1:
            # Left click
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
        elif event.button == 3:
            # Right click
            self.xs = self.xs[:-1]
            self.ys = self.ys[:-1]
        elif event.button == 2:
            # Middle click
            self.line.figure.canvas.mpl_disconnect(self.cid)
            save_line_data(self.line, '/home/hugke729/.tmp/' + self.name + '.txt')

        # Update the figure with new line details
        self.line.set_data(self.xs, self.ys)
        self.line.figure.canvas.draw()


def linspace_xy(x, y, d):
    """from input x and y, create new x, y with additional points in between
    spaced by d"""
    x, y = np.array(x), np.array(y)
    dists = (np.sqrt((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2) / d).astype(int)
    dists_cs = np.insert(np.cumsum(dists), 0, 0)
    newx, newy = np.zeros(sum(dists) + 1), np.zeros(sum(dists) + 1)
    newx[dists_cs] = x
    newy[dists_cs] = y
    i = 0
    for d1, d2 in zip(dists_cs[:-1], dists_cs[1:]):
        newx[d1:d2] = np.linspace(x[i], x[i+1], dists[i])
        newy[d1:d2] = np.linspace(y[i], y[i+1], dists[i])
        i += 1

    return newx, newy


def save_line_data(line, name='temp'):
    """save x, y coordinates of line to '~/.tmp/' + name + '.txt'"""
    x, y = line.get_xdata(), line.get_ydata()
    x, y = x[1:], y[1:]  # remove initial NaN value
    # x, y = linspace_xy(x, y, 1e3)
    xy = np.column_stack((x, y))

    with open(name, 'wt') as f:
        fmt_str = '{0:f} {1:f}'
        for row in xy:
            print(fmt_str.format(row[0], row[1]), file=f)


def add_map_line(name='temp'):
    """Add a line to the map with the mouse"""
    ax = plt.gca()
    # Add instructions
    ax.set_title('Left click to add point\n' +
                 'Right click to remove last point\n' +
                 'Middle click (or Left + Right) to finish')
    plt.show(block=False)
    line, = ax.plot(np.nan, np.nan, 'r')  # initialise an empty line
    plt.show()
    LineBuilder(line, name)


def rm_map_line(which_line=-1):
    """Remove a line from the map figure. Default: remove last added line"""
    ax = plt.gca()
    ax.lines[which_line].remove()
    plt.draw()


def add_labels(ax, m=m):
    """Add labels for islands around Penny Strait"""
    # Add a few markers
    labels = [[-94.91178, 75.11714, 'Cornwallis Is'],
              [-90.9207, 75.52962, 'Devon Is'],
              [-99.6325, 75.80672, 'Bathurst Is']]
    for row in labels:
        xlabel, ylabel = m(row[0], row[1])
        ax.annotate(row[2], xy=(xlabel, ylabel), ha='center', fontsize=6)


def simple_map(m, name, x, y, z=None):
    """Create and save a map with just the islands and the line drawn"""
    if z is None:
        fig, ax = plt.subplots(figsize=(11, 8.5))
    else:
        fig, (ax, ax_z) = plt.subplots(nrows=2, figsize=(11, 8.5),
                                       gridspec_kw={'height_ratios': [2, 1]})
        ax_z.plot(np.sqrt((x-x[0])**2 + (y-y[0])**2)/1e3, z)
        ax_z.invert_yaxis()
        ax_z.set_ylim(top=0)
        ax_z.set(xlabel='Distance from dot (km)', ylabel='Depth (m)')

    m.fillcontinents(ax=ax)
    m.plot(x, y, 'k', ax=ax)
    m.plot(x[0], y[0], 'ko', ax=ax)
    add_labels(ax=ax)
    plt.tight_layout()
    plt.savefig(name + '.pdf')
    plt.close(fig)


def print_latlon(latlon, fmt='dm', name='temp', z=None, console_output=True):
    """Save latlon array to name.txt as either decimal degrees or
    decimal minutes"""
    with open(name + '.txt', 'wt') as f:
        # Print header
        if z is not None:
            z_str = ' Depths (m)'
        else:
            z_str = ''

        if fmt == 'dd':
            print('Lat (decimal degrees) Lon (decimal degrees)' + z_str, file=f)
            fmt_str = '{0:g} {1:g}'
        elif fmt == 'dm':
            print('Lat (degrees, decimal minutes) ' +
                  'Lon (degrees, decimal minutes)' + z_str, file=f)
            lat_str = '{0:g} {1:05.2f} '
            lon_str = '{2:g} {3:05.2f}'
            fmt_str = lat_str + lon_str

        for i, row in enumerate(latlon):
            lat, lon = row[0], row[1]
            latd, latm = decdeg2degdm(lat)
            lond, lonm = decdeg2degdm(lon)
            if fmt == 'dd':
                output = fmt_str.format(lat, lon)
            elif fmt == 'dm':
                output = fmt_str.format(latd, latm, lond, lonm)

            if z is None:
                print(output, file=f)
                print(output)
            else:
                print(output + ' {0:4.0f}'.format(z[i]), file=f)
                print(output + ' {0:4.0f}'.format(z[i]))


def output_map_line(m=m, name='temp', fmt='dm', xyz=None):
    """Output coordinates of the line drawn and save a simple image as well"""
    try:
        x, y = np.genfromtxt('/home/hugke729/.tmp/' + name + '.txt').T
        lon, lat = m(x, y, inverse=True)
        latlon = np.column_stack((lat, lon))

        if xyz is not None:
            X, Y, Z = xyz
            z = np.zeros(x.shape)
            x_bounds = (X[0, :] - np.gradient(X)[1])[0, :]
            y_bounds = (Y - np.gradient(Y)[0])[:, 0]
            for i, _ in enumerate(x):
                z[i] = Z[np.argmax(y[i] < y_bounds) - 1,
                         np.argmax(x[i] < x_bounds) - 1]

        if xyz is None:
            print_latlon(latlon, fmt=fmt, name=name)
            simple_map(m, name, x, y)
        else:
            print_latlon(latlon, z=z, fmt=fmt, name=name)
            simple_map(m, name, x, y, z=z)

    except IOError:
        print('Specify the name used by add_map_line')


def decdeg2degdm(dd):
    is_positive = dd >= 0
    dd = abs(dd)
    degrees, dm = divmod(dd*60, 60)
    degrees = degrees if is_positive else -degrees
    return (degrees, dm)


def disp_km(ax):
    """Displays curson position in kilometres"""

    def format_coord(x, y):
        xy_str = 'x(km) = {0:6.1f}    y(km) = {1:6.1f}'
        return xy_str.format(x/1e3, y/1e3)

    ax.format_coord = format_coord
    return ax


def disp_latlon(ax, m, form='dd', xyz=None):
    """Displays lat and lon for the cursor position

    Inputs
    ------
    ax: matplotlib axis
    m: basemap object
    form:'dd' for decimal degrees, 'dm' for degrees, decimal minutes
    xyz: tuple of X, Y, Z where X, Y are 2D cartesian grids with z having
    the corresponding depths
    """

    # Let me put first two arguments in wrong order
    if type(ax) is Basemap:
        ax, m = m, ax

    def format_coord(x, y):
        """Create string showing lat/lon and x/y in km"""
        lon, lat = m(x, y, inverse=True)
        xy_str = 'x(km) = {4:6.1f}    y(km) = {5:6.1f}'

        if form == 'dm':
            latd, latm = decdeg2degdm(lat)
            lond, lonm = decdeg2degdm(lon)
            lat_str = 'Lat = {0:g}\N{DEGREE SIGN} {1:05.2f}\N{PRIME}   '
            lon_str = 'Lon = {2:g}\N{DEGREE SIGN} {3:05.2f}\N{PRIME}   |   '
            lat_lon_str = lat_str + lon_str
            fmt_str = lat_lon_str + xy_str
            output = (fmt_str).format(latd, latm, lond, lonm, x/1e3, y/1e3)

        elif form == 'dd':
            lat_lon_str = 'Lat = {0:.5f}   Lon = {1:.5f}   |   '
            fmt_str = lat_lon_str + xy_str
            output = (fmt_str).format(lat, lon, 0, 0, x/1e3, y/1e3)

        # if X, Y, Z are specified, then look up closest Z value to current
        # (x, y) mouse coordinate
        if xyz is not None:
            X, Y, Z = xyz
            z_str = '   |   Depth = {0:3.0f} m'
            x_bounds = (X[0, :] - np.gradient(X)[1])[0, :]
            y_bounds = (Y - np.gradient(Y)[0])[:, 0]
            z = Z[np.argmax(y < y_bounds) - 1, np.argmax(x < x_bounds) - 1]
            output += z_str.format(z)

        return output

    ax.format_coord = format_coord
    return ax


def click_km():
    warnings.filterwarnings('ignore', 'Using default event loop*.')
    pts = plt.ginput(2)
    x0, y0 = pts[0]
    x1, y1 = pts[1]
    dist_km = np.hypot(x1-x0, y1-y0)/1e3

    print('{0:6.1f} km'.format(dist_km))

    return dist_km

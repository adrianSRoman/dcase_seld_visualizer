###### Plotting Utils #######
# Refer to: https://github.com/imagingofthings/DeepWave/blob/master/datasets/Pyramic/color_plot.py
import collections.abc as abc

import astropy.coordinates as coord
import astropy.units as u
import matplotlib.pyplot as plt
import mpl_toolkits.basemap as basemap
import matplotlib.tri as tri
import numpy as np

from sklearn.cluster import KMeans

import csv
def wrapped_rad2deg(lat_r, lon_r):
    """
    Equatorial coordinate [rad] -> [deg] unit conversion.
    Output longitude guaranteed to lie in [-180, 180) [deg].
    """
    lat_d = coord.Angle(lat_r * u.rad).to_value(u.deg)
    lon_d = coord.Angle(lon_r * u.rad).wrap_at(180 * u.deg).to_value(u.deg)
    return lat_d, lon_d


def cart2pol(x, y, z):
    """
    Cartesian coordinates to Polar coordinates.
    """
    cart = coord.CartesianRepresentation(x, y, z)
    sph = coord.SphericalRepresentation.from_cartesian(cart)

    r = sph.distance.to_value(u.dimensionless_unscaled)
    colat = u.Quantity(90 * u.deg - sph.lat).to_value(u.rad)
    lon = u.Quantity(sph.lon).to_value(u.rad)

    return r, colat, lon


def cart2eq(x, y, z):
    """
    Cartesian coordinates to Equatorial coordinates.
    """
    r, colat, lon = cart2pol(x, y, z)
    lat = (np.pi / 2) - colat
    return r, lat, lon


def is_scalar(x):
    """
    Return :py:obj:`True` if `x` is a scalar object.
    """
    if not isinstance(x, abc.Container):
        return True

    return False


def eq2cart(r, lat, lon):
    """
    Equatorial coordinates to Cartesian coordinates.
    """
    r = np.array([r]) if is_scalar(r) else np.array(r, copy=False)
    if np.any(r < 0):
        raise ValueError("Parameter[r] must be non-negative.")

    XYZ = (
        coord.SphericalRepresentation(lon * u.rad, lat * u.rad, r)
            .to_cartesian()
            .xyz.to_value(u.dimensionless_unscaled)
    )
    return XYZ


def cmap_from_list(name, colors, N=256, gamma=1.0):
    """
    Parameters
    ----------
    name : str
    colors :
        * a list of (value, color) tuples; or
        * list of color strings
    N : int
        Number of RGB quantization levels.
    gamma : float
        Something?

    Returns
    -------
    cmap : :py:class:`matplotlib.colors.LinearSegmentedColormap`
    """
    from collections.abc import Sized
    import matplotlib.colors

    if not isinstance(colors, abc.Iterable):
        raise ValueError('colors must be iterable')

    if (isinstance(colors[0], Sized) and
            (len(colors[0]) == 2) and
            (not isinstance(colors[0], str))):  # List of value, color pairs
        vals, colors = zip(*colors)
    else:
        vals = np.linspace(0, 1, len(colors))

    cdict = dict(red=[], green=[], blue=[], alpha=[])
    for val, color in zip(vals, colors):
        r, g, b, a = matplotlib.colors.to_rgba(color)
        cdict['red'].append((val, r, r))
        cdict['green'].append((val, g, g))
        cdict['blue'].append((val, b, b))
        cdict['alpha'].append((val, a, a))

    return matplotlib.colors.LinearSegmentedColormap(name, cdict, N, gamma)


def draw_map(I, R, lon_ticks, ground_truth_info=None, catalog=None, show_labels=False, show_axis=False):
    """
    Parameters
    ==========
    I : :py:class:`~numpy.ndarray`
        (3, N_px)
    R : :py:class:`~numpy.ndarray`
        (3, N_px)
    """

    _, R_el, R_az = cart2eq(*R)
    R_el, R_az = wrapped_rad2deg(R_el, R_az)
    R_el_min, R_el_max = np.around([np.min(R_el), np.max(R_el)])
    R_az_min, R_az_max = np.around([np.min(R_az), np.max(R_az)])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    bm = basemap.Basemap(projection='mill',
                         llcrnrlat=R_el_min, urcrnrlat=R_el_max,
                         llcrnrlon=R_az_min, urcrnrlon=R_az_max,
                         resolution='c',
                         ax=ax)

    if show_axis:
        bm_labels = [1, 0, 0, 1]
    else:
        bm_labels = [0, 0, 0, 0]
    bm.drawparallels(np.linspace(R_el_min, R_el_max, 5),
                     color='w', dashes=[1, 0], labels=bm_labels, labelstyle='+/-',
                     textcolor='#565656', zorder=0, linewidth=2)
    bm.drawmeridians(lon_ticks,
                     color='w', dashes=[1, 0], labels=bm_labels, labelstyle='+/-',
                     textcolor='#565656', zorder=0, linewidth=2)

    if show_labels:
        ax.set_xlabel('Azimuth (degrees)', labelpad=20)
        ax.set_ylabel('Elevation (degrees)', labelpad=40)
    ax.set_title('Sound Events (SELDNet Prediction)', pad=20)
    R_x, R_y = bm(R_az, R_el)
    triangulation = tri.Triangulation(R_x, R_y)

    N_px = I.shape[1]
    mycmap = cmap_from_list('mycmap', I.T, N=N_px)
    colors_cmap = np.arange(N_px)
    ax.tripcolor(triangulation, colors_cmap, cmap=mycmap,
                 shading='gouraud', alpha=0.9, edgecolors='w', linewidth=0.1)

    # Npts = 12
    # I_s = np.square(I).sum(axis=0)
    # max_idx = I_s.argsort()[-Npts:][::-1]
    # x_y = np.column_stack((R_x[max_idx], R_y[max_idx]))
    # km_res = KMeans(n_clusters=1).fit(x_y)
    # clusters = km_res.cluster_centers_
    
    if ground_truth_info and len(ground_truth_info["gt"]) != 0:
        for i in range(len(ground_truth_info["gt"])):
            ground_truth = ground_truth_info["gt"][i]
            gt_num = ground_truth_info["num"][i]
            gt_color = ground_truth_info["color"][i]
            gt_x, gt_y = bm(ground_truth[0], ground_truth[1])
            ax.scatter(gt_x, gt_y, s=500, alpha=0.3, c=gt_color)
            ax.text(gt_x, gt_y, gt_num, ha='center', va='center', color='white')
    # # Note, the cases where the intensity is at 180 deg gives a funny error with kmeans. 
    # # Essentially it finds the centroid right at 0 deg since the intensity is split
    # # Maybe think on how to solve this @Sivan
    return fig, ax, triangulation

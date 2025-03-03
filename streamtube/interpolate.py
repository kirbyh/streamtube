"""
Interpolator functions

Kirby Heck
2025 March 3
"""

import numpy as np
from scipy.interpolate import interpn


def linear_interp(x, xp, yp, axis=0, extrapolate=True): 
    """
    Simple linear interpolation/extrapolation function 
    """

    if not hasattr(x, '__iter__'): 
        x = [x]
    
    l_ls = np.zeros_like(x, dtype=int)
    u_ls = np.zeros_like(x, dtype=int)
    min_x = xp.min()

    # this is slow for large len(x)
    for k, xi in enumerate(x): 
        # find nearest indices
        if xi <= min_x: 
            if extrapolate: 
                lower = 0
            else: 
                raise ValueError(
                    'Value {:.3f} below min(x_p). Set `extrapolate=True` to extrapolate.'.format(xi)
                    )
        else: 
            lower = np.where(xp < xi)[0][-1]
        if lower == len(xp) - 1:  # need to extrapolate
            if extrapolate: 
                lower -= 1
            else: 
                raise ValueError(
                    'Value {:.3f} above max(x_p). Set `extrapolate=True` to extrapolate.'.format(xi)
                    )
        u_ls[k] = lower + 1
        l_ls[k] = lower

    # actually do the interpolating: 
    # y(x*) = y_lower + (x*-x_lower)/(x_upper-x_lower) * (y_upper - y_lower)
    x_l = xp[l_ls] 
    x_u = xp[u_ls]
    y_l = np.take(yp, indices=l_ls, axis=axis)
    y_u = np.take(yp, indices=u_ls, axis=axis) 

    # ensure we broadcast multiplication over the `axis` axis
    s = [None] * yp.ndim
    s[axis] = slice(None)

    y = y_l + ((x - x_l) / (x_u - x_l))[tuple(s)] *  (y_u - y_l)
    return np.squeeze(y)
    

def trilinear_interpolation(xp, f, points):
    """
    Perform trilinear interpolation for a 3D array f onto a list of points (xi, yi, zi).
    Same as interpn for linear interpolation, but less function overhead. 

    Parameters:
    xp = (x, y, z) : 1D arrays
        The grid points in each dimension.
    f : 3D array
        The values at the grid points.
    points : 2D array
        The points at which to interpolate, shape (n_points, 3).

    Returns:
    values : 1D array
        Interpolated values at the given points.
    """
    x, y, z = xp
    points = np.atleast_2d(points)
    x0 = np.searchsorted(x, points[:, 0]) - 1
    y0 = np.searchsorted(y, points[:, 1]) - 1
    z0 = np.searchsorted(z, points[:, 2]) - 1

    x0 = np.clip(x0, 0, len(x) - 2)
    y0 = np.clip(y0, 0, len(y) - 2)
    z0 = np.clip(z0, 0, len(z) - 2)

    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    xd = (points[:, 0] - x[x0]) / (x[x1] - x[x0])
    yd = (points[:, 1] - y[y0]) / (y[y1] - y[y0])
    zd = (points[:, 2] - z[z0]) / (z[z1] - z[z0])

    c00 = f[x0, y0, z0] * (1 - xd) + f[x1, y0, z0] * xd
    c01 = f[x0, y0, z1] * (1 - xd) + f[x1, y0, z1] * xd
    c10 = f[x0, y1, z0] * (1 - xd) + f[x1, y1, z0] * xd
    c11 = f[x0, y1, z1] * (1 - xd) + f[x1, y1, z1] * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    values = c0 * (1 - zd) + c1 * zd

    return values

    
def sample_at_point(xp, x, y, z, *args): 
    """
    Samples fields in *args at point xp. 
    """
    nd = len(args)
    if nd == 1: 
        return interpn((x, y, z), *args, xp)[0]
    # broadcasting magicks: 
    return interpn((range(nd), x, y, z), args, [(k,) + xp for k in range(nd)]) 


def sample_at_points(xps, x, y, z, *args): 
    """
    Samples fields in *args at points xps. 
    # TODO: improve this, vectorize
    """
    ret = []
    for xp in xps: 
        ret.append(sample_at_point(xp, x, y, z, *args)) 
    return ret


def interp_and_mask(xloc, x, field, mask): 
    """
    Interpolates fields in x, masks the resultwith `mask`, and 
    returns the yz-averaged result
    """
    field2 = linear_interp(x=xloc, xp=x, yp=field, axis=0)
    mask = linear_interp(x=xloc, xp=x, yp=mask, axis=0)
    normfact = np.sum(mask)
    return np.sum(field2 * mask) / normfact
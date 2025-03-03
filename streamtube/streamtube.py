"""
Streamtube object for creating streamtube masks

Kirby Heck
2023 June 30
Edited 2025 March 3
"""

import numpy as np
from matplotlib import path

from . import stream3 as stream3
from .integrate import streamline_2way, streamlines_2way
from .block import block_reduce_centered


class Streamtube:
    """
    Creates a streamtube object for masking arrays
    """

    def __init__(
        self, x, y, z, u, v, w, masktype=None, maxlength=100.0, method="solve_ivp"
    ):
        """
        Creates a streamtube object.

        Parameters
        ----------
        x, y, z : 1D or 3D arrays
            See stream3.Grid3 class
        u, v, w : (Nx, Ny, Nz) arrays
            3D arrays of velocities in the x, y, and z directions, respectively.
        masktype : either 'float' or 'bool'
            Default is a floating point mask (see self.compute_mask())
        method : str
            Integration method for streamlines. Default is 'solve_ivp'. Other
            (older, deprecated) options are 'stream3' and 'integrator' (worst).
        """
        self.u = np.array(u)
        self.v = np.array(v)
        self.w = np.array(w)
        self.maxlength = maxlength  # maximum streamline length (non-dim)
        self.method = method  # either 'solve_ivp', 'stream3' or 'integrator'

        self.grid = stream3.Grid3(x, y, z)

        # basic checks: u, v, w are the right size
        if (
            u.shape != self.grid.shape
            or v.shape != self.grid.shape
            or w.shape != self.grid.shape
        ):
            raise ValueError("'u' and 'v', 'w' must match the shape of 'Grid(x, y, z)'")

        self.start_points = None
        self.trajectories = None
        self.mask = None

        # we might want a floating point mask and not just a boolean mask
        self.set_masktype(masktype)

    def get_start_points(
        self, sample_disk=True, return_pts=False, R=0.5, N=64, origin=(0, 0, 0)
    ):
        """
        Set initial points to seed the streamtube

        Parameters
        ----------
        sample_disk : bool
            Default True: Calls start_points_disk()
        return_pts : bool
            if True, returns the starting points.
            Default: False (saves to self.start_points)
        R : float
            Radius for start_points_disk()
        N : integer
            Number of points for start_points_disk()
        origin : (x, y, z) tuple
            Origin of start points
        """
        if sample_disk:
            # default: sample on disk at origin
            start_points = start_points_disk(R, N, origin)
        else:
            raise ValueError(
                "get_start_points(): sample_disk=True is the only option implemented"
            )

        if return_pts:
            return start_points
        else:
            self.start_points = start_points

    def compute_streamtube(
        self,
        start_points=None,
        return_trajectories=False,
        method=None,
        R=0.5,
        N=128,
        origin=(0, 0, 0),
        stream_kwargs=None,
    ):
        """
        Computes a streamtube from given starting points
        """

        if start_points is None:
            self.get_start_points(R=R, N=N, origin=origin)
            start_points = self.start_points

        method = self.method if method is None else method
        # compute streamline trajectories in both directions
        if method == "solve_ivp":
            # THIS IS THE RECOMMENDED METHOD AND THE NEW DEFAULT
            _kwargs = dict(atol=1e-5, rtol=1e-5, method="RK23")  # default values
            _kwargs.update(stream_kwargs or dict())
            trajectories = streamlines_2way(
                self.grid.x,
                self.grid.y,
                self.grid.z,
                self.u,
                self.v,
                self.w,
                start_points,
                ivp_kwargs=_kwargs,
            )

        elif method == "stream3":
            _kwargs = dict(minlength=0, check_overlap=False)
            _kwargs.update(stream_kwargs or dict())

            trajectories = stream3.stream3(
                self.grid.x,
                self.grid.y,
                self.grid.z,
                self.u,
                self.v,
                self.w,
                start_points,
                integration_direction="both",
                maxlength=self.maxlength,
                **_kwargs
            )

        elif method == "integrator":
            trajectories = []
            _kwargs = dict(dt=0.5, T=[0, 50])
            _kwargs.update(stream_kwargs or dict())

            for s in start_points:
                _, t = streamline_2way(
                    self.grid.x,
                    self.grid.y,
                    self.grid.z,
                    self.u,
                    self.v,
                    self.w,
                    x0=s,
                    **_kwargs
                )
                trajectories.append(t)

        else:
            raise ValueError("check self.method")

        if return_trajectories:
            return trajectories
        else:
            self.trajectories = trajectories

    def compute_mask(
        self,
        start_points=None,
        trajectories=None,
        recompute_trajectories=False,
        upsample_fact=5,
        method=None,
        stream_kwargs=None,
        masktype=None,
        return_mask=False,
        R=0.5,
        N=128,
        origin=(0, 0, 0),
        downsample_func=block_reduce_centered,
    ):
        """
        Computes a streamtube mask from streamtube trajectories.

        Parameters
        ----------
        start_points : optional, list of starting points
            Default None; calls self.get_start_points() with default values
        trajectories : optional, list of trajectories
            Default None; calls self.compute_streamtube()
        upsample_fact : integer
            Upsampling rate for streamtube mask (see weighted_mask_streamtube())
        method : str
            Method for computing streamlines. Default: None, uses self.method.
        masktype : `float` or `bool`
            Switches between floating point or boolean mask.
        return_mask : bool
            if True, returns the mask. Default: False, access mask with self.mask
        R, N, origin : Any
            see get_start_points()
        """

        # need streamline trajectories
        if trajectories is None:
            if self.trajectories is None or recompute_trajectories:
                self.compute_streamtube(
                    start_points=start_points,
                    method=method,
                    R=R,
                    N=N,
                    origin=origin,
                    stream_kwargs=stream_kwargs,
                )
            trajectories = self.trajectories

        if masktype is not None:
            self.set_masktype(masktype)

        # compute mask
        if self.masktype == bool:
            mask = mask_streamtube(trajectories, self.grid.x, self.grid.y, self.grid.z)
        else:
            mask = weighted_mask_streamtube(
                trajectories,
                self.grid.x,
                self.grid.y,
                self.grid.z,
                upsample_fact,
                downsample_func=downsample_func,
            )

        if return_mask:
            return mask
        else:
            self.mask = mask

    def set_masktype(self, masktype):
        """
        Toggles between masktype=bool and self.masktype=float
        """
        if masktype == bool or masktype == "bool":
            self.masktype = bool
        elif (
            masktype is None or masktype == "float" or masktype == float
        ):  # this is default
            self.masktype = float
        else:
            raise ValueError("__init__(): `masktype` must be 'bool' or 'float'")


def start_points_disk(R=0.5, N=128, origin=(0, 0, 0)):
    """
    Choose starting points for the streamtube: ring centered at the
    origin in the y-z plane
    """
    dtheta = 2 * np.pi / N
    theta = np.arange(0, 2 * np.pi, dtheta)
    y0, z0 = np.cos(theta) * R + origin[1], np.sin(theta) * R + origin[2]
    x0 = np.zeros(y0.shape) + origin[0]
    return np.array([x0, y0, z0]).T  # returns Nx3 array


def mask_streamtube(trajectories, x, y, z):
    """
    Computes a mask along some given grid axes x, y, z with the give trajectories.

    Parameters
    ----------
    trajectories : list of 3D streamlines
        output streamlines from stream3.stream3
    x, y, z : 1D or 3D axes.
        See stream3.Grid3

    Returns
    -------
    mask : boolean array of cells, TRUE = inside streamtube, FALSE = outside.
        mask has dimensions nx x ny x nz.

    Note:
    Trajectories do not have to be generated on the same resolution as the given
    grid (results will be interpolated), but they do need to share a common origin.
    """

    # step 1: create grid and boolean mask
    grid = stream3.Grid3(x, y, z)
    mask = np.zeros(grid.shape, dtype=bool)

    # step 2: interpolate the trajectories values to the x-grid
    traj_interp = np.zeros(
        (grid.nx, len(trajectories), 2)
    )  # we'll store the trajectory points in here
    for k, t in enumerate(trajectories):
        traj_interp[:, k, 0] = np.interp(x, t[0], t[1])  # linear interpolation y-values
        traj_interp[:, k, 1] = np.interp(x, t[0], t[2])  # linear interpolation y-values

    # step 3: loop through x-values. For each x-value, detect points
    Y, Z = np.meshgrid(grid.y, grid.z, indexing="ij")  # 2D plane
    yv, zv = Y.flatten(), Z.flatten()  # flatten points
    pts = np.array([yv, zv]).T  # (Ny*Nz, 2) array
    for k in range(grid.nx):
        shape = path.Path(traj_interp[k, :, :])
        # detect points, reshape to y-z grid:
        mask[k, :, :] = shape.contains_points(pts).reshape(grid.ny, grid.nz)

    return mask


def weighted_mask_streamtube(
    trajectories, x, y, z, upsample_fact, downsample_func=block_reduce_centered
):
    """
    Creates a floating point (weighted) mask of values [0, 1.].

    Three-step process:
        1. Snip to (y, z) streamtube extent and upsample fields within those extents
        2. Compute upsampled wake mask
        3. Downsample with block averaging and "insert" into the original grid shape

    Parameters
    ----------
    trajectories : list of 3D streamlines
        see stream3.py
    x, y, z : arrays or

    """

    # 1. Snip to streamtube extent and upsample lines
    grid = stream3.Grid3(x, y, z)

    miny = np.min([t[1].min() for t in trajectories])
    maxy = np.max([t[1].max() for t in trajectories])
    minz = np.min([t[2].min() for t in trajectories])
    maxz = np.max([t[2].max() for t in trajectories])

    yids = np.where((y > miny) & (y < maxy))[0][[0, -1]]
    zids = np.where((z > minz) & (z < maxz))[0][[0, -1]]
    ibuff = [-1, 2]  # index buffer
    yfilt = slice(max(yids[0] + ibuff[0], 0), min(yids[1] + ibuff[1], len(y)))
    zfilt = slice(
        max(zids[0] + ibuff[0], 0), min(zids[1] + ibuff[1], len(z))
    )  # ugly but works
    ys = y[yfilt]
    zs = z[zfilt]  # subsets of original axes

    y_up = upsample_line(ys, upsample_fact)
    z_up = upsample_line(zs, upsample_fact)

    # 2. Streamtube mask:
    mask_up = mask_streamtube(trajectories, x, y_up, z_up)

    # 3. Downsample and insert into original grid
    if downsample_func is None:
        return y_up, z_up, mask_up  # return upsampled mask with axes, for debugging

    mask = np.zeros(grid.shape, dtype=float)  # original; will be returned
    mask_down = downsample_func(
        mask_up, (1, upsample_fact, upsample_fact), func=np.mean
    )
    mask[:, yfilt, zfilt] = mask_down

    return mask


def upsample_line(x, nfact):
    """
    Upsamples an array `x` by integer factor nfact. Assumes x is evenly spaced.
    """
    nx = len(x)
    dx = x[1] - x[0]
    x_up = np.arange(nfact * nx - nfact + 1) * dx / nfact + x.min()

    return x_up


def interp_to_x(trajectories, x):
    """
    Interpolates trajectories to the 1D array x.

    Parameters
    ----------
    trajectories : list of 3D streamlines
        output streamlines from stream3.stream3
    x : 1D array

    Returns
    -------
    ndarray (nx, nt, 2)
        Array of values, where nx is the number of points in x, nt is the number of
        trajectories (streamlines), and the final axis is for [y, z] location.
    """
    nx = len(x)

    # interpolate the trajectories values to the x-grid
    traj_interp = np.zeros(
        (nx, len(trajectories), 2)
    )  # we'll store the trajectory points in here
    for k, t in enumerate(trajectories):
        traj_interp[:, k, 0] = np.interp(x, t[0], t[1])  # linear interpolation y-values
        traj_interp[:, k, 1] = np.interp(x, t[0], t[2])  # linear interpolation z-values

    return traj_interp


def interp_to_xyz(trajectories, x):
    """
    Without deprecating `interp_to_x`, this function does what
    that function probably should have done.

    Returns
    -------
    3d array
        [n_trajectories, 3, n_x]
    """
    nx = len(x)
    traj_interp = np.zeros((len(trajectories), 3, nx))
    for k, t in enumerate(trajectories):
        traj_interp[k, 0, :] = x
        traj_interp[k, 1, :] = np.interp(x, t[0], t[1])
        traj_interp[k, 2, :] = np.interp(x, t[0], t[2])
    return traj_interp


def interp_to_theta(traj_interp, theta=None, n_theta=64):
    """
    Interpolates trajectories to a unified azimuthal coordinate.

    Parameters
    ----------
    traj_interp : 3D array
        Trajectories interpolated onto an x-grid from interp_to_xgrid
    theta : 1D array, optional
        Array of azimuthal values. Defaults to None.
    n_theta : int, optional
        Number of theta grid points if `theta` is None. Otherwise, this
        variable is not used.

    Returns
    -------
    theta, y, z
    """
    if theta is None:
        # include both 0 and 2*pi for smooth surfaces
        theta = np.linspace(0, 2 * np.pi, n_theta + 1)
        n_theta += 1
    else:
        n_theta = len(theta)

    nx = traj_interp.shape[0]
    y = np.zeros((nx, n_theta))
    z = np.zeros_like(y)

    # compute existing (non-gridded, in x) azimuthal angles
    _y = traj_interp[..., 0]
    _z = traj_interp[..., 1]
    _theta = np.arctan2(_z, _y)

    # do the interpolation:
    for xi in range(nx):
        y[xi, :] = np.interp(theta, _theta[xi, :], _y[xi, :], period=2 * np.pi)
        z[xi, :] = np.interp(theta, _theta[xi, :], _z[xi, :], period=2 * np.pi)

    return theta, y, z


def interp_x_theta(trajectories, x, theta=None, n_theta=64, return_theta=False):
    """
    Combines the interpolation to an x-grid and theta-grid into one function.
    Returns a parameterization of the trajectories in (x, theta) space where
    theta is the azimuthal coordinate.

    This is useful for plotting the streamtube, as in Axes3D.plot_surface(xG, yG, zG)

    Parameters
    ----------
    traj_interp : 3D array
        Trajectories interpolated onto an x-grid from interp_to_xgrid.
    x : 1D array
    theta : 1D array, optional
        Array of azimuthal values. Defaults to None.
    n_theta : int, optional
        Number of theta grid points if `theta` is None. Otherwise, this
        variable is not used.

    Returns
    -------
    xG, yG, zG : 2D array
        Array of streamtube coordinates parameterized by x, theta.
    """
    # interpolate to x-grid
    traj_xinterp = interp_to_x(trajectories, x)
    # interpolate to theta grid
    theta, yG, zG = interp_to_theta(traj_xinterp, theta=theta, n_theta=n_theta)
    # form x-theta meshgrid (theta unused)
    xG, tG = np.meshgrid(x, theta, indexing="ij")

    if return_theta:
        return xG, tG, yG, zG
    else:
        return xG, yG, zG

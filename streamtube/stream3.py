"""
Streamline computation for 3D vector fields.

Based on Matplotlib's `streamplot` function

Kirby Heck
2023 June 29
Edited 2025 March 3
"""

import numpy as np


__all__ = ["stream3"]


def stream3(
    x,
    y,
    z,
    u,
    v,
    w,
    start_points,
    density=1,
    minlength=0.0,
    maxlength=None,
    integration_direction="both",
    check_overlap=False,
):
    """
    Compute 3D streamlines of a vector flow field.

    Parameters
    ----------
    x, y, z : 1D/3D evenly spaced arrays.
        3D arrays are expected to have x in the 0-axis, y in the 1-axis, z in the 2-axis
    u, v, w : 3D arrays. Velocities in the x, y, z directions (respectively)
    start_points : Nx3 array of tuples to seed streamlines.
    density : float or (float, float, float)
        Controls the closeness of streamlines. When ``density = 1``, the domain
        is divided into a 30x30 grid. *density* linearly scales this grid.
        Each cell in the grid can have, at most, one traversing streamline.
        For different densities in each direction, use a tuple
        (density_x, density_y, density_z).
    minlength : Minimum streamline length in grid coordinates. Default 0.
    maxlength : Maximum streamline length in grid coordinates. Default None.
    integration_direction : {'forward', 'backward', 'both'}, default: 'both'
        Integrate the streamline in forward, backward or both directions.
    check_overlap : boolean, prevents overlapping streamlines if True. Default False.
    """
    grid = Grid3(x, y, z)
    mask = StreamMask3(density)
    dmap = DomainMap3(grid, mask, check_overlap=check_overlap)

    # Sanity checks.
    if u.shape != grid.shape or v.shape != grid.shape or w.shape != grid.shape:
        raise ValueError("'u' and 'v', 'w' must match the shape of 'Grid(x, y, z)'")

    u = np.ma.masked_invalid(u)
    v = np.ma.masked_invalid(v)
    w = np.ma.masked_invalid(w)

    trajectories = []
    sp = np.asanyarray(start_points, dtype=float).copy()  # array of starting points

    # Check if start_points are outside the data boundaries
    for xs, ys, zs in sp:
        if not (
            grid.x_origin <= xs <= grid.x_origin + grid.Lx
            and grid.y_origin <= ys <= grid.y_origin + grid.Ly
            and grid.z_origin <= zs <= grid.z_origin + grid.Lz
        ):
            raise ValueError(
                "Starting point ({}, {}, {}) outside of data "
                "boundaries".format(xs, ys, zs)
            )

    if integration_direction == "both" and maxlength is not None:
        maxlength /= 2.0

    integrate = get_integrator3(
        u, v, w, dmap, minlength, maxlength, integration_direction
    )

    # Convert start_points from data to array coords
    # Shift the seed points from the bottom left of the data so that
    # data2grid works properly.
    sp[:, 0] -= grid.x_origin
    sp[:, 1] -= grid.y_origin
    sp[:, 2] -= grid.z_origin

    for xs, ys, zs in sp:
        xg, yg, zg = dmap.data2grid(xs, ys, zs)
        t = integrate(xg, yg, zg)
        # Rescale from grid-coordinates to data-coordinates.

        if t is not None:
            tx, ty, tz = dmap.grid2data(*np.array(t))
            tx += grid.x_origin
            ty += grid.y_origin
            tz += grid.z_origin

            trajectories.append((tx, ty, tz))

    return trajectories


# Coordinate definitions
# ========================
# These follow explicitly from Matplotlib's streamplot class


class DomainMap3:
    """
    3D map representing different coordinate systems.

    Coordinate definitions:

    * axes-coordinates goes from 0 to 1 in the domain.
    * data-coordinates are specified by the input x-y coordinates.
    * grid-coordinates goes from 0 to N and 0 to M for an N x M grid,
      where N and M match the shape of the input data.
    * mask-coordinates goes from 0 to N and 0 to M for an N x M mask,
      where N and M are user-specified to control the density of streamlines.

    This class also has methods for adding trajectories to the StreamMask.
    Before adding a trajectory, run `start_trajectory` to keep track of regions
    crossed by a given trajectory. Later, if you decide the trajectory is bad
    (e.g., if the trajectory is very short) just call `undo_trajectory`.
    """

    def __init__(
        self, grid, mask, check_overlap=False
    ):  # grid, mask are Grid3, StreamMask3 objects
        self.grid = grid
        self.mask = mask
        self.check_overlap = check_overlap
        # Constants for conversion between grid- and mask-coordinates
        self.x_grid2mask = (mask.nx - 1) / (grid.nx - 1)
        self.y_grid2mask = (mask.ny - 1) / (grid.ny - 1)
        self.z_grid2mask = (mask.nz - 1) / (grid.nz - 1)

        self.x_mask2grid = 1.0 / self.x_grid2mask
        self.y_mask2grid = 1.0 / self.y_grid2mask
        self.z_mask2grid = 1.0 / self.z_grid2mask

        self.x_data2grid = 1.0 / grid.dx
        self.y_data2grid = 1.0 / grid.dy
        self.z_data2grid = 1.0 / grid.dz

    def grid2mask(self, xi, yi, zi):
        """Return nearest space in mask-coords from given grid-coords."""
        return (
            int(xi * self.x_grid2mask + 0.5),
            int(yi * self.y_grid2mask + 0.5),
            int(zi * self.z_grid2mask + 0.5),
        )

    def mask2grid(self, xm, ym, zm):
        return xm * self.x_mask2grid, ym * self.y_mask2grid, zm * self.z_mask2grid

    def data2grid(self, xd, yd, zd):
        return xd * self.x_data2grid, yd * self.y_data2grid, zd * self.z_data2grid

    def grid2data(self, xg, yg, zg):
        return xg / self.x_data2grid, yg / self.y_data2grid, zg / self.z_data2grid

    def start_trajectory(self, xg, yg, zg):
        """Takes an initial grid location to start a trajectory"""
        xm, ym, zm = self.grid2mask(xg, yg, zg)
        self.mask._start_trajectory(xm, ym, zm)

    def reset_start_point(self, xg, yg, zg):
        """Sets the current xyz point"""
        xm, ym, zm = self.grid2mask(xg, yg, zg)
        self.mask._current_xyz = (xm, ym, zm)

    def update_trajectory(self, xg, yg, zg):
        """Updates a trajectory in the mask"""
        if not self.grid.within_grid(xg, yg, zg):
            raise InvalidIndexError
        xm, ym, zm = self.grid2mask(xg, yg, zg)
        self.mask._update_trajectory(xm, ym, zm)

    def undo_trajectory(self):
        self.mask._undo_trajectory()


class Grid3:
    """Grid of 3D data."""

    def __init__(self, x, y, z):

        if x.ndim == 1:
            pass
        elif x.ndim == 3:
            x_line = x[:, 0, 0]
            if not np.allclose(x_line, np.allclose(x_line, np.transpose(x, [1, 2, 0]))):
                raise ValueError("Axes 1 and 2 of 'x' must be equal")
            x = x_line
        else:
            raise ValueError("'x' can only have 1 or 3 dimensions")

        if y.ndim == 1:
            pass
        elif y.ndim == 3:
            y_line = y[0, :, 0]
            if not np.allclose(y_line, np.allclose(y_line, np.transpose(y, [0, 2, 1]))):
                raise ValueError("Axes 1 and 3 of 'y' must be equal")
            y = y_line
        else:
            raise ValueError("'y' can only have 1 or 3 dimensions")

        if z.ndim == 1:
            pass
        elif z.ndim == 3:
            z_line = z[0, 0, :]
            if not np.allclose(z_line, z):
                raise ValueError("Axes 1 and 3 of 'z' must be equal")
            z = z_line
        else:
            raise ValueError("'z' can only have 1 or 3 dimensions")

        if not (np.diff(x) > 0).all():
            raise ValueError("'x' must be strictly increasing")
        if not (np.diff(y) > 0).all():
            raise ValueError("'y' must be strictly increasing")
        if not (np.diff(z) > 0).all():
            raise ValueError("'z' must be strictly increasing")

        self.x = x
        self.y = y
        self.z = z

        self.nx = len(x)
        self.ny = len(y)
        self.nz = len(z)

        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.dz = z[1] - z[0]

        self.x_origin = x[0]
        self.y_origin = y[0]
        self.z_origin = z[0]

        self.Lx = x[-1] - x[0]
        self.Ly = y[-1] - y[0]
        self.Lz = z[-1] - z[0]

        if not np.allclose(np.diff(x), self.Lx / (self.nx - 1)):
            raise ValueError("'x' values must be equally spaced")
        if not np.allclose(np.diff(y), self.Ly / (self.ny - 1)):
            raise ValueError("'y' values must be equally spaced")
        if not np.allclose(np.diff(z), self.Lz / (self.nz - 1)):
            raise ValueError("'z' values must be equally spaced")

    @property
    def shape(self):
        return self.nx, self.ny, self.nz

    def within_grid(self, xi, yi, zi):
        """Return whether (*xi*, *yi*, *zi*) is a valid index of the grid."""
        # Note that xi/yi can be floats; so, for example, we can't simply check
        # `xi < self.nx` since *xi* can be `self.nx - 1 < xi < self.nx`
        return (
            0 <= xi <= self.nx - 1 and 0 <= yi <= self.ny - 1 and 0 <= zi <= self.nz - 1
        )


class StreamMask3:
    """
    3D Mask to keep track of discrete regions crossed by streamlines.

    The resolution of this grid determines the approximate spacing between
    trajectories. Streamlines are only allowed to pass through zeroed cells:
    When a streamline enters a cell, that cell is set to 1, and no new
    streamlines are allowed to enter.
    """

    def __init__(self, density):
        try:
            # upped number of points in each direction from 30 to 50
            self.nx, self.ny, self.nz = (50 * np.broadcast_to(density, 3)).astype(int)
        except ValueError as err:
            raise ValueError("'density' must be a scalar or be of length " "3") from err
        if self.nx < 0 or self.ny < 0 or self.nz < 0:
            raise ValueError("'density' must be positive")
        self._mask = np.zeros((self.ny, self.nx, self.nz))
        self.shape = self._mask.shape

        self._current_xyz = None

    def __getitem__(self, args):
        return self._mask[args]

    def _start_trajectory(self, xm, ym, zm):
        """Start recording streamline trajectory"""
        self._traj = []
        self._update_trajectory(xm, ym, zm)

    def _undo_trajectory(self):
        """Remove current trajectory from mask"""
        for t in self._traj:
            self._mask[t] = 0

    def _update_trajectory(self, xm, ym, zm):
        """
        Update current trajectory position in mask.

        If the new position has already been filled, raise `InvalidIndexError`.
        """
        if self._current_xyz != (xm, ym, zm):
            if self[ym, xm, zm] == 0:
                self._traj.append((xm, ym, zm))
                self._mask[ym, xm, zm] = 1
                self._current_xyz = (xm, ym, zm)
            else:
                raise InvalidIndexError


class InvalidIndexError(Exception):
    pass


class TerminateTrajectory(Exception):
    pass


class OutOfBounds(IndexError):
    pass


# Integrator definitions
# =======================


def get_integrator3(u, v, w, dmap, minlength, maxlength, integration_direction):

    # rescale velocity onto grid-coordinates for integrations.
    u, v, w = dmap.data2grid(u, v, w)

    # speed (path length) will be in axes-coordinates
    u_ax = u / (dmap.grid.nx - 1)
    v_ax = v / (dmap.grid.ny - 1)
    w_ax = w / (dmap.grid.nz - 1)
    speed = np.ma.sqrt(u_ax**2 + v_ax**2 + w_ax**2)

    def forward_time(xi, yi, zi):
        if not dmap.grid.within_grid(xi, yi, zi):
            raise OutOfBounds
        ds_dt = interpgrid3(speed, xi, yi, zi)
        if ds_dt == 0:
            raise TerminateTrajectory()
        dt_ds = 1.0 / ds_dt
        ui = interpgrid3(u, xi, yi, zi)
        vi = interpgrid3(v, xi, yi, zi)
        wi = interpgrid3(w, xi, yi, zi)
        return ui * dt_ds, vi * dt_ds, wi * dt_ds

    def backward_time(xi, yi, zi):
        dxi, dyi, dzi = forward_time(xi, yi, zi)
        return -dxi, -dyi, -dzi

    def integrate(x0, y0, z0):
        """
        Return x, y grid-coordinates of trajectory based on starting point.

        Integrate both forward and backward in time from starting point in
        grid coordinates.

        Integration is terminated when a trajectory reaches a domain boundary
        or when it crosses into an already occupied cell in the StreamMask. The
        resulting trajectory is None if it is shorter than `minlength`.
        """

        stotal, x_traj, y_traj, z_traj = 0.0, [], [], []

        try:
            dmap.start_trajectory(x0, y0, z0)
        except InvalidIndexError:
            return None
        if integration_direction in ["both", "backward"]:
            s, xt, yt, zt = _integrate3_rk12(x0, y0, z0, dmap, backward_time, maxlength)
            stotal += s
            x_traj += xt[::-1]  # reverse, then append
            y_traj += yt[::-1]
            z_traj += zt[::-1]

        if integration_direction in ["both", "forward"]:
            dmap.reset_start_point(x0, y0, z0)
            s, xt, yt, zt = _integrate3_rk12(x0, y0, z0, dmap, forward_time, maxlength)
            if len(x_traj) > 0:
                xt = xt[1:]
                yt = yt[1:]
                zt = zt[1:]
            stotal += s
            x_traj += xt
            y_traj += yt
            z_traj += zt

        if stotal > minlength:
            return x_traj, y_traj, z_traj
        else:  # reject short trajectories
            dmap.undo_trajectory()
            return None

    return integrate


def _integrate3_rk12(x0, y0, z0, dmap, f, maxlength):
    """
    2nd-order Runge-Kutta algorithm with adaptive step size.

    This method is also referred to as the improved Euler's method, or Heun's
    method. This method is favored over higher-order methods because:

    1. To get decent looking trajectories and to sample every mask cell
       on the trajectory we need a small timestep, so a lower order
       solver doesn't hurt us unless the data is *very* high resolution.
       In fact, for cases where the user inputs
       data smaller or of similar grid size to the mask grid, the higher
       order corrections are negligible because of the very fast linear
       interpolation used in `interpgrid`.

    2. For high resolution input data (i.e. beyond the mask
       resolution), we must reduce the timestep. Therefore, an adaptive
       timestep is more suited to the problem as this would be very hard
       to judge automatically otherwise.

    This integrator is about 1.5 - 2x as fast as both the RK4 and RK45
    solvers in most setups on my machine. I would recommend removing the
    other two to keep things simple.
    """
    # This error is below that needed to match the RK4 integrator. It
    # is set for visual reasons -- too low and corners start
    # appearing ugly and jagged. Can be tuned.
    maxerror = 0.0003  # default value: 0.003

    # This limit is important (for all integrators) to avoid the
    # trajectory skipping some mask cells. We could relax this
    # condition if we use the code which is commented out below to
    # increment the location gradually. However, due to the efficient
    # nature of the interpolation, this doesn't boost speed by much
    # for quite a bit of complexity.
    maxds = min(1.0 / dmap.mask.nx, 1.0 / dmap.mask.ny, 1.0 / dmap.mask.nz, 0.1)

    ds = maxds  # initial guess for step size
    stotal = 0
    s_dim = 0  # grid length of the path
    xi = x0
    yi = y0
    zi = z0
    xf_traj = []
    yf_traj = []
    zf_traj = []

    while True:
        try:
            if dmap.grid.within_grid(xi, yi, zi):
                xf_traj.append(xi)
                yf_traj.append(yi)
                zf_traj.append(zi)
            else:
                raise OutOfBounds

            # Compute the two intermediate gradients.
            # f should raise OutOfBounds if the locations given are
            # outside the grid.
            k1x, k1y, k1z = f(xi, yi, zi)
            k2x, k2y, k2z = f(xi + ds * k1x, yi + ds * k1y, zi + ds * k1z)

        except OutOfBounds:
            # Out of the domain during this step.
            # Take an Euler step to the boundary to improve neatness
            # unless the trajectory is currently empty.
            if xf_traj:  # if not an empty trajectory
                ds, dx, dy, dz, xf_traj, yf_traj, zf_traj = _euler_step3(
                    xf_traj, yf_traj, zf_traj, dmap, f
                )
                stotal += ds
                ds_dim = np.linalg.norm(dmap.grid2data(dx, dy, dz))
                s_dim += ds_dim
            break

        except TerminateTrajectory:
            break

        dx1 = ds * k1x
        dy1 = ds * k1y
        dz1 = ds * k1z
        dx2 = ds * 0.5 * (k1x + k2x)
        dy2 = ds * 0.5 * (k1y + k2y)
        dz2 = ds * 0.5 * (k1z + k2z)

        nx, ny, nz = dmap.grid.shape
        # Error is normalized to the axes coordinates
        error = np.linalg.norm(
            ((dx2 - dx1) / (nx - 1), (dy2 - dy1) / (ny - 1), (dz2 - dz1) / (nz - 1))
        )

        # Only save step if within error tolerance
        if error < maxerror:
            xi += dx2
            yi += dy2
            zi += dz2
            ds_dim = np.linalg.norm(dmap.grid2data(dx2, dy2, dz2))
            if dmap.check_overlap:
                try:
                    dmap.update_trajectory(xi, yi, zi)
                except InvalidIndexError:
                    break
            # compare streamline length in data coordinates
            if maxlength is not None and s_dim + ds_dim > maxlength:
                break
            stotal += ds
            s_dim += ds_dim

        # recalculate stepsize based on step error
        if error == 0:
            ds = maxds
        else:
            ds = min(
                maxds, 0.85 * ds * (maxerror / error) ** 0.5
            )  # change ds based on error

    return s_dim, xf_traj, yf_traj, zf_traj  # returns path length in grid dimensions


def _euler_step3(xf_traj, yf_traj, zf_traj, dmap, f):
    """Simple Euler integration step that extends streamline to boundary."""
    nx, ny, nz = dmap.grid.shape
    xi = xf_traj[-1]  # last trajectory point
    yi = yf_traj[-1]
    zi = zf_traj[-1]

    # what is this doing?
    cx, cy, cz = f(xi, yi, zi)  # compute
    if cx == 0:
        dsx = np.inf
    elif cx < 0:
        dsx = xi / -cx
    else:
        dsx = (nx - 1 - xi) / cx
    if cy == 0:
        dsy = np.inf
    elif cy < 0:
        dsy = yi / -cy
    else:
        dsy = (ny - 1 - yi) / cy
    if cz == 0:
        dsz = np.inf
    elif cz < 0:
        dsz = zi / -cz
    else:
        dsz = (nz - 1 - zi) / cz
    ds = min(dsx, dsy, dsz)

    dx, dy, dz = (cx * ds, cy * ds, cz * ds)
    xf_traj.append(xi + dx)
    yf_traj.append(yi + dy)
    zf_traj.append(zi + dz)
    return ds, dx, dy, dz, xf_traj, yf_traj, zf_traj


# Utility functions
# ========================


def interpgrid3(a, xi, yi, zi):
    """Fast 3D, linear interpolation on an integer grid"""

    Ny, Nx, Nz = np.shape(a)
    if isinstance(xi, np.ndarray):
        x = xi.astype(int)
        y = yi.astype(int)
        z = zi.astype(int)
        # Check that xn, yn don't exceed max index
        xn = np.clip(x + 1, 0, Nx - 1)
        yn = np.clip(y + 1, 0, Ny - 1)
        zn = np.clip(z + 1, 0, Nz - 1)
    else:
        x = int(xi)
        y = int(yi)
        z = int(zi)
        # conditional is faster than clipping for integers
        if x == (Nx - 1):
            xn = x
        else:
            xn = x + 1
        if y == (Ny - 1):
            yn = y
        else:
            yn = y + 1
        if z == (Nz - 1):
            zn = z
        else:
            zn = z + 1

    xt = xi - x
    yt = yi - y
    zt = zi - z

    # trilinear interpolation:
    a000 = a[x, y, z]
    a100 = a[xn, y, z]
    a010 = a[x, yn, z]
    a001 = a[x, y, zn]
    a110 = a[xn, yn, z]
    a101 = a[xn, y, zn]
    a011 = a[x, yn, zn]
    a111 = a[xn, yn, zn]

    # stage 1 interpolation:
    a00 = a000 * (1 - xt) + a100 * xt
    a01 = a001 * (1 - xt) + a101 * xt
    a10 = a010 * (1 - xt) + a110 * xt
    a11 = a011 * (1 - xt) + a111 * xt

    # stage 2 interpolation:
    a0 = a00 * (1 - yt) + a01 * yt
    a1 = a10 * (1 - yt) + a11 * yt
    ai = a0 * (1 - zt) + a1 * zt

    if not isinstance(xi, np.ndarray):
        if np.ma.is_masked(ai):
            raise TerminateTrajectory

    return ai


def _gen_starting_points(shape):
    """
    Yield starting points for streamlines.  Maybe we want to fix this for 3D. # TODO

    Trying points on the boundary first gives higher quality streamlines.
    This algorithm starts with a point on the mask corner and spirals inward.
    This algorithm is inefficient, but fast compared to rest of streamplot.
    """
    ny, nx = shape
    xfirst = 0
    yfirst = 1
    xlast = nx - 1
    ylast = ny - 1
    x, y = 0, 0
    direction = "right"
    for i in range(nx * ny):
        yield x, y

        if direction == "right":
            x += 1
            if x >= xlast:
                xlast -= 1
                direction = "up"
        elif direction == "up":
            y += 1
            if y >= ylast:
                ylast -= 1
                direction = "left"
        elif direction == "left":
            x -= 1
            if x <= xfirst:
                xfirst += 1
                direction = "down"
        elif direction == "down":
            y -= 1
            if y <= yfirst:
                yfirst += 1
                direction = "right"

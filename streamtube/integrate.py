"""
Numerical integration for parabolic PDEs

Kirby Heck
2025 March 3
"""

import numpy as np
from scipy.interpolate import interpn
from scipy.integrate import solve_ivp
from streamtube.interpolate import (
    trilinear_interpolation,
)
from .stream3 import TerminateTrajectory


def rk4_step(t_n, u_n, dudt, dt):
    """
    Computes the next timestep of u_n given the finite difference function du/dt
    with a 4-stage, 4th order accurate Runge-Kutta method.

    Parameters
    ----------
    t_n : float
        time for time step n
    u_n : array-like
        condition at time step n
    dudt : function
        function du/dt(t, u)
    dt : float
        time step

    Returns u_(n+1)
    """
    k1 = dt * dudt(t_n, u_n)
    k2 = dt * dudt(t_n + dt / 2, u_n + k1 / 2)
    k3 = dt * dudt(t_n + dt / 2, u_n + k2 / 2)
    k4 = dt * dudt(t_n + dt, u_n + k3)

    u_n1 = u_n + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return u_n1


def EF_step(t_n, u_n, dudt, dt):
    """
    Forward Euler stepping scheme
    """
    u_n1 = u_n + dt * dudt(t_n, u_n)
    return u_n1


def integrate(u0, dudt, dt=0.1, T=[0, 1], f=rk4_step):
    """
    General integration function which calls a step function multiple times depending
    on the parabolic integration strategy.

    Parameters
    ----------
    u0 : array-like
        Initial condition of values
    dudt : function
        Evolution function du/dt(t, u, ...)
    dt : float
        Time step
    T : (2, )
        Time range
    f : function
        Integration stepper function (e.g. RK4, EF, etc.)

    Returns
    -------
    t : (Nt, ) vector
        Time vector
    u(t) : (Nt, ...) array-like
        Solution to the parabolic ODE.
    """
    t = []
    ut = []

    u_n = u0  # initial condition
    t_n = T[0]

    while True:
        ut.append(u_n)
        t.append(t_n)

        # update timestep
        t_n1 = t_n + dt
        if t_n1 > T[1]:
            break

        try:
            u_n1 = f(t_n, u_n, dudt, dt)
        except TerminateTrajectory:
            break
        except Exception as e:
            print("DEBUG: ", e)  # not sure why we are ending up here ...?
            print(
                f"type {type(e)} is TerminateTrajectory? {str(isinstance(e, TerminateTrajectory))}"
            )
            raise e

        # update:
        u_n = u_n1
        t_n = t_n1

    return np.array(t), np.array(ut)


def streamline_2way(x, y, z, u, v, w, x0=(0, 0, 0), dt=0.5, T=[0, 50], xlim=None):
    t_rev, x_rev = _streamline(x, y, z, -u, -v, -w, x0=x0, dt=dt, T=T, xlim=xlim)
    t, x = _streamline(x, y, z, u, v, w, x0=x0, dt=dt, T=T, xlim=xlim)
    # return -t_rev, x_rev, t, x
    _t = np.concatenate([np.flip(t_rev), t])
    _x = np.concatenate([np.fliplr(x_rev), x], axis=1)
    return _t, _x


def _streamline(x, y, z, u, v, w, x0=(0, 0, 0), dt=0.5, T=[0, 50], xlim=None):
    """See streamline()"""
    x0 = np.array(x0)

    def get_u(_x, _y, _z):
        if xlim is not None:
            # check x-bounds indepedently of bounds on T
            if _x > xlim.max() or _x < xlim.min():
                raise TerminateTrajectory

        try:
            _u = interpn((x, y, z), u, (_x, _y, _z))
            _v = interpn((x, y, z), v, (_x, _y, _z))
            _w = interpn((x, y, z), w, (_x, _y, _z))
        except ValueError:
            # this means we are outside of the domain, i.e., extrapolating
            raise TerminateTrajectory

        return np.squeeze(np.array([_u, _v, _w]))

    def dxdt(t, xi):
        """Get vector U = dxdt at position x (time independent; steady)"""
        return get_u(xi[0], xi[1], xi[2])

    t, xt = integrate(x0, dxdt, dt=dt, T=T)
    return t, xt.T


def streamlines(x, y, z, u, v, w, start_points, T=[0, 50], ivp_kwargs=None):
    """Vectorized integration of streamlines"""

    ivp_kwargs = dict() if ivp_kwargs is None else ivp_kwargs

    def in_bounds_vectorized(states):
        _x, _y, _z = states.T
        # Check if the states are outside the grid bounds
        in_bounds_mask = (
            (x.min() <= _x)
            & (_x <= x.max())
            & (y.min() <= _y)
            & (_y <= y.max())
            & (z.min() <= _z)
            & (_z <= z.max())
        )
        return in_bounds_mask.astype(int)

    def get_u_vectorized(_t, points_ravel):
        """
        Uses fast, vectorized trilinear interpolation to compute
        the velocity at the given points.
        """
        points_2d = points_ravel.reshape((-1, 3))
        _u = trilinear_interpolation((x, y, z), u, points_2d)
        _v = trilinear_interpolation((x, y, z), v, points_2d)
        _w = trilinear_interpolation((x, y, z), w, points_2d)
        return (
            np.squeeze([_u, _v, _w]).T * in_bounds_vectorized(points_2d)[:, None]
        ).flatten()

    sol = solve_ivp(get_u_vectorized, T, start_points.flatten(), **ivp_kwargs)
    n_points = len(start_points)
    return sol.y.reshape(
        n_points, 3, -1
    )  # rearrange into consistent format as streamline function


def streamlines_2way(x, y, z, u, v, w, start_points, T=[0, 50], ivp_kwargs=None):
    """Vectorized two-way integration of streamlines"""
    fwd = streamlines(x, y, z, u, v, w, start_points, T=T, ivp_kwargs=ivp_kwargs)
    bkwd = streamlines(x, y, z, -u, -v, -w, start_points, T=T, ivp_kwargs=ivp_kwargs)
    return np.concat((bkwd[..., ::-1], fwd), axis=-1)  # concatenate together

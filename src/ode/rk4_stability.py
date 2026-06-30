import numpy as np

# Absolute stability region of the classical 4th-order Runge-Kutta method.
#
# For the test equation  u' = lambda * u,  one RK4 step gives:
#
#   u_{n+1} = p(z) * u_n,   z = k * lambda
#
# where p(z) is the stability polynomial (Taylor expansion of e^z to 4th order):
#
#   p(z) = 1 + z + z^2/2 + z^3/6 + z^4/24
#
# The method is stable when  |p(z)| <= 1.
#
# Usage:
#   Re, Im, mask = rk4_stability_region(re_range=(-5, 2), im_range=(-4, 4), n_pts=500)
#   plt.contourf(Re, Im, mask, levels=[0.5, 1.5], colors=['grey'])
#   plt.contour( Re, Im, mask, levels=[0.5],      colors=['black'])


def rk4_stability_region(re_range=(-5, 2), im_range=(-4, 4), n_pts=500):
    """
    Evaluate the RK4 stability polynomial on a grid of complex numbers.

    Parameters
    ----------
    re_range : (re_min, re_max) — real axis bounds
    im_range : (im_min, im_max) — imaginary axis bounds
    n_pts    : number of grid points in each direction

    Returns
    -------
    Re   : (n_pts, n_pts) array — real part of z
    Im   : (n_pts, n_pts) array — imaginary part of z
    mask : (n_pts, n_pts) bool array — True where |p(z)| <= 1
    """
    re = np.linspace(re_range[0], re_range[1], n_pts)
    im = np.linspace(im_range[0], im_range[1], n_pts)

    Re, Im = np.meshgrid(re, im)
    z = Re + 1j * Im

    # stability polynomial  p(z) = e^z truncated at order 4
    p = 1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24

    mask = np.abs(p) <= 1.0

    return Re, Im, mask

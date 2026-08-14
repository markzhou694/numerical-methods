import numpy as np


# Standard grid convention used throughout the project:
#
#     N = number of intervals
#     x_i = x_left + i*h,  i = 0, ..., N
#     h = (x_right - x_left)/N
#
# Hence there are N+1 grid points x_0, ..., x_N and N-1 interior points.
#
# The differential equation is
#
#     a(x) u'' + b(x) u' + c(x) u = f(x).
#
# Both boundary conditions use the same Robin form:
#
#     alpha*u + beta*u' = gamma.
#
# Dirichlet u=value is (alpha, beta, gamma) = (1, 0, value).
# Neumann   u'=value is (alpha, beta, gamma) = (0, 1, value).
#
# Boundary-condition examples:
#
#     u(0) = 2                 -> bc_left  = (1, 0, 2)
#     u'(1) = -1               -> bc_right = (0, 1, -1)
#     2*u(0) + 3*u'(0) = 4     -> bc_left  = (2, 3, 4)
#
# Complete example: solve
#
#     -u'' + 2*u' + 3*u = sin(pi*x),  0 <= x <= 1,
#     u(0) = 0,  u'(1) = 1.
#
#     x, A, F = fd_bvp_1d(
#         N=100,
#         a=-1.0,
#         b=2.0,
#         c=3.0,
#         f=lambda x: np.sin(np.pi*x),
#         x_left=0.0,
#         x_right=1.0,
#         bc_left=(1.0, 0.0, 0.0),
#         bc_right=(0.0, 1.0, 1.0),
#     )
#     U = np.linalg.solve(A, F)


def _values_on_grid(coefficient, x):
    """Evaluate a scalar or callable coefficient on the grid x."""
    if callable(coefficient):
        values = np.asarray(coefficient(x), dtype=float)
    else:
        values = np.asarray(coefficient, dtype=float)

    if values.ndim == 0:
        values = np.full(x.shape, float(values))

    if values.shape != x.shape:
        raise ValueError("coefficient functions must return one value per grid point")

    return values


def fd_bvp_1d(
    N,
    a=1.0,
    b=0.0,
    c=0.0,
    f=0.0,
    x_left=0.0,
    x_right=1.0,
    bc_left=(1.0, 0.0, 0.0),
    bc_right=(1.0, 0.0, 0.0),
):
    """
    Build the full finite-difference system for a linear 1D BVP.

    The equation is

        a(x) u'' + b(x) u' + c(x) u = f(x)

    on x_left <= x <= x_right.  The standard grid has N intervals and
    N+1 nodes x_0, ..., x_N.

    Parameters
    ----------
    N : int
        Number of equal grid intervals.  There are N+1 grid points.
    a, b, c : scalar or callable
        Coefficients of u'', u', and u.
    f : scalar or callable
        Right-hand side.
    x_left, x_right : float
        Interval endpoints.
    bc_left, bc_right : tuple (alpha, beta, gamma)
        Robin data alpha*u + beta*u' = gamma.  Dirichlet and Neumann
        conditions are obtained by setting beta=0 or alpha=0.

    Returns
    -------
    x : (N+1,) ndarray
        Full grid x_0, ..., x_N.
    A : (N+1, N+1) ndarray
        Full system matrix.  Rows 1, ..., N-1 contain the differential
        equation; rows 0 and N contain the boundary conditions.
    F : (N+1,) ndarray
        Full right-hand side.
    """
    if not isinstance(N, (int, np.integer)) or N < 2:
        raise ValueError("N must be an integer with N >= 2")
    if x_right <= x_left:
        raise ValueError("x_right must be greater than x_left")

    h = (x_right - x_left) / N
    x = np.linspace(x_left, x_right, N + 1)

    A = np.zeros((N + 1, N + 1))
    F = np.zeros(N + 1)

    x_interior = x[1:N]
    a_i = _values_on_grid(a, x_interior)
    b_i = _values_on_grid(b, x_interior)
    c_i = _values_on_grid(c, x_interior)
    f_i = _values_on_grid(f, x_interior)

    # Interior rows: centered differences for u'' and u'.
    for i in range(1, N):
        j = i - 1
        A[i, i - 1] = a_i[j] / h**2 - b_i[j] / (2.0 * h)
        A[i, i] = -2.0 * a_i[j] / h**2 + c_i[j]
        A[i, i + 1] = a_i[j] / h**2 + b_i[j] / (2.0 * h)
        F[i] = f_i[j]

    alpha_left, beta_left, gamma_left = bc_left
    alpha_right, beta_right, gamma_right = bc_right

    # Left BC: alpha*u_0 + beta*(-3u_0 + 4u_1 - u_2)/(2h) = gamma.
    A[0, 0] = alpha_left - 3.0 * beta_left / (2.0 * h)
    A[0, 1] = 4.0 * beta_left / (2.0 * h)
    A[0, 2] = -beta_left / (2.0 * h)
    F[0] = gamma_left

    # Right BC: alpha*u_N + beta*(3u_N - 4u_{N-1} + u_{N-2})/(2h) = gamma.
    A[N, N - 2] = beta_right / (2.0 * h)
    A[N, N - 1] = -4.0 * beta_right / (2.0 * h)
    A[N, N] = alpha_right + 3.0 * beta_right / (2.0 * h)
    F[N] = gamma_right

    return x, A, F

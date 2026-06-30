import numpy as np
from scipy import sparse

# Delta u = f  on a rectangle,  Dirichlet BCs,  9-point compact stencil  (4th order)
# Row-major ordering (x-fast):  u_{k,j} -> row k*m + j
#
# A9 = 1/(6h^2) * block-tridiag,   m^2 x m^2
#
#     [ T9  Q9               ]    T9 = tridiag(-20, 4, 4),   m x m
#     [ Q9  T9  Q9           ]    Q9 = tridiag(  4, 1, 1),   m x m
#     [     Q9  T9  Q9       ]
#     [         .   .   .    ]
#     [             Q9  T9   ]
#
# 4th-order accuracy requires the modified RHS:
#   f* = f + (h^2/12) * Delta(f)
#
# Usage:
#   xi, yi, A9 = poisson9pt_matrix(m, x_range, y_range)
#   h = xi[1] - xi[0]
#   X, Y = np.meshgrid(xi, yi)
#   F9 = f(X, Y) + (h**2 / 12) * laplacian_f(X, Y) # for 4th-order accuracy
#   U  = spsolve(A9, F9.flatten()).reshape((m, m))


def poisson9pt_matrix(m, x_range=(-0.5, 0.5), y_range=(-0.5, 0.5)):
    """
    Build the 9-point compact Laplacian matrix on interior nodes.

    Parameters
    ----------
    m       : number of interior points in each direction
    x_range : (x_left, x_right)
    y_range : (y_bottom, y_top)

    Returns
    -------
    xi : (m,) array — interior x-grid
    yi : (m,) array — interior y-grid
    A9 : (m^2, m^2) sparse CSC matrix — 4th-order Laplacian / (6h^2)
         Use with modified RHS: f* = f + (h^2/12) * Delta(f)
    """
    hx = (x_range[1] - x_range[0]) / (m + 1)
    hy = (y_range[1] - y_range[0]) / (m + 1)
    assert abs(hx - hy) < 1e-14 * max(hx, hy), "x and y spacings must match"
    h = hx

    xi = x_range[0] + np.arange(1, m + 1) * h
    yi = y_range[0] + np.arange(1, m + 1) * h

    # T9: m x m diagonal block — centre stencil weights
    T9 = sparse.diags([-20.0 * np.ones(m), 4.0 * np.ones(m - 1), 4.0 * np.ones(m - 1)],
                      [0, 1, -1], format='csc')

    # Q9: m x m off-diagonal block — corner and edge stencil weights
    Q9 = sparse.diags([4.0 * np.ones(m), np.ones(m - 1), np.ones(m - 1)],
                      [0, 1, -1], format='csc')

    # A9: block-tridiagonal m^2 x m^2
    A9     = sparse.kron(sparse.eye(m), T9, format='csc')
    Q9_off = sparse.kron(sparse.diags([1.0], [1], shape=(m, m)), Q9, format='csc')
    A9     = (A9 + Q9_off + Q9_off.T) / (6.0 * h**2)

    return xi, yi, A9

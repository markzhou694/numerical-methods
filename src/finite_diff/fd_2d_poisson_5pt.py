import numpy as np
from scipy import sparse

# Delta u = f  on a rectangle,  Dirichlet BCs,  5-point stencil  (2nd order)
# Row-major ordering (x-fast):  u_{k,j} -> row k*m + j
#
# A = 1/h^2 * block-tridiag,   m^2 x m^2
#
#     [ T  I              ]        T = tridiag(-4, 1, 1),   m x m
#     [ I  T  I           ]        I = identity,            m x m
#     [    I  T  I        ]
#     [       .  .  .     ]
#     [          I  T  I  ]
#     [             I  T  ]
#
# A @ u_int = f_int   (equation is Delta u = f, not -Delta u = f)
#
# Usage:
#   xi, yi, A = poisson5pt_matrix(m, x_range, y_range)
#   X, Y = np.meshgrid(xi, yi)
#   U = spsolve(A, f(X, Y).flatten()).reshape((m, m))


def poisson5pt_matrix(m, x_range=(-0.5, 0.5), y_range=(-0.5, 0.5)):
    """
    Build the 5-point sparse Laplacian matrix on interior nodes.

    Parameters
    ----------
    m       : number of interior points in each direction
    x_range : (x_left, x_right)
    y_range : (y_bottom, y_top)

    Returns
    -------
    xi : (m,) array — interior x-grid
    yi : (m,) array — interior y-grid
    A  : (m^2, m^2) sparse CSC matrix — Laplacian discretization / h^2
         Diagonal: -4/h^2,  neighbours: +1/h^2
    """
    hx = (x_range[1] - x_range[0]) / (m + 1)
    hy = (y_range[1] - y_range[0]) / (m + 1)
    # require square cells for the standard 5-point stencil
    assert abs(hx - hy) < 1e-14 * max(hx, hy), "x and y spacings must match"
    h = hx

    xi = x_range[0] + np.arange(1, m + 1) * h
    yi = y_range[0] + np.arange(1, m + 1) * h

    # T: m x m tridiagonal (x-direction second differences)
    T = sparse.diags([-4.0 * np.ones(m), np.ones(m - 1), np.ones(m - 1)],
                     [0, 1, -1], format='csc')

    # A: block-tridiagonal m^2 x m^2
    A     = sparse.kron(sparse.eye(m), T, format='csc')
    I_off = sparse.kron(sparse.diags([1.0], [1], shape=(m, m)),
                        sparse.eye(m), format='csc')
    A = (A + I_off + I_off.T) / h**2

    return xi, yi, A

import numpy as np

# u'' + b*u' - c*u = f(x)  on [x_left, x_right],  Dirichlet BCs
# h = (x_right - x_left) / (m + 1)
#
# m (ie. m+1 intervals) is the number of interior points, so the full grid has m+2 points including endpoints.
#
# A = D_2 + b*D_1 - c*I,   m x m interior operator
#
#        D_2                          D_1
#   [ -2  1              ]       [  0  1              ]
#   [  1 -2  1           ]       [ -1  0  1           ]
#   [     1 -2  1        ] /h^2  +  [    -1  0  1     ] /(2h)  - c*I_m
#   [        .  .  .     ]       [        .  .  .     ]
#   [           1 -2  1  ]       [          -1  0  1  ]
#   [              1 -2  ]       [             -1  0  ]
#
# Usage:
#   x, A = fd_matrix_1d(m, c, b, x_left, x_right)
#   h = x[1] - x[0]
#   F = f(x[1:-1]).copy()
#   F[0]  -= u_a / h**2;  F[0]  += b * u_a / (2*h)   # left BC  u(x_left)  = u_a
#   F[-1] -= u_b / h**2;  F[-1] -= b * u_b / (2*h)   # right BC u(x_right) = u_b
#   U_int = np.linalg.solve(A, F) #or other linear solver


def fd_matrix_1d(m, c=0.0, b=0.0, x_left=-1.0, x_right=1.0):
    """
    Build the interior finite-difference operator for  u'' + b*u' - c*u = f.

    Parameters
    ----------
    m       : number of interior grid points
    c       : coefficient on u   (scalar)
    b       : coefficient on u'  (scalar);  b=0 gives no advection
    x_left  : left endpoint
    x_right : right endpoint

    Returns
    -------
    x : (m+2,) array — full grid including both endpoints
    A : (m, m) ndarray — operator  D_2 + b*D_1 - c*I
        D_2[j,j]   = -2/h^2,  D_2[j,j±1] =  1/h^2
        D_1[j,j+1] =  1/(2h), D_1[j,j-1] = -1/(2h)
    """
    h = (x_right - x_left) / (m + 1)
    x = x_left + np.arange(m + 2) * h

    # D_2: centered second-difference, m x m
    D_2 = (np.diag(-2.0 * np.ones(m))
           + np.diag(np.ones(m - 1),  1)
           + np.diag(np.ones(m - 1), -1)) / h**2

    # D_1: centered first-difference, m x m
    D_1 = (np.diag(np.ones(m - 1),  1)
           - np.diag(np.ones(m - 1), -1)) / (2.0 * h)

    A = D_2 + b * D_1 - c * np.eye(m)

    return x, A

import numpy as np
from scipy.sparse import lil_matrix

# u'' = f(x)  on [0, 1],  Neumann BCs:  u'(0) = sigma_0,  u'(1) = sigma_1
# h = 1/(m+1),  n = m+2 nodes
#
# neumann_matrix_pinned — dense, n x n,  row 0 pinned to u(0)=0  (invertible)
#
#     [ 1                              ]   <- pin u(0) = 0
#     [ 1  -2   1                      ]
#     [    1   -2   1                  ]  / h^2
#     [         .    .    .            ]
#     [              1   -2   1        ]
#     [                  1/h  -1/h     ]   <- (u_{n-1} - u_{n-2})/h = sigma_1
#
# neumann_matrix_singular — sparse, n x n,  both Neumann rows intact  (rank n-1)
#
#     [ -1/h   1/h                     ]   <- (u_1 - u_0)/h = sigma_0
#     [ 1/h^2 -2/h^2  1/h^2           ]
#     [        .       .      .        ]
#     [              1/h^2 -2/h^2  1/h^2 ]
#     [                      1/h   -1/h  ]   <- (u_{n-1} - u_{n-2})/h = sigma_1
#
# RHS trapezoidal correction (O(h) stencil + h/2*f term = O(h^2) overall):
#   F[0]  = sigma_0 + (h/2) * f(x[0])
#   F[-1] = sigma_1 + (h/2) * f(x[-1])


def neumann_matrix_pinned(m):
    """
    Dense Neumann system matrix with u(0) = 0 pinned (uniquely solvable).

    Parameters
    ----------
    m : number of interior grid points  (n = m+2 total nodes)

    Returns
    -------
    x : (m+2,) grid on [0, 1]
    A : (m+2, m+2) ndarray
        Row 0  : identity row — pins u(0) = 0
        Row n-1: backward-difference Neumann row for u'(1)
        Interior rows: centered second-difference / h^2

    Usage
    -----
    x, A = neumann_matrix_pinned(m)
    h = x[1] - x[0]
    F       = f(x).copy()
    F[0]    = 0.0                              # pinned
    F[-1]   = sigma_1 + (h / 2) * f(x[-1])   # Neumann at x = 1
    U = np.linalg.solve(A, F)
    """
    h = 1.0 / (m + 1)
    n = m + 2
    x = np.linspace(0.0, 1.0, n)

    # base: centered second-difference, n x n
    A = (np.diag(-2.0 * np.ones(n))
         + np.diag(np.ones(n - 1),  1)
         + np.diag(np.ones(n - 1), -1)) / h**2

    # Neumann row at x = 1: (u_{n-1} - u_{n-2}) / h = sigma_1
    A[-1, :]  = 0.0
    A[-1, -2] =  1.0 / h
    A[-1, -1] = -1.0 / h

    # pin u(0) = 0
    A[0, :] = 0.0
    A[0, 0] = 1.0

    return x, A


def neumann_matrix_singular(m):
    """
    Sparse Neumann system matrix with both BC rows intact (singular).

    Parameters
    ----------
    m : number of interior grid points  (n = m+2 total nodes)

    Returns
    -------
    x : (m+2,) grid on [0, 1]
    A : (m+2, m+2) sparse CSC matrix  (rank n-1, null-space = constants)
        Row 0  : forward-difference Neumann row for u'(0)
        Row n-1: backward-difference Neumann row for u'(1)
        Interior rows: centered second-difference / h^2

    Usage
    -----
    from scipy.sparse.linalg import gmres
    x, A = neumann_matrix_singular(m)
    h = x[1] - x[0]
    F       = f(x).copy()
    F[0]    = sigma_0 + (h / 2) * f(x[0])     # Neumann at x = 0
    F[-1]   = sigma_1 + (h / 2) * f(x[-1])    # Neumann at x = 1
    # restart=n gives full (unrestarted) GMRES — required for singular systems
    U, _ = gmres(A, F, restart=n, atol=1e-12, rtol=1e-12, maxiter=n)
    """
    h = 1.0 / (m + 1)
    n = m + 2
    x = np.linspace(0.0, 1.0, n)

    A = lil_matrix((n, n))

    # interior rows: centered second-difference
    for i in range(1, n - 1):
        A[i, i - 1] =  1.0 / h**2
        A[i, i]     = -2.0 / h**2
        A[i, i + 1] =  1.0 / h**2

    # Neumann at x = 0: (u_1 - u_0) / h = sigma_0
    A[0, 0] = -1.0 / h
    A[0, 1] =  1.0 / h

    # Neumann at x = 1: (u_{n-1} - u_{n-2}) / h = sigma_1
    A[-1, -2] =  1.0 / h
    A[-1, -1] = -1.0 / h

    return x, A.tocsc()

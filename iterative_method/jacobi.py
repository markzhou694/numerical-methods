import numpy as np

# Jacobi iteration for solving a linear system
#
#     A x = b,    A in R^{n x n},    x, b in R^n
#
# Split the matrix into its diagonal and off-diagonal parts:
#
#     A = D + (A - D),    D = diag(A)
#
# The Jacobi method solves the i-th equation for x[i] while holding every
# other component fixed at the previous iterate x_k: 
#
#     x_{k+1}[i] = ( b[i] - sum_{j != i} A[i,j] x_k[j] ) / A[i,i],   i = 1,...,n
#
# All components of x_{k+1} are built from the old iterate x_k and updated
# at the same time, so the whole sweep reads x_old and writes x_new.
#
# Requires A[i,i] != 0 for all i.  Converges for any x_0 when A is strictly
# diagonally dominant.
#
#
# Stopping test: stop once the residual r_k = b - A x_k has
# dropped by the factor 1/tol relative to the initial residual r_0:
#
#     ||r_k|| / ||r_0|| < tol


def jacobi_solve(A, b, x0=None, max_iter=20000, tol=1e-8):
    """
    Solve A x = b by the Jacobi iteration.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        System matrix with nonzero diagonal entries.

    b : ndarray, shape (n,)
        Right-hand side.

    x0 : ndarray, shape (n,), optional
        Initial guess.  Defaults to the zero vector.

    max_iter : int
        Maximum number of iterations.

    tol : float
        Stop when ||r_k|| / ||r_0|| < tol.

    Returns
    -------
    x : ndarray, shape (n,)
        Approximate solution.

    iters : int
        Number of iterations used.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = b.shape[0]

    if x0 is None: # default initial guess is zero vector
        x_k = np.zeros(n)
    else:
        x_k = np.asarray(x0, dtype=float).copy()

    # initial residual r_0 = b - A x_0
    r0 = b - A @ x_k 
    r0_norm = np.linalg.norm(r0) 

    for k in range(max_iter):
        # freeze the current iterate: Jacobi uses only old values
        x_old = x_k.copy()
        x_new = np.zeros(n)

        # component-wise sweep:
        #  x_new[i] = ( b[i] - sum_{j != i} A[i,j] x_old[j] ) / A[i,i]
        for i in range(n):
            s = 0.0
            for j in range(n):
                if j != i:
                    s = s + A[i, j] * x_old[j]
            x_new[i] = (b[i] - s) / A[i, i]

        x_k = x_new

        # residual r_k = b - A x_k and relative stopping test
        r_k = b - A @ x_k
        if np.linalg.norm(r_k) / r0_norm < tol:
            return x_k, k + 1

    return x_k, max_iter

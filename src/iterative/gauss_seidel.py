import numpy as np

# Gauss-Seidel iteration for solving a linear system
#
#     A x = b,    A in R^{n x n},    x, b in R^n
#
# Split A into its lower-triangular (including diagonal) and strictly
# upper-triangular parts:
#
#     A = (D + L) + U
#
# where D = diag(A), L is the strictly lower triangle, U the strictly upper.
# Gauss-Seidel solves
#
#     (D + L) x_{k+1} = b - U x_k
#
# by forward substitution.  Component-wise, the i-th update already uses the
# newly computed values x_{k+1}[j] for j < i, and the old values x_k[j]
# for j > i:
#
#     x_{k+1}[i] = ( b[i] - sum_{j < i} A[i,j] x_{k+1}[j]
#                         - sum_{j > i} A[i,j] x_k[j]   ) / A[i,i]
#
# Because the most recent values are reused immediately, the update can be
# done in place on a single vector x.
#
# Requires A[i,i] != 0 for all i.  Converges for any x_0 when A is strictly
# diagonally dominant or symmetric positive definite.
#
# Stopping test: stop once the residual r_k = b - A x_k has
# dropped by the factor 1/tol relative to the initial residual r_0:
#
#     ||r_k|| / ||r_0|| < tol


def gauss_seidel_solve(A, b, x0=None, max_iter=20000, tol=1e-8):
    """
    Solve A x = b by the Gauss-Seidel iteration.

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
        # in-place sweep over i = 0,...,n-1:
        #   for j < i  the entry x_k[j] already holds the NEW value x_{k+1}[j]
        #   for j > i  the entry x_k[j] still holds the OLD value x_k[j]
        for i in range(n):
            s_lower = 0.0
            for j in range(i):              # j < i : updated earlier this sweep
                s_lower = s_lower + A[i, j] * x_k[j]

            s_upper = 0.0
            for j in range(i + 1, n):       # j > i : still the previous iterate
                s_upper = s_upper + A[i, j] * x_k[j]

            x_k[i] = (b[i] - s_lower - s_upper) / A[i, i]

        # residual r_k = b - A x_k and relative stopping test
        r_k = b - A @ x_k
        if np.linalg.norm(r_k) / r0_norm < tol:
            return x_k, k + 1

    return x_k, max_iter

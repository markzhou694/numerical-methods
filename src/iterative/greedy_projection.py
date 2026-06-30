import numpy as np

# Greedy one-dimensional projection for  A x = b, A symmetric positive definite. 
#
# Each step is a one-dimensional projection with K = L = span{ v_{k+1} }:
#
#     x_{k+1} = x_k + alpha * v_{k+1},     (x_{k+1} - x_k) in span(v_{k+1})
#
# The Galerkin condition  r_{k+1} _|_ v_{k+1}  fixes the step length:
#
#     <r_k - alpha A v_{k+1}, v_{k+1}> = 0
#     alpha = <r_k, v_{k+1}> / <A v_{k+1}, v_{k+1}>
#
# Greedy choice of direction: pick the coordinate vector e_m corresponding to
# the largest-magnitude residual component,
#
#     m = argmax_i |r_k[i]|,     v_{k+1} = e_m.
#
# With v_{k+1} = e_m the step length collapses to
#
#     <r_k, e_m>      = r_k[m]
#     <A e_m, e_m>    = A[m, m]
#     alpha           = r_k[m] / A[m, m],
#
# and only the m-th component of x changes:  x_{k+1}[m] = x_k[m] + alpha.
#
# Stopping test: stop once the residual has dropped relative to r_0,
#
#     ||r_k|| / ||r_0|| < tol.


def greedy_projection_solve(A, b, x0=None, max_iter=20000, tol=1e-8):
    """
    Solve A x = b (A symmetric positive definite) by greedy one-dimensional
    projection along the coordinate of largest residual (Gauss-Southwell).

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Symmetric positive-definite system matrix.

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

    if x0 is None:  # default initial guess is zero vector
        x = np.zeros(n)
    else:
        x = np.asarray(x0, dtype=float).copy()

    # initial residual r_0 = b - A x_0
    r0 = b - A @ x
    r0_norm = np.linalg.norm(r0)

    if r0_norm == 0:
        return x, 0

    for k in range(max_iter):
        # residual r_k = b - A x_k
        r = b - A @ x

        # relative stopping test
        if np.linalg.norm(r) / r0_norm < tol:
            return x, k

        # greedy direction: coordinate m with the largest |r_k[i]|
        m = int(np.argmax(np.abs(r)))

        # one-dimensional projection step length
        #   alpha = <r_k, e_m> / <A e_m, e_m> = r_k[m] / A[m, m]
        alpha = r[m] / A[m, m]

        # update only the m-th component:  x_{k+1} = x_k + alpha e_m
        x[m] = x[m] + alpha

    return x, max_iter

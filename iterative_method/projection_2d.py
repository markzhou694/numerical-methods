import numpy as np

# Two-dimensional projection method for  A x = b,  A symmetric positive definite. 
# 
#
# At each step the correction is taken from the two-dimensional space
#
#     K = L = span{ r_k, A r_k },     r_k = b - A x_k.
#
# To get an orthogonal basis of K, orthogonalize A r_k against r_k:
#
#     p_k = A r_k - ( <A r_k, r_k> / ||r_k||^2 ) r_k          (so <r_k, p_k> = 0)
#
# Writing the update as  x_{k+1} = x_k + alpha1 r_k + alpha2 p_k,
# the Galerkin condition  r_{k+1} = b - A x_{k+1}  _|_  span{r_k, p_k}  gives
# the 2x2 system
#
#     [ <A r_k, r_k>   <A r_k, p_k> ] [ alpha1 ]   [ ||r_k||^2 ]
#     [ <A r_k, p_k>   <A p_k, p_k> ] [ alpha2 ] = [    0      ]
#
# (the (2,1) entry uses <A p_k, r_k> = <A r_k, p_k> because A is symmetric).
#
# Stopping test: stop once the residual has dropped relative to r_0,
#
#     ||r_k|| / ||r_0|| < tol.


def projection_2d_solve(A: np.ndarray, b: np.ndarray, x0: np.ndarray = None, max_iter: int = 1000, tol: float = 1e-8):
    """
    Solve A x = b (A symmetric positive definite) by the two-dimensional
    projection method on K = L = span{r_k, A r_k}.

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

        # A r_k
        Ar = A @ r

        # orthogonalize A r_k against r_k:
        #   p_k = A r_k - ( <A r_k, r_k> / ||r_k||^2 ) r_k
        Ar_dot_r = np.dot(Ar, r)
        r_dot_r = np.dot(r, r)
        p = Ar - (Ar_dot_r / r_dot_r) * r

        # entries of the 2x2 Galerkin system
        Ap = A @ p
        Ar_dot_p = np.dot(Ar, p)
        Ap_dot_p = np.dot(Ap, p)

        # solve  M [alpha1; alpha2] = [ ||r_k||^2 ; 0 ]
        M = np.array([[Ar_dot_r, Ar_dot_p],
                      [Ar_dot_p, Ap_dot_p]])
        rhs = np.array([r_dot_r, 0.0])

        alpha = np.linalg.solve(M, rhs) # a 2x2 system, so direct solve is fine
        alpha1 = alpha[0]
        alpha2 = alpha[1]

        # update  x_{k+1} = x_k + alpha1 r_k + alpha2 p_k
        x = x + alpha1 * r + alpha2 * p

    return x, max_iter


if __name__ == "__main__":  
   A = np.array([[4, 1], [1, 3]])
   b = np.array([1, 2])
   x, iters = projection_2d_solve(A, b) 
   print("Solution:", x)

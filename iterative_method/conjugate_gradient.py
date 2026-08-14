import numpy as np
from scipy import sparse



def conjugate_gradient(A, b, x0=None, max_iter=20000, tol=1e-8):
    """
    Solve A x = b by the conjugate gradient iteration.

    # Start with an initial guess x0 (default: the zero vector).
    # Build the Krylov subspace
    #     K_k(A, r0) = span{r0, Ar0, A²r0, ..., A^(k-1)r0}.
    # Construct A-conjugate search directions {p0, p1, ..., p(k-1)},
    # where p0 = r0 and p_i^T A p_j = 0 for i != j.
    # Update the solution:
    #     x_(k+1) = x_k + alpha_k * p_k.
    # The new residual
    #     r_(k+1) = b - A x_(k+1)
    # satisfies the Galerkin condition:
    #     r_(k+1) ⟂ K_(k+1)(A, r0).

    Parameters
    ----------
    A : ndarray or sparse matrix, shape (n, n)
        Symmetric positive definite system matrix.

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

    if sparse.issparse(A):
        A = A.astype(float, copy=False)
    else:
        A = np.asarray(A, dtype=float)

    b = np.asarray(b, dtype=float)
    n = b.shape[0]

    if x0 is None: # default initial guess is zero vector
        x_k = np.zeros(n)
    else:
        x_k = np.asarray(x0, dtype=float).copy()

    # initial residual r_0 = b - A x_0
    r_k = b - A @ x_k
    p_k = r_k.copy()
    r0_norm = np.linalg.norm(r_k)

    if r0_norm == 0:
        return x_k, 0

    for k in range(max_iter):
        Ap_k = A @ p_k 
        alpha_k = (r_k @ r_k) / (p_k @ Ap_k)
        x_k += alpha_k * p_k
        r_next = r_k - alpha_k * Ap_k  # rk+1

        if np.linalg.norm(r_next) / r0_norm < tol:
            return x_k, k + 1

        beta_k = (r_next @ r_next) / (r_k @ r_k)
        p_k = r_next + beta_k * p_k
        r_k = r_next

    return x_k, max_iter


if __name__ == "__main__":
    from time import perf_counter

    from finite_diff import fd_bvp_1d
    from scipy.sparse.linalg import cg as scipy_cg

    N = 100

    A = fd_bvp_1d(N, a=-1.0, b=0.0, c=0.0, f=0.0)[1][1:N, 1:N]
    b = np.sin(np.random.rand(N - 1)*2*np.pi)

    t0 = perf_counter()
    x, iters = conjugate_gradient(A, b, tol=1e-10)
    t1 = perf_counter()
    print(f"Conjugate gradient: {iters} iterations, time = {t1 - t0:.6f} s")    
    print(f"Residual norm = {np.linalg.norm(b - A @ x):.2e}")



    
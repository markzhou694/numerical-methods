import numpy as np

# TriMRES — a GMRES variant whose Arnoldi process is started from
#
#     v0 = r0 = b - A x0,     v1 = A v0 / ||A v0||.
# Building v1, v2, ... by the usual Arnoldi orthogonalization gives
#
#     A v_j = sum_{i=1}^{j+1} h_{i,j} v_i,     A v0 = ||A v0|| v1,
#
# so the approximate solution is sought in the basis {v0, v1, ..., v_{m-1}}:
#
#     A [v0, v1, ..., v_{m-1}] = [v1, v2, ..., vm] H,
#
# where H is m x m and UPPER TRIANGULAR (not Hessenberg) because the run was
# started at v1 = A v0 / ||A v0||.  Writing x = x0 + [v0,...,v_{m-1}] y and
# minimizing ||r_m||_2 leads to the normal equations  Vt^T Vt H y = Vt^T v0
# with Vt = [v1,...,vm].  Since Vt has orthonormal columns (Vt^T Vt = I),
#
#     H y = Vt^T v0,
#
# which is solved by a single back substitution because H is triangular.
#
# Cheap relative-residual estimate, without forming the solution (uses
# r_k = v0 - Vt Vt^T v0, r_k _|_ v1,...,v_{k}):
#
#     ||r_k|| / ||v0|| = sqrt(||v0||^2 - ||Vt^T v0||^2) / ||v0||,   Vt = [v1,...,vk].
#
# Stopping test: the cheap estimate above (res_est) is updated every Arnoldi
# step.  The true residual ||b - A x_m|| / ||v0|| is only formed — and checked
# against tol — once the cheap estimate looks converged, or periodically
# every `check_every` steps.  The best true-residual iterate seen so far is
# always kept, so a bad final Arnoldi step can never overwrite a better
# earlier one.
#
# Numerical notes:
#  * The Arnoldi orthogonalization is run TWICE (reorthogonalization).  A single
#    modified Gram-Schmidt pass loses orthogonality of v1,v2,... on ill-
#    conditioned Krylov bases; that breaks the identity Vt^T Vt = I on which both
#    the residual estimate above and the triangular solve rely, and the recovered
#    x degrades to garbage.  Two passes restore Vt^T Vt = I to machine precision,
#    so the cheap estimate is trustworthy and the solve is well-behaved.
#  * The triangular factor H is itself severely ill-conditioned (cond(H) can
#    reach ~1e11) because the search basis {v0,v1,...} is geometrically skewed.
#    Solving H y = Vt^T v0 therefore caps the attainable accuracy near
#    sqrt(eps)*cond(H): at full Krylov dimension TriMRES trails a stable GMRES
#    (Hessenberg + Givens) by a few digits.  This is intrinsic to the
#    parametrization, not a coding fault.


def trimres(A: np.ndarray, b: np.ndarray, x0: np.ndarray = None, tol: float = 1e-6, max_iter: int = None, check_every: int = 10):
    """
    Solve A x = b by TriMRES (a GMRES variant with a triangular projected
    system).

    The cheap projected residual estimate is used during the Arnoldi process,
    but the true relative residual ||b - A x|| / ||r0|| is checked
    periodically. The method returns the best true-residual iterate seen,
    not necessarily the last one.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        System matrix (need not be symmetric).
    b : ndarray, shape (n,)
        Right-hand side.
    x0 : ndarray, shape (n,), optional
        Initial guess.  Defaults to the zero vector.
    tol : float
        Stop when the relative residual ||b - A x|| / ||r0|| < tol.
    max_iter : int, optional
        Maximum Krylov dimension.  Defaults to n.
    check_every : int, optional
        How often (in Arnoldi steps) to verify the cheap residual estimate
        against the true residual, independent of what the cheap estimate says.

    Returns
    -------
    x : ndarray, shape (n,)
        Best solution found (smallest true relative residual seen).
    iters : int
        Krylov dimension m of the returned solution.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = b.shape[0]

    if max_iter is None:
        max_iter = n
    else:
        max_iter = min(max_iter, n)

    if x0 is None:
        x_base = np.zeros(n)
    else:
        x_base = np.asarray(x0, dtype=float).copy()

    # v0 = r0 = b - A x0
    v0 = b - A @ x_base
    v0_norm = np.linalg.norm(v0)

    if v0_norm == 0:
        return x_base, 0

    # v1 = A v0 / ||A v0||
    Av0 = A @ v0
    Av0_norm = np.linalg.norm(Av0)

    if Av0_norm < 1e-14:
        return x_base, 0

    v1 = Av0 / Av0_norm

    # V holds [v0, v1, v2, ...]
    V = [v0, v1]

    # H is the triangular projected matrix
    H = np.zeros((max_iter, max_iter))
    H[0, 0] = Av0_norm

    def build_solution(m):
        """
        Build x_m = x0 + [v0,...,v_{m-1}] y.
        """
        Vt = np.column_stack(V[1:m + 1])      # [v1, ..., vm]
        basis = np.column_stack(V[:m])        # [v0, ..., v_{m-1}]

        rhs_small = Vt.T @ v0
        y = np.linalg.solve(H[:m, :m], rhs_small)

        return x_base + basis @ y

    def rel_residual(x):
        """
        Relative residual: ||b - A x|| / ||r0||.
        """
        return np.linalg.norm(b - A @ x) / v0_norm

    # residual estimate for m = 1
    proj_sq = np.dot(v1, v0) ** 2
    res_est = np.sqrt(max(v0_norm**2 - proj_sq, 0.0)) / v0_norm

    m = 1

    # best true-residual solution seen so far
    x_best = x_base.copy()
    res_best = 1.0
    m_best = 0

    # Arnoldi process
    for j in range(1, max_iter):
        # If the cheap estimate is small, verify with the true residual.
        # Also check periodically so a bad final step does not overwrite a good one.
        if res_est < tol or (check_every is not None and m % check_every == 0):
            x_curr = build_solution(m)
            res_curr = rel_residual(x_curr)

            if res_curr < res_best:
                x_best = x_curr
                res_best = res_curr
                m_best = m

            if res_curr < tol:
                return x_curr, m

        # w = A v_j
        w = A @ V[j]

        # modified Gram-Schmidt, run twice for reorthogonalization
        for _ in range(2):
            for i in range(1, j + 1):
                h = np.dot(w, V[i])
                H[i - 1, j] += h
                w = w - h * V[i]

        H[j, j] = np.linalg.norm(w)

        # Happy breakdown: do not append a nearly zero vector
        if H[j, j] < 1e-14:
            break

        # next basis vector
        v_next = w / H[j, j]
        V.append(v_next)
        m = j + 1

        # update cheap residual estimate
        proj_sq += np.dot(v_next, v0) ** 2
        res_est = np.sqrt(max(v0_norm**2 - proj_sq, 0.0)) / v0_norm

    # final check
    x_final = build_solution(m)
    res_final = rel_residual(x_final)

    if res_final < res_best:
        return x_final, m

    return x_best, m_best

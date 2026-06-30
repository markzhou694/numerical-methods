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
# Residual norm without forming the solution (uses r_k = v0 - Vt Vt^T v0,
# r_k _|_ v1,...,v_{k}):
#
#     ||r_k||^2 = ||v0||^2 - ||Vt^T v0||^2,     Vt = [v1, ..., vk].
#
# Stopping test: stop once ||r_k|| < tol (or the Arnoldi vector vanishes).


def trimres(A, b, x0=None, tol=1e-6, max_iter=1000):
    """
    Solve A x = b by TriMRES (a GMRES variant with a triangular projected
    system).

    Parameters
    ----------
    A : ndarray, shape (n, n)
        System matrix (need not be symmetric).

    b : ndarray, shape (n,)
        Right-hand side.

    x0 : ndarray, shape (n,), optional
        Initial guess.  Defaults to the zero vector.

    tol : float
        Stop when the estimated residual norm ||r_k|| < tol.

    max_iter : int
        Maximum Krylov dimension.

    Returns
    -------
    x : ndarray, shape (n,)
        Approximate solution.

    iters : int
        Number of Arnoldi steps used (the dimension m of the basis).
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = b.shape[0]

    if x0 is None:  # default initial guess is zero vector
        x = np.zeros(n)
    else:
        x = np.asarray(x0, dtype=float).copy()

    # v0 = r0 = b - A x0
    v0 = b - A @ x
    v0_sq = np.dot(v0, v0)

    # first Arnoldi vector:  v1 = A v0 / ||A v0||,   H[0,0] = ||A v0||
    Av0 = A @ v0
    Av0_norm = np.linalg.norm(Av0)
    if Av0_norm < 1e-14:
        return x, 0
    v1 = Av0 / Av0_norm

    # V holds [v0, v1, v2, ...];  H is the upper-triangular coefficient matrix
    V = [v0, v1]
    H = np.zeros((max_iter + 1, max_iter))
    H[0, 0] = Av0_norm

    # Arnoldi: build v_2, v_3, ... by orthogonalizing A v_j against v_1,...,v_j
    for j in range(1, max_iter):
        # w = A v_j
        w = A @ V[j]

        # modified Gram-Schmidt against v_1, ..., v_j
        #   h_{i,j} = <w, v_i>,   w <- w - h_{i,j} v_i
        for i in range(1, j + 1):
            H[i - 1, j] = np.dot(w, V[i])
            w = w - H[i - 1, j] * V[i]

        # diagonal entry  h_{j+1,j} = ||w||  (kept on the diagonal: H is triangular)
        H[j, j] = np.linalg.norm(w)

        # residual norm without forming x:
        #   ||r_j||^2 = ||v0||^2 - ||Vt^T v0||^2,   Vt = [v1, ..., vj]
        Vt = np.column_stack(V[1:j + 1])
        proj_sq = np.linalg.norm(Vt.T @ v0) ** 2
        res_norm = np.sqrt(max(v0_sq - proj_sq, 0.0))

        if res_norm < tol or H[j, j] < 1e-14:
            break

        # next basis vector  v_{j+1} = w / h_{j+1,j}
        V.append(w / H[j, j])

    # solve the m x m triangular system  H y = Vt^T v0  (single back substitution)
    m = len(V) - 1
    Vt = np.column_stack(V[1:m + 1])         # [v1, ..., vm]
    y = np.linalg.solve(H[:m, :m], Vt.T @ v0)

    # x = x0 + [v0, v1, ..., v_{m-1}] y
    basis = np.column_stack(V[:m])           # [v0, ..., v_{m-1}]
    x = x + basis @ y

    return x, m

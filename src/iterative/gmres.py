import numpy as np


def gmres(A, b, x0=None, max_iter=1000, tol=1e-6, restart=100):
    """
    Solve A x = b using restarted GMRES.

    Notation:
        r0        = initial residual
        beta      = ||r||
        V         = [v_1, ..., v_{m+1}]
        Hbar      = (m+1) x m upper Hessenberg matrix
        x_restart = starting point of the current restart cycle

    At step j, GMRES solves

        min_y || beta e_1 - Hbar_j y ||_2

    and updates

        x_j = x_restart + V_j y.
    """

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    n = b.shape[0]

    if x0 is None:
        x = np.zeros(n)
    else:
        x = np.asarray(x0, dtype=float).copy()

    if restart is None:
        restart = max_iter

    r_initial = b - A @ x
    r_initial_norm = np.linalg.norm(r_initial)

    if r_initial_norm == 0:
        return x, 0

    total_iters = 0

    while total_iters < max_iter:
        # Start one restart cycle
        x_restart = x.copy()

        r = b - A @ x_restart
        beta = np.linalg.norm(r)

        if beta / r_initial_norm < tol:
            return x_restart, total_iters

        # m is the Krylov dimension in this restart cycle
        m = min(restart, max_iter - total_iters)

        # V stores v_1, ..., v_{m+1}
        V = np.zeros((n, m + 1))

        # Hbar stores the (m+1) x m Arnoldi Hessenberg matrix
        Hbar = np.zeros((m + 1, m))

        V[:, 0] = r / beta

        for j in range(m):
            # Arnoldi step: w = A v_j
            w = A @ V[:, j]

            # Modified Gram-Schmidt
            for i in range(j + 1):
                Hbar[i, j] = np.dot(V[:, i], w)
                w -= Hbar[i, j] * V[:, i]

            Hbar[j + 1, j] = np.linalg.norm(w)

            # Happy breakdown
            if Hbar[j + 1, j] > 1e-12:
                V[:, j + 1] = w / Hbar[j + 1, j]

            # Current small least-squares problem:
            # min_y || beta e_1 - Hbar_j y ||
            e1 = np.zeros(j + 2)
            e1[0] = 1.0

            Hbar_j = Hbar[:j + 2, :j + 1]
            y = np.linalg.lstsq(Hbar_j, beta * e1, rcond=None)[0]

            # Current approximation:
            # x_j = x_restart + V_j y
            V_j = V[:, :j + 1]
            x = x_restart + V_j @ y

            rel_res = np.linalg.norm(b - A @ x) / r_initial_norm

            total_iters += 1

            if rel_res < tol:
                return x, total_iters

            if Hbar[j + 1, j] <= 1e-12:
                return x, total_iters

    return x, total_iters


# test on a larger matrix
np.random.seed(0)

n = 500

A = np.random.randn(n, n)
A = A + n * np.eye(n)

x_exact = np.ones(n)
b = A @ x_exact

x0 = np.zeros(n)

x, iters = gmres(A, b, x0, max_iter=500, tol=1e-8, restart=20)

print("Number of iterations:", iters)
print("Relative error:", np.linalg.norm(x - x_exact) / np.linalg.norm(x_exact))
print("Relative residual:", np.linalg.norm(b - A @ x) / np.linalg.norm(b))
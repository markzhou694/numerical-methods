import numpy as np

# Newton iteration for solving a nonlinear system 
#
#     F(theta) = 0
#
# where
#
#     theta in R^m
#     F(theta) in R^m
#     J(theta) = dF/dtheta in R^{m x m} # Jacobian matrix
#
# theta is the vector of unknowns:
#
#     theta = [theta_1, theta_2, ..., theta_m]^T
#
# F(theta) is the residual vector:
#
#     F(theta) = [F_1(theta), F_2(theta), ..., F_m(theta)]^T
#
# Derivation:
# Around the current iterate theta_k, write
#
#     s = theta - theta_k
#
# so that
#
#     theta = theta_k + s
#
# By first-order Taylor expansion,
#
#     F(theta_k + s) ≈ F(theta_k) + J(theta_k) s
#
# We want F(theta_k + s) ≈ 0, so
#
#     F(theta_k) + J(theta_k) s = 0
#
# Hence
#
#     J(theta_k) s = -F(theta_k)
#
# Equivalently, define
#
#     delta_k = -s
#
# Then
#
#     J(theta_k) delta_k = F(theta_k)
#
# and update
#
#     theta_{k+1} = theta_k - delta_k
#
# With damping(not required for convergence, but can help with robustness):
#
#     theta_{k+1} = theta_k - damping * delta_k
#
# If damping = 1, this is standard Newton.
# If 0 < damping < 1, this is damped Newton.


def newton_solve(F_func, J_func, theta0,
                 max_iter=2000, tol=1e-8, damping=0.5,
                 linear_solver=np.linalg.solve):
    """
    Solve F(theta) = 0 by damped Newton iteration.

    Parameters
    ----------
    F_func : callable
        Returns F(theta), shape (m,).

    J_func : callable
        Returns J(theta), shape (m, m).

    theta0 : ndarray, shape (m,)
        Initial guess.

    max_iter : int
        Maximum number of iterations.

    tol : float
        Stop when ||delta|| < tol.

    damping : float
        Step size factor in (0, 1].

    linear_solver : callable, optional
        Solves the Newton system  J delta = F.
        Contract: linear_solver(A, b) -> x  (returns the solution vector).
        Default np.linalg.solve (direct dense solve). To use an iterative
        solver from src/iterative/, wrap it so it returns only x, e.g.
            from iterative.gmres import gmres
            linear_solver=lambda A, b: gmres(A, b)[0]
        (jacobi_solve / gauss_seidel_solve / gmres return (x, iters),
         so the [0] picks out x to satisfy the contract.)

    Returns
    -------
    theta : ndarray, shape (m,)
        Approximate solution.

    iters : int
        Number of iterations used.
    """
    theta = theta0.copy()

    for k in range(max_iter):
        F = F_func(theta)
        J = J_func(theta)

        # solve the Newton system  J(theta_k) delta_k = F(theta_k)
        delta = linear_solver(J, F)

        theta = theta - damping * delta

        if np.linalg.norm(delta) < tol:
            return theta, k + 1

    return theta, max_iter
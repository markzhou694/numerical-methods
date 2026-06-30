import numpy as np

from iterative.newton import newton_solve 
from iterative.gmres import gmres
from iterative.trimres import trimres

# Backward Euler for
#
#     U' = f(t, U)
#
# Implicit update:
#
#     U_{n+1} = U_n + k*f(t_{n+1}, U_{n+1})
#
# Linear case:
#
#     U' = J*U + g(t)
#
# Then
#
#     U_{n+1} = U_n + k*(J*U_{n+1} + g(t_{n+1}))
#
# so
#
#     (I - kJ) U_{n+1} = U_n + k*g(t_{n+1})
#
# Nonlinear case:
#
# Define
#
#     G(V) = V - U_n - k*f(t_n+k, V) 

# where V = U_{n+1}. Then Backward Euler requires
#
#     G(V) = 0
#
# The Jacobian is
#
#     dG(V) = I - k*Jf(t_n+k, V)
#
# We solve this nonlinear system using Newton's method.


def backward_euler_matrix(J, k):
    """
    Build the Backward Euler matrix

        M = I - k*J

    for the linear system

        U' = J*U + g(t).

    Parameters
    ----------
    J : ndarray, shape (n, n)
        Constant matrix.

    k : float
        Time step.

    Returns
    -------
    M : ndarray, shape (n, n)
        Backward Euler matrix.
    """
    n = J.shape[0]
    M = np.eye(n) - k * J
    return M


def backward_euler_step_linear(J, g, t, U, k, linear_solver=np.linalg.solve):
    """
    One Backward Euler step for

        U' = J*U + g(t).

    Solves

        (I - kJ) U_{n+1} = U_n + k*g(t+k).

    Parameters
    ----------
    J : ndarray, shape (n, n)
        Constant matrix.
    g : callable
        g(t) returns array of shape (n,).
    t : float
        Current time.
    U : ndarray, shape (n,)
        Current state.
    k : float
        Time step.
    linear_solver : callable, optional
        Solves (I - kJ) U_new = rhs.  Contract: linear_solver(A, b) -> x.
        Default np.linalg.solve. Pass e.g. lambda A, b: gmres(A, b)[0]
        to use an iterative solver from src/iterative/.

    Returns
    -------
    U_new : ndarray, shape (n,)
        State at t + k.
    """
    M = backward_euler_matrix(J, k)
    rhs = U + k * g(t + k)

    # solve (I - kJ) U_{n+1} = U_n + k g(t+k)
    U_new = linear_solver(M, rhs)
    return U_new


def backward_euler_solve_linear(J, g, t0, U0, t_final, k,
                                linear_solver=np.linalg.solve):
    """
    Solve

        U' = J*U + g(t)

    from t0 to t_final using Backward Euler.

    Parameters
    ----------
    J : ndarray, shape (n, n)
        Constant matrix.
    g : callable
        g(t) returns array of shape (n,).
    t0 : float
        Initial time.
    U0 : ndarray, shape (n,)
        Initial condition.
    t_final : float
        Final time.
    k : float
        Time step.
    linear_solver : callable, optional
        Linear solver for each implicit step. Contract: linear_solver(A, b) -> x.
        Default np.linalg.solve.

    Returns
    -------
    t_values : ndarray
        Time values.
    U_values : ndarray
        Solution values.
    """
    t = t0
    U = U0.copy()

    t_values = [t]
    U_values = [U.copy()]

    while t < t_final:
        if t + k > t_final:
            k = t_final - t

        U = backward_euler_step_linear(J, g, t, U, k,
                                       linear_solver=linear_solver)
        t += k

        t_values.append(t)
        U_values.append(U.copy())

    return np.array(t_values), np.array(U_values)


def backward_euler_step_newton(f, Jf, t, U, k, tol=1e-10, max_iter=50,
                               linear_solver=np.linalg.solve):
    """
    One nonlinear Backward Euler step.

    Solves
        G(V) = V - U - k*f(t+k, V) = 0
    for
        V = U_{n+1}
    using newton_solve.

    Parameters
    ----------
    f : callable
        f(t, U) returns array of shape (n,).
    Jf : callable
        Jf(t, U) returns Jacobian matrix of shape (n, n).
    t : float
        Current time.
    U : ndarray, shape (n,)
        Current state.
    k : float
        Time step.
    tol : float
        Newton tolerance.
    max_iter : int
        Maximum Newton iterations.
    linear_solver : callable, optional
        Linear solver for the Newton system inside each step.
        Contract: linear_solver(A, b) -> x.  Default np.linalg.solve.

    Returns
    -------
    V : ndarray, shape (n,)
        State at t + k.
    """
    n = len(U)
    I = np.eye(n)

    # Explicit Euler predictor as initial guess
    V0 = U + k * f(t, U)

    V, _ = newton_solve(
        lambda V: V - U - k * f(t + k, V),
        lambda V: I - k * Jf(t + k, V),
        V0,
        tol=tol,
        max_iter=max_iter,
        linear_solver=linear_solver
    )

    return V


def backward_euler_solve_newton(f, Jf, t0, U0, t_final, k,
                                tol=1e-10, max_iter=50,
                                linear_solver=np.linalg.solve):
    """
    Solve

        U' = f(t, U)

    from t0 to t_final using Backward Euler.

    Each implicit step is solved by Newton iteration.

    Parameters
    ----------
    f : callable
        f(t, U) returns array of shape (n,).

    Jf : callable
        Jf(t, U) returns Jacobian matrix of shape (n, n).

    t0 : float
        Initial time.

    U0 : ndarray, shape (n,)
        Initial condition.

    t_final : float
        Final time.

    k : float
        Time step.

    tol : float
        Newton tolerance.

    max_iter : int
        Maximum Newton iterations per time step.

    linear_solver : callable, optional
        Linear solver for the Newton system inside each step.
        Contract: linear_solver(A, b) -> x.  Default np.linalg.solve.

    Returns
    -------
    t_values : ndarray
        Time values.

    U_values : ndarray
        Solution values.
    """
    t = t0
    U = U0.copy()

    t_values = [t]
    U_values = [U.copy()]

    while t < t_final:
        if t + k > t_final:
            k = t_final - t

        U = backward_euler_step_newton(
            f, Jf, t, U, k,
            tol=tol,
            max_iter=max_iter,
            linear_solver=linear_solver
        )

        t += k

        t_values.append(t)
        U_values.append(U.copy())

    return np.array(t_values), np.array(U_values)



if __name__ == "__main__":
    # Example: nonlinear coupled 2D system
    #
    #     u' = -2u + v + exp(-t) u(1-u)
    #     v' = -u - 3v + sin(t)
    #
    # This tests the nonlinear Backward Euler solver with Newton iteration.

    def f(t, U):
        u = U[0]
        v = U[1]

        du = -2.0 * u + v + np.exp(-t) * u * (1.0 - u)
        dv = -u - 3.0 * v + np.sin(t)

        return np.array([du, dv])


    def Jf(t, U):
        u = U[0]

        return np.array([
            [-2.0 + np.exp(-t) * (1.0 - 2.0 * u), 1.0],
            [-1.0, -3.0]
        ])


    t0 = 0.0
    t_final = 10.0
    k = 0.01

    U0 = np.array([0.5, -0.2])

    t_values, U_values = backward_euler_solve_newton(
        f, Jf, t0, U0, t_final, k,
        tol=1e-10,
        max_iter=50,
        # gmres returns (x, iters); wrap it so it returns only x  (contract: solver(A,b)->x)
        linear_solver=lambda A, b: gmres(A, b, tol=1e-10)[0]
    )

    print("Backward Euler nonlinear test")
    print("final t =", t_values[-1])
    print("final U =", U_values[-1])
    print("number of time steps =", len(t_values) - 1)

    # Also test the linear solver
    #
    #     U' = J U + g(t)

    J = np.array([
        [-2.0, 1.0],
        [-1.0, -3.0]
    ])

    def g(t):
        return np.array([np.exp(-t), np.sin(t)])

    U0_linear = np.array([0.5, -0.2])

    t_lin, U_lin = backward_euler_solve_linear(
        J, g, t0, U0_linear, t_final, k
    )

    print()
    print("Backward Euler linear test")
    print("final t =", t_lin[-1])
    print("final U =", U_lin[-1])
    print("number of time steps =", len(t_lin) - 1)
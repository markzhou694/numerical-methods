import numpy as np

# Forward Euler for  u' = f(t, u)
#
# Explicit update:
#
#   U_{n+1} = U_n + k * f(t, U_n)
#
# One step evaluates f once at the current state — O(k) local truncation error.


def forward_euler_step(f, t, U, k):
    """
    One forward Euler step.

    Parameters
    ----------
    f : callable  f(t, u) -> array, shape (n,)
    t : current time (scalar)
    U : current state, shape (n,)
    k : time step (scalar)

    Returns
    -------
    U_new : shape (n,) — state at t + k
    """
    U_new = U + k * f(t, U)
    return U_new

def forward_euler_solve(f, t0, U0, t_final, N):
    """
    Solve u' = f(t, u) from t_0 to t_N with N equal time intervals.

    Parameters
    ----------
    f       : callable  f(t, u) -> array, shape (n,)
    t0      : initial time (scalar)
    U0      : initial state, shape (n,)
    t_final : final time t_N
    N       : number of equal time intervals

    Returns
    -------
    t_values : shape (N+1,)  time grid t_0, ..., t_N
    U_values : shape (N+1, n)  state values at each time
    """
    if not isinstance(N, (int, np.integer)) or N < 1:
        raise ValueError("N must be an integer with N >= 1")

    state_dimension = len(U0)
    t_values = np.linspace(t0, t_final, N + 1)
    U_values = np.zeros((N + 1, state_dimension))
    U_values[0] = U0
    k = (t_final - t0) / N

    for i in range(N):
        U_values[i + 1] = forward_euler_step(f, t_values[i], U_values[i], k)

    return t_values, U_values

if __name__ == "__main__":
    # Smoke test on  u' = -2u,  exact solution  u(t) = exp(-2t)
    # plotting grid: N_plot intervals and N_plot+1 points
    import matplotlib.pyplot as plt

    def f_test(t, u):
        return -2.0 * u

    U = np.array([1.0])
    t_values, U_values = forward_euler_solve(f_test, 0.0, U, 5.0, N=500)
    plt.plot(t_values, U_values, label="forward Euler")
    N_plot = 100
    t_exact = np.linspace(0, 5, N_plot + 1)
    u_exact = np.exp(-2 * t_exact)
    plt.plot(t_exact, u_exact, label="exact solution", linestyle="dashed")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title("Forward Euler method")
    plt.show()

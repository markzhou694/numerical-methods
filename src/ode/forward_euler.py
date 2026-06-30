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

def forward_euler_solve(f, t0, U0, k, n_steps):
    """
    Solve u' = f(t, u) from t0 to t0 + n_steps*k with forward Euler.

    Parameters
    ----------
    f       : callable  f(t, u) -> array, shape (n,)
    t0      : initial time (scalar)
    U0      : initial state, shape (n,)
    k       : time step (scalar)
    n_steps : number of steps to take

    Returns
    -------
    t_values : shape (n_steps+1,)  time values at each step
    U_values : shape (n_steps+1, n)  state values at each step
    """
    n = len(U0)
    t_values = np.zeros(n_steps + 1)
    U_values = np.zeros((n_steps + 1, n))
    t_values[0] = t0
    U_values[0] = U0

    for i in range(n_steps):
        t_values[i + 1] = t_values[i] + k
        U_values[i + 1] = forward_euler_step(f, t_values[i], U_values[i], k)

    return  np.array(t_values),  np.array(U_values)

if __name__ == "__main__":
    # Smoke test on  u' = -2u,  exact solution  u(t) = exp(-2t)
    # plotting:  t = np.linspace(0, 5, 100),  u = np.exp(-2*t)
    import matplotlib.pyplot as plt

    def f_test(t, u):
        return -2.0 * u

    U = np.array([1.0])
    k = 0.01
    t = 0.0

    t_values, U_values = forward_euler_solve(f_test, t, U, k, n_steps=500)
    plt.plot(t_values, U_values, label="forward Euler")
    t_exact = np.linspace(0, 5, 100)
    u_exact = np.exp(-2 * t_exact)
    plt.plot(t_exact, u_exact, label="exact solution", linestyle="dashed")
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.title("Forward Euler method")
    plt.show()


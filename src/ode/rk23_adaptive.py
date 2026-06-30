import numpy as np

# Bogacki-Shampine RK23 — adaptive step via embedded 2nd-order estimate
#
# Butcher tableau (both rows):
#
#   0   |
#   1/2 |  1/2
#   3/4 |  0     3/4
#   1   |  2/9   1/3   4/9       <- 3rd-order solution U_new (Y4 reuses this stage)
#   ----+----------------------------------
#   3rd |  2/9   1/3   4/9
#   2nd |  7/24  1/4   1/3   1/8
#
#  Local error estimate:  e = || U_new - Z ||
# Since e ~ Ck^3, with the fact that the method is 3rd-order accurate, then to achieve Err <= tol, we need
# We suppose that an improved/mild step size k_new, satisfying Ck_new^3 = tol, thus by
# by dividing the two equations, we get (k_new / k)^3 = tol / e, which leads to the formula for the next step size:
#
#   k_new = k * (tol / e)^(1/3)
#
# Next step size:        k_new = k * (tol / e)^(1/3) * safety # safety factor to prevent aggressive step size changes
#
# Usage:
#   U_new, Z, k_new = rk23_adaptive_step(f, t, U, k, tol)
#   if accept (e <= tol):
#       t += k;  U = U_new;  k = k_new
#   else:
#       k = k_new   # reject and retry with smaller k


def rk23_adaptive_step(f, t, U, k, tol, safety =0.9):
    """
    One adaptive step of the Bogacki-Shampine RK23 scheme.

    Computes the 3rd-order solution U_new and the embedded 2nd-order
    solution Z, then proposes a new step size based on the local error.

    Parameters
    ----------
    f   : callable  f(t, U) -> array, same shape as U
    t   : current time (scalar)
    U   : current state (array)
    k   : current step size (scalar)
    tol : local error tolerance (scalar) at each step
    safety : safety factor for step size adjustment (scalar)
    Returns
    -------
    U_new : 3rd-order state at t + k
    Z     : 2nd-order (embedded) state at t + k
    k_new : suggested next step size
    """
    

    Y1 = f(t,           U)
    Y2 = f(t + k/2,     U + (k/2)   * Y1)
    Y3 = f(t + 3*k/4,   U + (3*k/4) * Y2)
            
    # 3rd-order update
    U_new = U + k * (2/9 * Y1 + 1/3 * Y2 + 4/9 * Y3)

    # 4th stage — evaluated at the 3rd-order solution (FSAL property)
    Y4 = f(t + k, U_new)

    # embedded 2nd-order update
    Z = U + k * (7/24 * Y1 + 1/4 * Y2 + 1/3 * Y3 + 1/8 * Y4)

    # local error estimate and new step size
    error = np.linalg.norm(U_new - Z)
    k_new = safety * k * (tol / error) ** (1/3)

    return U_new, Z, k_new

# full rk23 method would involve a loop that calls rk23_adaptive_step, checks the error,
#  and updates t, U, and k accordingly until the final time is reached.
def rk23_solve(f, t0, U0, t_final, k_initial, tol, safety=0.9):
    """
    Solve the ODE u' = f(t, u) from t0 to t_final with initial condition U0
    using the adaptive RK23 method.

    Parameters
    ----------
    f       : callable  f(t, U) -> array, same shape as U
    t0      : initial time (scalar)
    U0      : initial state (array)
    t_final : final time (scalar)
    k_initial : initial step size (scalar)
    tol     : local error tolerance (scalar)

    Returns
    -------
    t_values : list of time points where the solution was computed
    U_values : list of solution states corresponding to t_values
    """
    t = t0
    U = U0
    k = k_initial

    t_values = [t]
    U_values = [U]

    while t < t_final:
        if t + k > t_final:
            k = t_final - t

        U_new, Z, k_new = rk23_adaptive_step(f, t, U, k, tol, safety)

        error = np.linalg.norm(U_new - Z)

        if error <= tol:
            t += k
            U = U_new

            t_values.append(t)
            U_values.append(U.copy())

        k = k_new

    return np.array(t_values), np.array(U_values)

#test and plot
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test on the linear ODE u' = -2u with exact solution u(t) = exp(-2t)
    def f(t, u):
        return -2 * u

    t0 = 0
    U0 = np.array([1.0])  # initial condition u(0) = 1
    t_final = 5
    k_initial = 0.4
    tol = 1e-11

    t_values, U_values = rk23_solve(f, t0, U0, t_final, k_initial, tol)

    # Plot the results
    plt.plot(t_values, U_values[:, 0], 'o-', label='RK23 Solution')
    plt.plot(t_values, np.exp(-2 * t_values), 'r--', label='Exact Solution')
    plt.xlabel('Time')
    plt.ylabel('u(t)')
    plt.title('Adaptive RK23 Method')
    plt.legend()
    plt.grid()
    plt.show()
import numpy as np

from interpolation.lagrange import lagrange_eval

# Newton-Cotes quadrature on [a, b] with N intervals and N+1 nodes.
#
# Nodes t_i = a + i*(b-a)/N,  i = 0, ..., N.  The rule is:
#
#     I_hat(f) = (b-a) * sum_{i=0}^{n} w_i * f(t_i)
#
# where weights are defined by integrating the Lagrange cardinal bases:
#
#     w_i = 1/(b-a) * integral_a^b L_i(t) dt
#
# This makes the rule exact for all polynomials of degree <= N.
#
# Special cases:
#   N=1 (trapezoidal): w = [1/2, 1/2]
#   N=2 (Simpson):     w = [1/6, 4/6, 1/6]
#
# Usage:
#   t, w = newton_cotes_weights(2)           # Simpson weights on [0,1]
#   I    = newton_cotes_integrate(f, a, b, 2)


def newton_cotes_weights(N: int):
    """
    Compute Newton-Cotes weights for N intervals and N+1 nodes on [0, 1].

    Each weight w_i = integral_0^1 L_i(t) dt is computed by evaluating
    the i-th Lagrange cardinal basis on a fine grid (via lagrange_eval)
    and integrating with the trapezoidal rule.

    Parameters
    ----------
    N : int — number of intervals; also the interpolating degree

    Returns
    -------
    t : (N+1,) array — nodes t_i = i/N on [0, 1]
    w : (N+1,) array — weights w_i, sum(w) = 1
    """
    if not isinstance(N, (int, np.integer)) or N < 1:
        raise ValueError("N must be an integer with N >= 1")

    t_nodes = np.linspace(0.0, 1.0, N + 1)

    # fine grid for numerical integration of each L_i
    N_fine = 2000
    t_fine = np.linspace(0.0, 1.0, N_fine + 1)

    w = np.zeros(N + 1)
    for i in range(N + 1):
        # i-th standard basis vector -> lagrange_eval gives L_i(t_fine)
        e_i    = np.zeros(N + 1)
        e_i[i] = 1.0
        L_i    = lagrange_eval(t_fine, t_nodes, e_i)   # L_i on fine grid
        w[i]   = np.trapezoid(L_i, t_fine)                 # w_i = integral_0^1 L_i dt

    return t_nodes, w


def newton_cotes_integrate(f, a: float, b: float, N: int):
    """
    Integrate f on [a, b] with N intervals and N+1 Newton-Cotes nodes.

    Parameters
    ----------
    f    : callable  f(x) -> scalar
    a, b : float     endpoints
    N    : int       number of intervals (1=trapezoidal, 2=Simpson, ...)

    Returns
    -------
    I_hat : float — estimate  (b-a) * sum_i w_i * f(t_i)
    """
    t_unit, w = newton_cotes_weights(N)
    t_phys = a + t_unit * (b - a)

    # I_hat = (b-a) * sum_i w_i * f(t_i)
    I_hat = (b - a) * np.dot(w, f(t_phys))
    return I_hat

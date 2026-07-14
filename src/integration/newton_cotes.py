import numpy as np

from interpolation.lagrange import lagrange_eval

# Newton-Cotes quadrature on [a, b] with n+1 equispaced nodes.
#
# Nodes t_i = a + i*(b-a)/n,  i = 0, ..., n.  The rule is:
#
#     I_hat(f) = (b-a) * sum_{i=0}^{n} w_i * f(t_i)
#
# where weights are defined by integrating the Lagrange cardinal bases:
#
#     w_i = 1/(b-a) * integral_a^b L_i(t) dt
#
# This makes the rule exact for all polynomials of degree <= n.
#
# Special cases:
#   n=0 (midpoint):    w = [1]
#   n=1 (trapezoidal): w = [1/2, 1/2]
#   n=2 (Simpson):     w = [1/6, 4/6, 1/6]
#
# Usage:
#   t, w = newton_cotes_weights(2)           # Simpson weights on [0,1]
#   I    = newton_cotes_integrate(f, a, b, 2)


def newton_cotes_weights(n):
    """
    Compute Newton-Cotes weights for n+1 equispaced nodes on [0, 1].

    Each weight w_i = integral_0^1 L_i(t) dt is computed by evaluating
    the i-th Lagrange cardinal basis on a fine grid (via lagrange_eval)
    and integrating with the trapezoidal rule.

    Parameters
    ----------
    n : int — polynomial degree (n+1 nodes)

    Returns
    -------
    t : (n+1,) array — nodes t_i = i/n on [0, 1]
    w : (n+1,) array — weights w_i,  sum(w) = 1
    """
    if n == 0:
        return np.array([0.0]), np.array([1.0])

    t_nodes = np.arange(n + 1) / n           # t_i = i/n on [0,1]

    # fine grid for numerical integration of each L_i
    t_fine  = np.linspace(0.0, 1.0, 2000)

    w = np.zeros(n + 1)
    for i in range(n + 1):
        # i-th standard basis vector -> lagrange_eval gives L_i(t_fine)
        e_i    = np.zeros(n + 1)
        e_i[i] = 1.0
        L_i    = lagrange_eval(t_fine, t_nodes, e_i)   # L_i on fine grid
        w[i]   = np.trapezoid(L_i, t_fine)                 # w_i = integral_0^1 L_i dt

    return t_nodes, w


def newton_cotes_integrate(f, a, b, n):
    """
    Integrate f on [a, b] with the degree-n Newton-Cotes rule.

    Parameters
    ----------
    f    : callable  f(x) -> scalar
    a, b : float     endpoints
    n    : int       degree (0=midpoint, 1=trapezoidal, 2=Simpson, ...)

    Returns
    -------
    I_hat : float — estimate  (b-a) * sum_i w_i * f(t_i)
    """
    t_unit, w = newton_cotes_weights(n)

    if n == 0:
        # midpoint: single evaluation at the centre
        t_phys = np.array([(a + b) / 2.0])
    else:
        # affine map [0,1] -> [a,b]: t_phys = a + t*(b-a)
        t_phys = a + t_unit * (b - a)

    # I_hat = (b-a) * sum_i w_i * f(t_i)
    I_hat = (b - a) * np.dot(w, f(t_phys))
    return I_hat

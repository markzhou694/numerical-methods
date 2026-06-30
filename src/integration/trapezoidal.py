import numpy as np

# Composite trapezoidal rule on [a, b] with m equal subintervals.
#
# Step size h = (b-a)/m, nodes x_i = a + i*h for i = 0, ..., m:
#
#     T(h) = h * [ (1/2)*f(x_0) + f(x_1) + ... + f(x_{m-1}) + (1/2)*f(x_m) ]
#          = h * ( (1/2)*(f(a) + f(b)) + sum_{i=1}^{m-1} f(a + i*h) )
#
# Global error: T(h) - integral(f) = (b-a)*h^2/12 * f''(tau),  tau in [a,b].
#
# Richardson extrapolation lifts to O(h^4):
#
#     T_R = (4*T(h/2) - T(h)) / 3
#
# trapezoidal_nodes_weights returns (nodes, weights) for use with tensor_product_2d.
#
# Usage:
#   T    = trapezoidal(f, a, b, m)
#   x, w = trapezoidal_nodes_weights(a, b, m)


def trapezoidal(f, a, b, m):
    """
    Composite trapezoidal rule on [a, b] with m subintervals.

    Parameters
    ----------
    f    : callable  f(x) -> scalar
    a, b : float     endpoints
    m    : int       number of subintervals

    Returns
    -------
    T : float — quadrature estimate
    """
    h = (b - a) / m
    x = a + np.arange(m + 1) * h     # m+1 nodes

    # T(h) = h * ( (1/2)*f(a) + interior sum + (1/2)*f(b) )
    T = h * (0.5 * f(x[0]) + np.sum(f(x[1:-1])) + 0.5 * f(x[-1]))
    return T


def trapezoidal_nodes_weights(a, b, m):
    """
    Nodes and weights for the composite trapezoidal rule on [a, b].

    Returns arrays so that  T = sum(w * f(x)).
    Intended for use with tensor_product_2d.

    Parameters
    ----------
    a, b : float  endpoints
    m    : int    number of subintervals

    Returns
    -------
    x : (m+1,) array — nodes x_i = a + i*h
    w : (m+1,) array — weights h * [1/2, 1, 1, ..., 1, 1/2]
    """
    h      = (b - a) / m
    x      = a + np.arange(m + 1) * h
    w      = h * np.ones(m + 1)
    w[0]  *= 0.5
    w[-1] *= 0.5
    return x, w

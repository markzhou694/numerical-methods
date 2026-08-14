import numpy as np

# Composite trapezoidal rule on [a, b] with N equal intervals.
#
# Step size h = (b-a)/N, nodes x_i = a + i*h for i = 0, ..., N:
#
#     T(h) = h * [ (1/2)*f(x_0) + f(x_1) + ... + f(x_{N-1}) + (1/2)*f(x_N) ]
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
#   T    = trapezoidal(f, a, b, N)
#   x, w = trapezoidal_nodes_weights(a, b, N)


def trapezoidal(f, a, b, N):
    """
    Composite trapezoidal rule on [a, b] with N intervals.

    Parameters
    ----------
    f    : callable  f(x) -> scalar
    a, b : float     endpoints
    N    : int       number of intervals; the grid has N+1 nodes

    Returns
    -------
    T : float — quadrature estimate
    """
    if not isinstance(N, (int, np.integer)) or N < 1:
        raise ValueError("N must be an integer with N >= 1")

    h = (b - a) / N
    x = np.linspace(a, b, N + 1)

    # T(h) = h * ( (1/2)*f(a) + interior sum + (1/2)*f(b) )
    T = h * (0.5 * f(x[0]) + np.sum(f(x[1:-1])) + 0.5 * f(x[-1]))
    return T


def trapezoidal_nodes_weights(a, b, N):
    """
    Nodes and weights for the composite trapezoidal rule on [a, b].

    Returns arrays so that  T = sum(w * f(x)).
    Intended for use with tensor_product_2d.

    Parameters
    ----------
    a, b : float  endpoints
    N    : int    number of intervals; the grid has N+1 nodes

    Returns
    -------
    x : (N+1,) array — nodes x_i = a + i*h, i=0,...,N
    w : (N+1,) array — weights h * [1/2, 1, 1, ..., 1, 1/2]
    """
    if not isinstance(N, (int, np.integer)) or N < 1:
        raise ValueError("N must be an integer with N >= 1")

    h      = (b - a) / N
    x      = np.linspace(a, b, N + 1)
    w      = h * np.ones(N + 1)
    w[0]  *= 0.5
    w[-1] *= 0.5
    return x, w

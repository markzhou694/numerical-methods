import numpy as np

# Interpolation nodes on the standard interval [-1, 1].
#
# Two node families for polynomial interpolation of degree N (N+1 nodes):
#
#   Equispaced:
#       xi_k = -1 + 2k/N ,            k = 0, 1, ..., N
#
#   Chebyshev (extremal):
#       xi_k = -cos(pi * k / N) ,     k = 0, 1, ..., N
#
# Equispaced nodes are simple but suffer from the Runge phenomenon: the
# Lebesgue constant grows exponentially with N, making high-degree
# interpolation of smooth functions diverge near the endpoints.
#
# Chebyshev extremal nodes cluster near ±1, keeping the Lebesgue constant
# O(log N) and giving near-optimal polynomial interpolation.
#
# User-supplied nodes need no generator. the caller passes their own array
# straight into the Lagrange routine.
#
# Usage:
#   xi = equispaced_nodes(10)    # 11 equally spaced points in [-1, 1]
#   xi = chebyshev_nodes(10)     # 11 Chebyshev points in [-1, 1]


def equispaced_nodes(N):
    """
    N+1 equally spaced nodes on [-1, 1].

    Parameters
    ----------
    N : int — polynomial degree (number of intervals)

    Returns
    -------
    xi : (N+1,) array — nodes xi_k = -1 + 2k/N,  k = 0, ..., N
    """
    if not isinstance(N, (int, np.integer)) or N < 1:
        raise ValueError("N must be an integer with N >= 1")

    k  = np.arange(N + 1)
    xi = -1.0 + 2.0 * k / N
    return xi


def chebyshev_nodes(N):
    """
    N+1 Chebyshev extremal nodes on [-1, 1].

    Parameters
    ----------
    N : int — polynomial degree

    Returns
    -------
    xi : (N+1,) array — nodes xi_k = -cos(pi k / N), k = 0, ..., N
         ordered from xi_0 = -1 (left) to xi_N = 1 (right)
    """
    if not isinstance(N, (int, np.integer)) or N < 1:
        raise ValueError("N must be an integer with N >= 1")

    # Increasing order: xi_0=-1 and xi_N=1.
    k  = np.arange(N + 1)
    xi = -np.cos(np.pi * k / N)
    return xi

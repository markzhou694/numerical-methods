import numpy as np

from interpolation.interval_map import to_standard, from_standard
from interpolation.nodes import equispaced_nodes, chebyshev_nodes

# Classic product-form Lagrange interpolation on the standard interval
# [-1, 1].  All interpolation is carried out in standard coordinates; the
# physical interval [a, b] enters only through the affine map in interval_map.
#
# Given N+1 distinct nodes {xi_0, ..., xi_N} and function values {f_0, ..., f_N},
# the unique interpolating polynomial of degree <= N is
#
#     p(xi) = sum_{k=0}^{N} f_k * L_k(xi)
#
# where the cardinal (Lagrange) basis functions are
#
#     L_k(xi) = prod_{j != k} (xi - xi_j) / (xi_k - xi_j)
#
# This satisfies L_k(xi_j) = delta_{kj}, so p(xi_k) = f_k exactly.
#
# The implementation uses an explicit double loop (over k and j != k) to keep
# the index structure of the product formula visible.
#
# Usage:
#   # low-level: evaluate on standard interval [-1, 1]
#   p = lagrange_eval(xi_eval, xi_nodes, f_nodes)
#
#   # high-level: interpolate f on physical interval [a, b]
#   x_nodes, f_nodes, x_eval, p_eval = lagrange_interpolate(f, a, b, N)


def lagrange_eval(xi_eval, xi_nodes, f_nodes):
    """
    Evaluate the Lagrange interpolant at xi_eval.

    Parameters
    ----------
    xi_eval  : (M,) array — evaluation points in standard coordinates
    xi_nodes : (N+1,) array — interpolation nodes in standard coordinates
    f_nodes  : (N+1,) array — function values at the nodes

    Returns
    -------
    p : (M,) array — interpolant values p(xi_eval)
    """

    N1 = len(xi_nodes)          # number of nodes = N + 1
    M  = len(xi_eval)
    p  = np.zeros(M)

    # accumulate p(xi) = sum_k f_k L_k(xi), one node at a time
    for k in range(N1):
        # build the cardinal basis L_k evaluated at all xi_eval
        L_k = np.ones(M)
        for j in range(N1):
            if j != k:
                # L_k(xi) *= (xi - xi_j) / (xi_k - xi_j)
                L_k = L_k * (xi_eval - xi_nodes[j]) / (xi_nodes[k] - xi_nodes[j])
        # add this node's contribution: f_k * L_k(xi)
        p = p + f_nodes[k] * L_k

    return p


def lagrange_interpolate(f, a, b, N, node_type="chebyshev", xi_nodes=None, x_eval=None):
    """
    Interpolate f on the physical interval [a, b] with N+1 nodes.

    Maps [a, b] to [-1, 1], builds the Lagrange interpolant in standard
    coordinates, and evaluates it on a grid mapped back to [a, b].

    Parameters
    ----------
    f         : callable  f(x) -> array, sampled at physical nodes
    a, b      : endpoints of the physical interval
    N         : polynomial degree (N+1 nodes)
    node_type : 'equispaced', 'chebyshev', or 'user'
    xi_nodes  : (N+1,) array — required when node_type='user'
    x_eval    : (M,) array — physical evaluation points; defaults to
                linspace(a, b, 10*N+1)

    Returns
    -------
    x_nodes : (N+1,) array — node locations in [a, b]
    f_nodes : (N+1,) array — f sampled at x_nodes
    x_eval  : (M,) array — evaluation points in [a, b]
    p_eval  : (M,) array — interpolant values at x_eval
    """

    # 1. standard-interval nodes
    if node_type == "equispaced":
        xi_nodes = equispaced_nodes(N)
    elif node_type == "chebyshev":
        xi_nodes = chebyshev_nodes(N)
    elif node_type == "user":
        xi_nodes = np.asarray(xi_nodes)      # caller supplied them
    else:
        raise ValueError("node_type must be 'equispaced', 'chebyshev', or 'user'")

    # 2. place nodes in the physical domain [a, b]
    x_nodes = from_standard(xi_nodes, a, b)

    # 3. sample f at the physical nodes
    f_nodes = f(x_nodes)

    # 4. default eval grid, then map eval points to the standard interval
    if x_eval is None:
        x_eval = np.linspace(a, b, 10 * N + 1)
    xi_eval = to_standard(x_eval, a, b)

    # 5. evaluate the standard-interval interpolant
    p_eval = lagrange_eval(xi_eval, xi_nodes, f_nodes)

    return x_nodes, f_nodes, x_eval, p_eval


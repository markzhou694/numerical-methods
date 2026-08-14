import numpy as np

# 2D tensor-product quadrature on [a1,b1] x [a2,b2].
#
# Apply the same 1D rule independently in each variable:
#
#     integral integral f(x,y) dx dy  ~=  sum_i sum_j  w_i * w_j * f(x_i, y_j)
#
# where {x_i, w_i} and {y_j, w_j} are the 1D nodes and weights in each direction.
#
# The 1D rule is supplied as a callable:
#     quad_fn(a, b, rule_parameter) -> (nodes, weights)
# so any 1D rule (trapezoidal_nodes_weights, gauss_legendre_nodes_weights, ...)
# can be plugged in without modification.
#
# The bilinear basis structure phi_{ij}(x,y) = phi_i(x) * phi_j(y) from the
# slides is made explicit in the double loop: each (i,j) pair contributes
# w_i * w_j * f(x_i, y_j).
#
# Cost is the product of the one-dimensional node counts.
#
# Usage:
#   from integration.gauss_legendre import gauss_legendre_nodes_weights
#   from integration.trapezoidal    import trapezoidal_nodes_weights
#
#   I = tensor_product_2d(f, 0, 1, 0, 1, gauss_legendre_nodes_weights, 10)


def tensor_product_2d(f, a1, b1, a2, b2, quad_fn, rule_parameter):
    """
    2D tensor-product quadrature on [a1,b1] x [a2,b2].

    Parameters
    ----------
    f         : callable  f(x, y) -> scalar
    a1, b1    : float     x-interval endpoints
    a2, b2    : float     y-interval endpoints
    quad_fn   : callable  quad_fn(a, b, rule_parameter) -> (nodes, weights)
                          — returns 1D nodes and weights in [a, b]
    rule_parameter : int  parameter passed to quad_fn; its meaning is defined
                          by that rule (N intervals or n_nodes nodes)

    Returns
    -------
    I_hat : float — estimate  sum_i sum_j w_i * w_j * f(x_i, y_j)
    """
    x_nodes, w_x = quad_fn(a1, b1, rule_parameter)
    y_nodes, w_y = quad_fn(a2, b2, rule_parameter)

    I_hat = 0.0
    for i in range(len(x_nodes)):
        for j in range(len(y_nodes)):
            # phi_{ij} contribution: w_i * w_j * f(x_i, y_j)
            I_hat += w_x[i] * w_y[j] * f(x_nodes[i], y_nodes[j])

    return I_hat

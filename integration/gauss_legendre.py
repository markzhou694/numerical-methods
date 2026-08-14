import numpy as np

from interpolation.interval_map import from_standard

# Gauss-Legendre quadrature on [a, b] with n_nodes nodes.
#
# Nodes tau_i are roots of the n_nodes-th Legendre polynomial.
# This is not an endpoint grid, so n_nodes counts nodes rather than intervals.
#
#     integral_{-1}^{1} f(xi) dxi  ~=  sum_{i=0}^{n_nodes-1} lam_i * f(tau_i)
#
# For a general interval [a, b], the affine map
#     x = from_standard(xi, a, b) = (a+b)/2 + (b-a)/2 * xi
# has Jacobian (b-a)/2, so:
#
#     integral_a^b f(x) dx
#         ~= (b-a)/2 * sum_{i=0}^{n_nodes-1} lam_i*f(from_standard(tau_i, a, b))
#
# Nodes and weights on [-1,1]:
#     tau, lam = np.polynomial.legendre.leggauss(n_nodes)
#
# gauss_legendre_nodes_weights returns (nodes, weights) in physical coordinates
# for use with tensor_product_2d.
#
# Usage:
#   I    = gauss_legendre(f, a, b, n_nodes)
#   x, w = gauss_legendre_nodes_weights(a, b, n_nodes)


def gauss_legendre(f, a: float, b: float, n_nodes: int):
    """
    n_nodes-point Gauss-Legendre quadrature on [a, b].

    Exact for all polynomials of degree <= 2*n_nodes-1.

    Parameters
    ----------
    f    : callable  f(x) -> scalar
    a, b : float     endpoints
    n_nodes : int    number of Gauss nodes

    Returns
    -------
    I_hat : float — quadrature estimate
    """
    if not isinstance(n_nodes, (int, np.integer)) or n_nodes < 1:
        raise ValueError("n_nodes must be an integer with n_nodes >= 1")

    # nodes tau_i and weights lam_i on [-1,1]
    if not isinstance(n_nodes, (int, np.integer)) or n_nodes < 1:
        raise ValueError("n_nodes must be an integer with n_nodes >= 1")

    tau, lam = np.polynomial.legendre.leggauss(n_nodes)

    # map nodes to the physical interval [a, b]
    x_phys = from_standard(tau, a, b)

    # Jacobian: dx/dxi = (b-a)/2
    jac = (b - a) / 2.0

    # I_hat = jac * sum_i lam_i * f(x_phys_i)
    I_hat = jac * np.dot(lam, f(x_phys))
    return I_hat


def gauss_legendre_nodes_weights(a, b, n_nodes):
    """
    Nodes and weights for n_nodes-point Gauss-Legendre on [a, b].

    Returns arrays so that  I = sum(w * f(x)).
    Intended for use with tensor_product_2d.

    Parameters
    ----------
    a, b : float  endpoints
    n_nodes : int  number of Gauss nodes

    Returns
    -------
    x : (n_nodes,) array — Gauss-Legendre nodes in [a, b]
    w : (n_nodes,) array — weights scaled by the Jacobian (b-a)/2
    """
    tau, lam = np.polynomial.legendre.leggauss(n_nodes)
    x   = from_standard(tau, a, b)
    jac = (b - a) / 2.0
    w   = jac * lam
    return x, w

# test function 
def f(x):
    return np.sin(x)* x**2 + 2*x + 1

def main():
    # Example usage
    a, b = 0, 1
    n_nodes = 10
    result = gauss_legendre(f, a, b, n_nodes)
    print(f"Quadrature estimate: {result}")

if __name__ == "__main__":
    main()

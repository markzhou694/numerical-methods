import __main__

import numpy as np

from interpolation.interval_map import from_standard

# Gauss-Legendre quadrature on [a, b] with n+1 nodes.
#
# Nodes tau_i are the roots of the (n+1)-st Legendre polynomial P_{n+1} on [-1,1].
# Weights lam_i > 0 are chosen so the rule is exact for all polynomials of degree <= 2n+1.
#
#     integral_{-1}^{1} f(xi) dxi  ~=  sum_{i=0}^{n} lam_i * f(tau_i)
#
# For a general interval [a, b], the affine map
#     x = from_standard(xi, a, b) = (a+b)/2 + (b-a)/2 * xi
# has Jacobian (b-a)/2, so:
#
#     integral_a^b f(x) dx  =  (b-a)/2 * sum_{i=0}^{n} lam_i * f(from_standard(tau_i, a, b))
#
# Nodes and weights on [-1,1]:
#     tau, lam = np.polynomial.legendre.leggauss(n+1)
#
# gauss_legendre_nodes_weights returns (nodes, weights) in physical coordinates
# for use with tensor_product_2d.
#
# Usage:
#   I    = gauss_legendre(f, a, b, n)
#   x, w = gauss_legendre_nodes_weights(a, b, n)


def gauss_legendre(f, a, b, n):
    """
    (n+1)-point Gauss-Legendre quadrature on [a, b].

    Exact for all polynomials of degree <= 2n+1.

    Parameters
    ----------
    f    : callable  f(x) -> scalar
    a, b : float     endpoints
    n    : int       degree parameter; uses n+1 nodes

    Returns
    -------
    I_hat : float — quadrature estimate
    """
    # nodes tau_i and weights lam_i on [-1,1]
    tau, lam = np.polynomial.legendre.leggauss(n + 1)

    # map nodes to the physical interval [a, b]
    x_phys = from_standard(tau, a, b)

    # Jacobian: dx/dxi = (b-a)/2
    jac = (b - a) / 2.0

    # I_hat = jac * sum_i lam_i * f(x_phys_i)
    I_hat = jac * np.dot(lam, f(x_phys))
    return I_hat


def gauss_legendre_nodes_weights(a, b, n):
    """
    Nodes and weights for (n+1)-point Gauss-Legendre on [a, b].

    Returns arrays so that  I = sum(w * f(x)).
    Intended for use with tensor_product_2d.

    Parameters
    ----------
    a, b : float  endpoints
    n    : int    degree parameter; returns n+1 nodes/weights

    Returns
    -------
    x : (n+1,) array — Gauss-Legendre nodes in [a, b]
    w : (n+1,) array — weights scaled by the Jacobian (b-a)/2
    """
    tau, lam = np.polynomial.legendre.leggauss(n + 1)
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
    n = 9
    result = gauss_legendre(f, a, b, n)
    print(f"Quadrature estimate: {result}")

if __name__ == "__main__":
    main()
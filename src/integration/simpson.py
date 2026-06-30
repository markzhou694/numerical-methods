import numpy as np

# Composite Simpson's rule on [a, b] with m subintervals (m must be even).
#
# Step size h = (b-a)/m, nodes xᵢ = a + i·h for i = 0, ..., m:
#
#     S(h) = h/3 · [ f(x0) + 4f(x1) + 2f(x2) + 4f(x3) + ... + 4f(x_m-1) + f(x_m) ]
#
# Weight pattern: 1, 4, 2, 4, 2, ..., 2, 4, 1  (interior: alternating 4 and 2).
#
# Global error: O(h^4)  (exact for polynomials of degree ≤ 3 on each pair of
# subintervals, i.e., degree ≤ 3 overall for the composite rule).
#
# Usage:
#   S = simpson(f, a, b, m)   # m must be even


def simpson(f, a, b, m):
    """
    Composite Simpson's rule on [a, b] with m subintervals.

    Parameters
    ----------
    f    : callable  f(x) -> scalar
    a, b : float     endpoints
    m    : int       number of subintervals (must be even)

    Returns
    -------
    S : float — quadrature estimate
    """
    assert m % 2 == 0, "m must be even for composite Simpson's rule"

    h = (b - a) / m
    x = a + np.arange(m + 1) * h

    # weight vector: 1, 4, 2, 4, 2, ..., 4, 1
    w         = np.ones(m + 1)
    w[1:-1:2] = 4     # odd interior indices
    w[2:-2:2] = 2     # even interior indices

    # S(h) = h/3 * Σ wᵢ f(xᵢ)
    S = (h / 3.0) * np.dot(w, f(x))
    return S

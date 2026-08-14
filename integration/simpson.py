import numpy as np

# Composite Simpson's rule on [a, b] with N intervals (N must be even).
#
# Step size h = (b-a)/N, nodes x_i = a + i*h for i = 0, ..., N:
#
#     S(h) = h/3 * [ f(x_0) + 4f(x_1) + ... + 4f(x_{N-1}) + f(x_N) ]
#
# Weight pattern: 1, 4, 2, 4, 2, ..., 2, 4, 1  (interior: alternating 4 and 2).
#
# Global error: O(h^4)  (exact for polynomials of degree ≤ 3 on each pair of
# subintervals, i.e., degree ≤ 3 overall for the composite rule).
#
# Usage:
#   S = simpson(f, a, b, N)   # N must be even


def simpson(f, a, b, N):
    """
    Composite Simpson's rule on [a, b] with N intervals.

    Parameters
    ----------
    f    : callable  f(x) -> scalar
    a, b : float     endpoints
    N    : int       number of intervals (must be even); N+1 nodes

    Returns
    -------
    S : float — quadrature estimate
    """
    if not isinstance(N, (int, np.integer)) or N < 2 or N % 2 != 0:
        raise ValueError("N must be a positive even integer")

    h = (b - a) / N
    x = np.linspace(a, b, N + 1)

    # weight vector: 1, 4, 2, 4, 2, ..., 4, 1
    w         = np.ones(N + 1)
    w[1:-1:2] = 4     # odd interior indices
    w[2:-2:2] = 2     # even interior indices

    # S(h) = h/3 * Σ wᵢ f(xᵢ)
    S = (h / 3.0) * np.dot(w, f(x))
    return S

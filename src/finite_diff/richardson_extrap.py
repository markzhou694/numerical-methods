import numpy as np

# Richardson extrapolation for a p-th order approximation S(h):
#   S(h)   = D + C*h^p + O(h^{p+2})
#   S(h/2) = D + C*(h/2)^p + O(h^{p+2})
#
# Here:
#   D      is the exact quantity we want
#   S(h)  is some numerical approximation of D using step size h
#   p      is the order of the leading error term
#
# Eliminate the leading error term:
#   S_rich = (2^p * S(h/2) - S(h)) / (2^p - 1)
#
# This gives a higher-order approximation to D.
#
# approximation(f, x, h) -> scalar approximation of D using f, x, and h.
#
# If p is not given, estimate p from:
#   S(h), S(h/2), S(h/4)
#
# Since
#   S(h) - S(h/2)     ≈ C*h^p*(1 - 1/2^p)
#   S(h/2) - S(h/4)   ≈ C*(h/2)^p*(1 - 1/2^p)
#
# we have
#   |S(h) - S(h/2)| / |S(h/2) - S(h/4)| ≈ 2^p
#
# Therefore:
#   p ≈ log( |S(h) - S(h/2)| / |S(h/2) - S(h/4)| ) / log(2)


def estimate_p(approximation, f, x, h):
    """
    Estimate order p using three step sizes:
        S(h), S(h/2), S(h/4)

    Assumption:
        S(h) = D + C h^p + higher order terms
    """
    S_h  = approximation(f, x, h)
    S_h2 = approximation(f, x, h / 2)
    S_h4 = approximation(f, x, h / 4)

    numerator   = abs(S_h - S_h2)
    denominator = abs(S_h2 - S_h4)

    if denominator == 0:
        raise ValueError("Cannot estimate p: denominator is zero.")

    p_est = np.log(numerator / denominator) / np.log(2)
    return p_est


def richardson(approximation, f, x, h, p=None):
    """
    Richardson extrapolation for a p-th order approximation.

    If p is None, estimate p numerically first.
    """

    if p is None:
        p = estimate_p(approximation, f, x, h)

    S_h  = approximation(f, x, h)
    S_h2 = approximation(f, x, h / 2)

    scale = 2**p

    if scale == 1:
        raise ValueError("Invalid p: 2**p - 1 is zero.")

    S_rich = (scale * S_h2 - S_h) / (scale - 1)

    return S_rich
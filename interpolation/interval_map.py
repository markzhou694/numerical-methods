import numpy as np

# Affine change of variable between a physical interval [a, b] and the
# standard interpolation interval [-1, 1].
#
#     forward : x in [a, b]   ->  xi in [-1, 1]
#     inverse : xi in [-1, 1] ->  x  in [a, b]
#
# The two maps are exact inverses of each other.


def to_standard(x: float, a: float, b: float) -> float:
    # map physical points to the standard interval:
    #     xi = (2*x - (a + b)) / (b - a)
    xi = (2.0 * x - (a + b)) / (b - a)
    return xi


def from_standard(xi: float, a: float, b: float) -> float:
    # inverse map back to the physical interval:
    #     x = (a + b)/2 + (b - a)/2 * xi
    x_mid  = (a + b) / 2.0
    x_half = (b - a) / 2.0
    x = x_mid + x_half * xi
    return x

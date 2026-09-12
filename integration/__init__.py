from .adaptive_quad import adaptive_trapezoidal
from .gauss_legendre import gauss_quadrature, gauss_legendre_nodes_weights
from .newton_cotes import newton_cotes_integrate, newton_cotes_weights
from .simpson import simpson
from .tensor_product_2d import tensor_product_2d
from .trapezoidal import trapezoidal, trapezoidal_nodes_weights

__all__ = [
    "adaptive_trapezoidal",
    "gauss_legendre",
    "gauss_legendre_nodes_weights",
    "gauss_quadrature",
    "gauss_quadrature",
    "newton_cotes_integrate",
    "newton_cotes_weights",
    "simpson",
    "tensor_product_2d",
    "trapezoidal",
    "trapezoidal_nodes_weights",
]

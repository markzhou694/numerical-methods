try:
    from .interval_map import to_standard, from_standard, from_standard_01, to_standard_01
    from .nodes import equispaced_nodes, chebyshev_nodes
    from .lagrange import lagrange_eval, lagrange_interpolate
except ImportError:
    from interval_map import to_standard, from_standard ,from_standard_01, to_standard_01
    from nodes import equispaced_nodes, chebyshev_nodes
    from lagrange import lagrange_eval, lagrange_interpolate

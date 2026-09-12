import numpy as np

from finite_element import fem_bvp_1d
from iterative_method import gmres


def test_fem_poisson_with_nonzero_dirichlet_values():
    """Solve -u''=2 with u(0)=1 and u(1)=2."""
    kappa = lambda x: np.ones_like(x)
    beta = lambda x: np.zeros_like(x)
    c = lambda x: np.zeros_like(x)
    f = lambda x: 2.0 * np.ones_like(x)

    x_nodes, U = fem_bvp_1d(
        kappa=kappa,
        beta=beta,
        c=c,
        f=f,
        a=0.0,
        b=1.0,
        n_elements=4,
        u_a=1.0,
        u_b=2.0,
    )

    # Exact solution: u(x) = 1 + 2x - x^2.
    U_exact = 1.0 + 2.0 * x_nodes - x_nodes**2

    np.testing.assert_allclose(U, U_exact)
    assert U[0] == 1.0
    assert U[-1] == 2.0


def test_fem_accepts_gmres_tuple_output_directly():
    """GMRES returns (solution, iterations), unlike numpy.linalg.solve."""
    one = lambda x: np.ones_like(x, dtype=float)
    zero = lambda x: np.zeros_like(x, dtype=float)
    two = lambda x: 2.0 * np.ones_like(x, dtype=float)

    x_nodes, U = fem_bvp_1d(
        kappa=one,
        beta=zero,
        c=zero,
        f=two,
        a=0.0,
        b=1.0,
        n_elements=4,
        u_a=1.0,
        u_b=2.0,
        linear_solver=gmres,
    )

    U_exact = 1.0 + 2.0 * x_nodes - x_nodes**2
    np.testing.assert_allclose(U, U_exact, atol=1e-12)

import numpy as np

from finite_diff.fd_1d_bvp import fd_matrix_1d
from integration.trapezoidal import trapezoidal
from interpolation.lagrange import lagrange_interpolate
from ode.forward_euler import forward_euler_step


def test_lagrange_reproduces_quadratic():
    f = lambda x: x**2 - 2.0 * x + 1.0
    x_eval = np.linspace(-1.0, 1.0, 9)

    _, _, _, p_eval = lagrange_interpolate(
        f,
        -1.0,
        1.0,
        2,
        node_type="chebyshev",
        x_eval=x_eval,
    )

    assert np.allclose(p_eval, f(x_eval))


def test_trapezoidal_integrates_linear_function():
    f = lambda x: 2.0 * x + 1.0

    T = trapezoidal(f, 0.0, 1.0, 8)

    assert np.allclose(T, 2.0)


def test_forward_euler_step():
    f = lambda t, U: -U

    U_new = forward_euler_step(f, 0.0, np.array([1.0]), 0.25)

    assert np.allclose(U_new, np.array([0.75]))


def test_fd_1d_bvp_boundary_values_are_preserved():
    # u'' = 0 on [0, 1] with u(0) = 1, u(1) = 3  ->  linear solution u(x) = 1 + 2x
    f = lambda x: np.zeros_like(x)
    u_a, u_b = 1.0, 3.0

    x, A = fd_matrix_1d(m=4, c=0.0, b=0.0, x_left=0.0, x_right=1.0)
    h = x[1] - x[0]

    # assemble RHS on interior nodes and fold Dirichlet BCs into it
    F = f(x[1:-1]).copy()
    F[0] -= u_a / h**2
    F[-1] -= u_b / h**2

    U_int = np.linalg.solve(A, F)
    U = np.concatenate(([u_a], U_int, [u_b]))

    assert np.allclose(x[[0, -1]], np.array([0.0, 1.0]))
    assert np.allclose(U[[0, -1]], np.array([1.0, 3.0]))
    assert np.allclose(U, 1.0 + 2.0 * x)

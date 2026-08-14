import numpy as np
from scipy import sparse


# Square-grid discretization of Delta u = f on a rectangle.
#
# N is the number of intervals in each coordinate direction.  Each full
# coordinate grid has N+1 points x_0, ..., x_N, while the unknown interior
# grid has N-1 points in each direction and (N-1)^2 points in total.


def poisson_2d_matrix(
    N: int,
    x_range: tuple = (-0.5, 0.5),
    y_range: tuple = (-0.5, 0.5),
    stencil: int = 5,
):
    """
    Build a 5-point or 9-point Laplacian on the interior grid.

    Parameters
    ----------
    N : int
        Number of equal intervals in each coordinate direction.
    x_range, y_range : tuple
        Rectangle endpoints.
    stencil : {5, 9}
        Five-point second-order stencil or nine-point compact stencil.

    Returns
    -------
    x_interior, y_interior : (N-1,) ndarray
        Interior coordinate grids x_1, ..., x_{N-1}.
    A : ((N-1)^2, (N-1)^2) sparse CSC matrix
        Discretization of Delta u.  The 9-point choice requires the modified
        right-hand side f + h^2*Delta(f)/12 for fourth-order accuracy.
    """
    if not isinstance(N, (int, np.integer)) or N < 2:
        raise ValueError("N must be an integer with N >= 2")
    if stencil not in (5, 9):
        raise ValueError("stencil must be 5 or 9")

    h_x = (x_range[1] - x_range[0]) / N
    h_y = (y_range[1] - y_range[0]) / N
    if not np.isclose(h_x, h_y):
        raise ValueError("x and y spacings must match")
    h = h_x

    x = np.linspace(x_range[0], x_range[1], N + 1)
    y = np.linspace(y_range[0], y_range[1], N + 1)
    x_interior = x[1:N]
    y_interior = y[1:N]
    M = N - 1

    if stencil == 5:
        # Centre row block: x-neighbours have weight 1; centre has weight -4.
        T = sparse.diags(
            [np.ones(M - 1), -4.0 * np.ones(M), np.ones(M - 1)],
            [-1, 0, 1],
            format="csc",
        )
        A = sparse.kron(sparse.eye(M), T, format="csc")
        y_neighbours = sparse.kron(
            sparse.diags([np.ones(M - 1)], [1], shape=(M, M)),
            sparse.eye(M),
            format="csc",
        )
        A = (A + y_neighbours + y_neighbours.T) / h**2

    else:
        # Nine-point weights: centre -20, edge neighbours 4, corners 1.
        T = sparse.diags(
            [4.0 * np.ones(M - 1), -20.0 * np.ones(M), 4.0 * np.ones(M - 1)],
            [-1, 0, 1],
            format="csc",
        )
        Q = sparse.diags(
            [np.ones(M - 1), 4.0 * np.ones(M), np.ones(M - 1)],
            [-1, 0, 1],
            format="csc",
        )
        A = sparse.kron(sparse.eye(M), T, format="csc")
        adjacent_rows = sparse.kron(
            sparse.diags([np.ones(M - 1)], [1], shape=(M, M)),
            Q,
            format="csc",
        )
        A = (A + adjacent_rows + adjacent_rows.T) / (6.0 * h**2)

    return x_interior, y_interior, A

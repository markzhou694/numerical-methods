import numpy as np

from integration import gauss_quadrature

from iterative_method import gmres

from matplotlib import pyplot as plt

# This file intentionally contains only the fixed building blocks for
# rebuilding the one-dimensional finite-element method:
#
#   1. global piecewise-linear nodal basis functions phi_j,
#   2. their first weak derivatives, represented element by element,
#   3. the bilinear (energy) form
#
#          a(u, v) = integral [
#              kappa*u'*v' + beta*u'*v + c*u*v
#          ] dx.
#
# Matrix assembly, load vectors, boundary conditions, solving, interpolation,
# and error analysis are deliberately left out so they can be worked through
# from the weak form step by step.

def standard_nodal_basis(a, b, n_elements):
    """Return all piecewise-linear nodal basis functions phi_0, ..., phi_N.

    This function returns function objects, not their values at one point.
    Afterward, use ``phi[j](x)`` to evaluate the j-th basis function.

    Parameters
    ----------
    a : float
        The left endpoint of the physical interval.
    b : float
        The right endpoint of the physical interval.
    n_elements : int
        The number of elements (subintervals) in the partition of [a, b].

    Returns
    -------
    phi : list of callable
        ``phi[j]`` is the function phi_j.  Each function accepts a scalar or
        a NumPy array.
    """
    if b <= a:
        raise ValueError("b must be greater than a")
    if not isinstance(n_elements, (int, np.integer)) or n_elements < 1:
        raise ValueError("n_elements must be a positive integer")

    h = (b - a) / n_elements
    x_nodes = np.linspace(a, b, n_elements + 1)
    phi = []

    for j in range(n_elements + 1):
        def phi_j(x, j=j):
            x_values = np.asarray(x, dtype=float)
            if np.any((x_values < a) | (x_values > b)):
                raise ValueError("x must lie in [a, b]")

            distance_from_node_j = np.abs(x_values - x_nodes[j])
            values = np.maximum(1.0 - distance_from_node_j / h, 0.0)

            if values.ndim == 0:
                return float(values)
            return values

        phi.append(phi_j)

    return phi


def standard_nodal_basis_derivative(a, b, n_elements):
    """Return d(phi_0)/dx, ..., d(phi_N)/dx as function objects.

    Afterward, use ``dphi[j](x)``.  At an interior mesh node the classical
    derivative does not exist; the implementation uses the derivative from
    the element to the right.  This pointwise convention does not affect FEM
    integrals.

    Parameters
    ----------
    a : float
        The left endpoint of the physical interval.
    b : float
        The right endpoint of the physical interval.
    n_elements : int
        The number of elements (subintervals) in the partition of [a, b].

    Returns
    -------
    dphi : list of callable
        ``dphi[j]`` is the function d(phi_j)/dx.
    """
    if b <= a:
        raise ValueError("b must be greater than a")
    if not isinstance(n_elements, (int, np.integer)) or n_elements < 1:
        raise ValueError("n_elements must be a positive integer")

    h = (b - a) / n_elements
    dphi = []

    for j in range(n_elements + 1):
        def dphi_j(x, j=j):
            x_values = np.asarray(x, dtype=float)
            if np.any((x_values < a) | (x_values > b)):
                raise ValueError("x must lie in [a, b]")

            element_index = np.floor((x_values - a) / h).astype(int)
            element_index = np.clip(element_index, 0, n_elements - 1)

            values = np.zeros_like(x_values, dtype=float)
            values = np.where(j == element_index, -1.0 / h, values)
            values = np.where(j == element_index + 1, 1.0 / h, values)

            if values.ndim == 0:
                return float(values)
            return values

        dphi.append(dphi_j)

    return dphi


def plot_standard_nodal_basis(a, b, n_elements):
    """Plot all nodal basis functions and their weak derivatives."""
    import matplotlib.pyplot as plt

    x_plot = np.linspace(a, b, 501)

    phi = standard_nodal_basis(a, b, n_elements)
    dphi = standard_nodal_basis_derivative(a, b, n_elements)

    figure, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for j in range(n_elements + 1):
        axes[0].plot(x_plot, phi[j](x_plot), label=rf"$\phi_{j}$")
        axes[1].plot(x_plot, dphi[j](x_plot), label=rf"$\phi_{j}'$")

    axes[0].set_title("Piecewise-Linear Nodal Basis Functions")
    axes[0].set_ylabel(r"$\phi_j(x)$")
    axes[0].legend()
    axes[1].set_xlabel("x")
    axes[1].set_ylabel(r"$\phi_j'(x)$")
    axes[1].legend()
    figure.tight_layout()
    plt.show()


def fem_bvp_1d(kappa, beta, c, f, a, b, n_elements, u_a, u_b, linear_solver=np.linalg.solve):
    """Solve the 1D FEM system with Dirichlet data u(a)=u_a, u(b)=u_b.

    Returns all physical mesh nodes and all nodal values
    U = [U_0, U_1, ..., U_N].

    ``linear_solver(A, b)`` may return either the solution array itself, as
    ``numpy.linalg.solve`` does, or a tuple whose first item is the solution,
    as this project's ``gmres`` does.
    """
    if b <= a:
        raise ValueError("b must be greater than a")

    if n_elements < 1:
        raise ValueError("n_elements must be at least 1")


    # interval length, factor 
    L = b - a

    #define mapped coefficients
    def kappa_mapped(xi):
   
        return kappa(a + L * xi) 
    def beta_mapped(xi):
      
        return beta(a + L * xi) 
    def c_mapped(xi):
     
        return c(a + L * xi)

    def f_mapped(xi):
        
        return f(a + L * xi) 

    # since works for all v in H_0^1, we can choose v = phi_j for j = 1, ..., n_elements - 1
    # then we have a linear system of equations for the unknown coefficients u_j
    # we will use the standard nodal basis functions phi_j defined on [0, 1] and then map them to [a, b] using an affine transformation

    # we don't need to care about the SUM Vi part since we are only interested in the coefficients u_j for j = 1, ..., n_elements - 1

    # The test functions vanish at the two boundary nodes, so the unknown
    # equations correspond to phi_1, ..., phi_{n_elements-1}.  The solution
    # itself may have nonzero values u_a and u_b at the boundary.

    # we will use the standard nodal basis functions phi_j defined on [0, 1] and then map them to [a, b] using an affine transformation

    #let's define the bilinear form a(u, v) = integral [ kappa/L *u'*v' + beta*u'*v + Lc*u*v ] dx
    # we will use the standard nodal basis functions phi_j defined on [0, 1] and then map them to [a, b] using an affine transformation 
   
    A = np.zeros((n_elements + 1, n_elements + 1))

    F = np.zeros(n_elements + 1)

    #each row of the matrix A corresponds to a test Vi basis function phi_j for j = 0, 1, ..., n_elements

    # These are lists of functions on the mapped interval [0, 1].
    # For example, phi[j](xi) evaluates phi_j at xi.
    phi = standard_nodal_basis(0.0, 1.0, n_elements)
    dphi = standard_nodal_basis_derivative(0.0, 1.0, n_elements)

    def element_bilinear_entry(test_i, trial_j, xi_left, xi_right):
        """Compute the element contribution to A[test_i, trial_j].

        The matrix convention is

            A[i, j] = a(phi_j, phi_i),

        so phi_j is the trial (solution) basis and phi_i is the test basis.
        """
        def diffusion_integrand(xi):
            return (
                kappa_mapped(xi)
                * dphi[trial_j](xi)
                * dphi[test_i](xi)
            )

        def convection_integrand(xi):
            return (
                beta_mapped(xi)
                * dphi[trial_j](xi)
                * phi[test_i](xi)
            )

        def reaction_integrand(xi):
            return (
                c_mapped(xi)
                * phi[trial_j](xi)
                * phi[test_i](xi)
            )

        diffusion = (1.0 / L) * gauss_quadrature(
            diffusion_integrand, xi_left, xi_right, 2
        )
        convection = gauss_quadrature(
            convection_integrand, xi_left, xi_right, 2
        )
        reaction = L * gauss_quadrature(
            reaction_integrand, xi_left, xi_right, 2
        )

        return diffusion + convection + reaction

    def element_load_entry(test_i, xi_left, xi_right):
        """Compute L times integral(f_mapped * phi_i) on one element."""
        def load_integrand(xi):
            return f_mapped(xi) * phi[test_i](xi)

        return L * gauss_quadrature(
            load_integrand, xi_left, xi_right, 2
        )

    # Assemble one 2-by-2 element block at a time.
    # Element e has global nodes [e, e + 1] and occupies
    # [e / n_elements, (e + 1) / n_elements] in the xi-coordinate.
    for element in range(n_elements):
        xi_left = element / n_elements
        xi_right = (element + 1) / n_elements
        global_nodes = [element, element + 1]

        for test_i in global_nodes:          # matrix row / test basis
            F[test_i] += element_load_entry(
                test_i,
                xi_left,
                xi_right,
            )

            for trial_j in global_nodes:     # matrix column
                A[test_i, trial_j] += element_bilinear_entry(
                    test_i,
                    trial_j,
                    xi_left,
                    xi_right,
                )

    # Insert the known Dirichlet values into the full nodal vector.
    U = np.zeros(n_elements + 1)
    U[0] = u_a
    U[-1] = u_b

    # Solve only for U_1, ..., U_{N-1}.  Move the two known boundary
    # contributions from the left-hand side to the load vector:
    #
    #   A_II U_I = F_I - A_I0 u_a - A_IN u_b.
    if n_elements > 1:
        A_interior = A[1:-1, 1:-1]
        F_interior = F[1:-1].copy()
        F_interior -= A[1:-1, 0] * u_a
        F_interior -= A[1:-1, -1] * u_b

        solver_output = linear_solver(A_interior, F_interior)

        if isinstance(solver_output, tuple):
            U_interior = solver_output[0]
        else:
            U_interior = solver_output

        U[1:-1] = U_interior

    x_nodes = np.linspace(a, b, n_elements + 1)
    return x_nodes, U


# ------------------------------------------------------------
# Problem data
#
#   -(kappa(x) u'(x))' + beta(x) u'(x) + c(x) u(x) = f(x)
#
#   u(0) = 0,  u(1) = 1
#
# Interfaces:
#   x = 0.3 and x = 0.7
# ------------------------------------------------------------

def kappa(x):
    """Three-layer discontinuous diffusion coefficient."""
    x = np.asarray(x, dtype=float)

    return np.where(
        x <= 0.3,
        1.0,
        np.where(x <= 0.7, 5.0, 20.0),
    )


def beta(x):
    """Variable convection coefficient that changes sign."""
    x = np.asarray(x, dtype=float)
    return 2.0 * x - 0.5


def c(x):
    """Positive variable reaction coefficient."""
    x = np.asarray(x, dtype=float)
    return 1.0 + x


def flux_shape(x):
    """
    Unscaled exact flux.

    We construct the exact solution by prescribing

        kappa(x) u'(x) = scale * (1 + x).

    Because 1+x is continuous, the physical flux is continuous
    across both material interfaces.
    """
    x = np.asarray(x, dtype=float)
    return 1.0 + x


def antiderivative_flux_shape(x):
    """Integral of 1+x."""
    x = np.asarray(x, dtype=float)
    return x + 0.5 * x**2


def unscaled_exact_solution(x):
    """
    Compute

        integral_0^x (1+s)/kappa(s) ds

    explicitly, one material region at a time.
    """
    x = np.asarray(x, dtype=float)

    G = antiderivative_flux_shape

    # Contribution from [0, 0.3].
    end_1 = np.minimum(x, 0.3)
    part_1 = (G(end_1) - G(0.0)) / 1.0

    # Contribution from [0.3, 0.7].
    end_2 = np.minimum(np.maximum(x, 0.3), 0.7)
    part_2 = (G(end_2) - G(0.3)) / 5.0

    # Contribution from [0.7, 1].
    end_3 = np.maximum(x, 0.7)
    part_3 = (G(end_3) - G(0.7)) / 20.0

    return part_1 + part_2 + part_3


# Choose the scaling so that u(1)=1.
normalization = float(unscaled_exact_solution(1.0))
flux_scale = 1.0 / normalization


def exact_solution(x):
    """
    Exact solution satisfying u(0)=0 and u(1)=1.
    """
    return flux_scale * unscaled_exact_solution(x)


def exact_derivative(x):
    """
    Piecewise derivative:

        u'(x) = flux_scale * (1+x)/kappa(x).

    The derivative jumps at the interfaces, but kappa*u'
    remains continuous.
    """
    x = np.asarray(x, dtype=float)
    return flux_scale * flux_shape(x) / kappa(x)


def exact_flux(x):
    """Exact continuous flux kappa*u'."""
    x = np.asarray(x, dtype=float)
    return flux_scale * flux_shape(x)


def f(x):
    """
    Manufactured right-hand side.

    Since

        kappa*u' = flux_scale*(1+x),

    we have

        (kappa*u')' = flux_scale.

    Therefore

        f = -flux_scale + beta*u' + c*u.
    """
    x = np.asarray(x, dtype=float)

    return (
        -flux_scale
        + beta(x) * exact_derivative(x)
        + c(x) * exact_solution(x)
    )


def main():
    """Solve the interface problem and check mesh convergence."""

    def gmres_linear_solver(A, b):
        solution, iterations = gmres(
            A,
            b,
            max_iter=300,
            tol=1e-12,
            restart=A.shape[0],
        )

        residual = np.linalg.norm(b - A @ solution)

        print(f"  GMRES iterations: {iterations}")
        print(f"  GMRES residual:   {residual:.3e}")

        return solution

    # Multiples of 10 ensure that x=0.3 and x=0.7 are mesh nodes.
    mesh_sizes = [20, 40, 80]
    nodal_errors = []

    finest_x = None
    finest_U = None

    for n_elements in mesh_sizes:
        print(f"\nNumber of elements: {n_elements}")

        x_nodes, U = fem_bvp_1d(
            kappa=kappa,
            beta=beta,
            c=c,
            f=f,
            a=0.0,
            b=1.0,
            n_elements=n_elements,
            u_a=0.0,
            u_b=1.0,
            linear_solver=gmres_linear_solver,
        )

        U_exact = exact_solution(x_nodes)
        max_nodal_error = np.max(np.abs(U - U_exact))

        nodal_errors.append(max_nodal_error)

        print(f"  Maximum nodal error: {max_nodal_error:.6e}")

        if len(nodal_errors) >= 2:
            observed_rate = np.log2(
                nodal_errors[-2] / nodal_errors[-1]
            )
            print(f"  Observed nodal rate: {observed_rate:.3f}")

        finest_x = x_nodes
        finest_U = U

    # Basic convergence check.
    assert nodal_errors[-1] < nodal_errors[0]

    # --------------------------------------------------------
    # Explicitly verify flux continuity at the interfaces.
    # --------------------------------------------------------

    epsilon = 1e-12

    for interface in [0.3, 0.7]:
        flux_left = (
            kappa(interface - epsilon)
            * exact_derivative(interface - epsilon)
        )

        flux_right = (
            kappa(interface + epsilon)
            * exact_derivative(interface + epsilon)
        )

        print(f"\nInterface x = {interface}")
        print(f"  Flux from left:  {flux_left:.12f}")
        print(f"  Flux from right: {flux_right:.12f}")
        print(f"  Flux jump:       {flux_right - flux_left:.3e}")

        np.testing.assert_allclose(
            flux_left,
            flux_right,
            atol=1e-10,
        )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    x_plot = np.linspace(0.0, 1.0, 1001)
    u_exact_plot = exact_solution(x_plot)

    plt.figure(figsize=(9, 5.5))

    plt.plot(
        x_plot,
        u_exact_plot,
        color="black",
        linewidth=2.2,
        label="Exact interface solution",
    )

    plt.plot(
        finest_x,
        finest_U,
        "o-",
        color="tab:blue",
        markersize=3.5,
        linewidth=1.2,
        label=f"FEM with GMRES ({mesh_sizes[-1]} elements)",
    )

    plt.axvline(
        0.3,
        color="gray",
        linestyle=":",
        label="Material interfaces",
    )

    plt.axvline(
        0.7,
        color="gray",
        linestyle=":",
    )

    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(
        "Convection-Diffusion-Reaction FEM\n"
        "with a Three-Layer Discontinuous Coefficient"
    )
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot kappa and the derivative jump.
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(
        x_plot,
        kappa(x_plot),
        color="tab:orange",
        linewidth=2,
    )
    axes[0].set_ylabel(r"$\kappa(x)$")
    axes[0].set_title("Discontinuous Coefficient and Exact Derivative")
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        x_plot,
        exact_derivative(x_plot),
        color="tab:green",
        linewidth=2,
    )
    axes[1].set_xlabel("x")
    axes[1].set_ylabel(r"$u'(x)$")
    axes[1].grid(alpha=0.3)

    for ax in axes:
        ax.axvline(0.3, color="gray", linestyle=":")
        ax.axvline(0.7, color="gray", linestyle=":")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
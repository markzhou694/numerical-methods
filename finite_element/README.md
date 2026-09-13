# Linear FEM in 1D: rebuild notes

This folder has intentionally been reduced to the fixed ingredients needed to
work through the finite-element method again from the weak form.

The retained functions are:

- `linear_hat_basis(j, x_nodes, x_eval)` for the global nodal basis \(\phi_j\);
- `linear_hat_basis_derivative(...)` for its elementwise first weak derivative;
- `energy_a(u, u_prime, v, v_prime, ...)` for

\[
a(u,v)=\int_{x_L}^{x_R}
\left(\kappa u'v'+\beta u'v+cuv\right)dx.
\]

The following steps are deliberately not implemented yet:

1. define the load functional \(L(v)\);
2. compute \(A_{ij}=a(\phi_j,\phi_i)\);
3. compute \(F_i=L(\phi_i)\);
4. assemble element contributions;
5. impose boundary conditions;
6. solve the linear system;
7. reconstruct and test \(u_h\).

Those steps can now be added one at a time while following the mathematics.

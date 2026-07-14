# Progress Log

---

## `src/finite_diff/` — 6 files complete

All finite-diff modules follow the **matrix-builder pattern**: functions return the operator matrix and grid; the caller assembles the RHS, folds BCs, and calls any solver.

---

### `src/finite_diff/fd_1d_bvp.py`
- **Exports:** `fd_matrix_sys_1d(m, c=0.0, b=0.0, f=..., x_left=-1.0, x_right=1.0, u_a=0.0, u_b=0.0) -> (x, A, F)`
- **Method:** Centered FD operator for `u'' + b·u' − c·u = f` on `[x_left, x_right]`; now assembles the full system (matrix + Dirichlet-adjusted RHS) instead of just the matrix
- **Key math:** `A = D_2 + b·D_1 − c·I` (m×m); `D_2 = tridiag(1,−2,1)/h²`; `D_1 = tridiag(−1,0,1)/(2h)`; advection term `b·D_1` included for generality; `F` folds in `u_a`, `u_b` at the boundary rows
- **Reference:** hw3/hw3-prob2a.py

### `src/finite_diff/richardson_extrap.py`
- **Exports:** `estimate_p(approximation, f, x, h) -> float` · `richardson(approximation, f, x, h, p=None) -> scalar`
- **Method:** Richardson extrapolation; stencil injected as callable `approximation(f, x, h)`
- **Key math:** `S_rich = (2^p · S(h/2) − S(h)) / (2^p − 1)` cancels leading error; if `p=None`, estimated from ratio `log|S(h)−S(h/2)| / |S(h/2)−S(h/4)|| / log 2`
- **Reference:** hw3/hw3-prob3b.py

### `src/iterative/newton.py` (moved out of `finite_diff/` — general-purpose, not BVP-specific)
- **Exports:** `newton_solve(F_func, J_func, theta0, max_iter, tol, damping) -> (theta, iters)`
- **Method:** General damped Newton solver; `F` and `J` injected as callables
- **Key math:** `J·delta = F`; `theta ← theta − damping·delta`; stop when `‖delta‖ < tol`; `damping=0.5` matches the pendulum BVP reference
- **Reference:** hw3/hw3-prob4.py

### `src/finite_diff/fd_2d_poisson_5pt.py`
- **Exports:** `poisson5pt_matrix(m, x_range, y_range) -> (xi, yi, A)`
- **Method:** 5-point Laplacian on interior nodes, 2nd-order accurate
- **Key math:** `A = (1/h²) · kron(I_m, T) + off-diagonal I blocks`; `T = tridiag(−4,1,1)`; domain parameterised via `x_range`, `y_range`; asserts `hx == hy`
- **Reference:** hw3/hw3-prob5.py

### `src/finite_diff/fd_2d_poisson_9pt.py`
- **Exports:** `poisson9pt_matrix(m, x_range, y_range) -> (xi, yi, A9)`
- **Method:** 9-point compact Laplacian, 4th-order accurate
- **Key math:** `A9 = (1/6h²) · block-tridiag(Q9, T9, Q9)`; `T9 = tridiag(−20,4,4)`, `Q9 = tridiag(4,1,1)`; caller supplies modified RHS `f* = f + (h²/12)·Δf`
- **Open question:** Modified RHS requires the exact Laplacian of f; for general f substitute a 5-point FD Laplacian
- **Reference:** hw3/hw3-prob5.py

### `src/finite_diff/fd_neumann_bvp.py`
- **Exports:** `neumann_matrix_pinned(m) -> (x, A)` · `neumann_matrix_singular(m) -> (x, A)`
- **Method:** Neumann BVP on [0,1]; two system matrices for the two solution strategies
- **Key math:** BC rows replaced with one-sided `O(h)` stencil `(u_{j+1}−u_j)/h`; pinned version uses dense numpy with row-0 replaced by identity (uniqueness); singular version uses explicit `lil_matrix` loop over interior rows → CSC (for GMRES); the two solutions differ by a null-space constant
- **Reference:** hw3/hw3-prob6c.py, hw3/hw3-prob6d.py

---

## `src/ode/` — 3 files complete

---

### `src/ode/forward_euler.py`
- **Exports:** `fe_step(f, t, U, k) -> U_new`
- **Method:** Forward (explicit) Euler — one function evaluation per step
- **Key math:** `U_{n+1} = U_n + k·f(t, U_n)`; O(k) local truncation error, O(1) cost

### `src/ode/rk23_adaptive.py`
- **Exports:** `rk23_adaptive_step(f, t, U, k, tol, safety) -> (U_new, Z, k_new)` · `rk23_solve(f, t0, U0, t_final, k_initial, tol, safety) -> (t_values, U_values)`
- **Method:** Bogacki-Shampine RK23 with embedded error control
- **Key math:** 3rd-order update `U_new = U + k(2/9·Y1 + 1/3·Y2 + 4/9·Y3)`; 2nd-order estimate `Z = U + k(7/24·Y1 + 1/4·Y2 + 1/3·Y3 + 1/8·Y4)`; `k_new = safety·k·(tol/‖U_new−Z‖)^(1/3)`; FSAL: `Y4` at accepted step reused as `Y1` next step
- **Reference:** hw5/hw5-prob3a.py, hw5/hw5-prob3b.py

### `src/ode/rk4_stability.py`
- **Exports:** `rk4_stability_region(re_range, im_range, n_pts) -> (Re, Im, mask)`
- **Method:** Absolute stability region of classical RK4
- **Key math:** `p(z) = 1 + z + z²/2 + z³/6 + z⁴/24`; `mask = |p(z)| ≤ 1` on complex grid; returns meshgrid arrays ready for `contourf`
- **Reference:** hw5/hw5-prob2.py

### `src/ode/backward_euler.py`
- **Exports:** `be_matrix(J, k) -> M` · `be_step_newton(f, Jf, t, U, k, tol, max_iter) -> V`
- **Method:** Backward Euler — linear matrix builder and nonlinear Newton stepper
- **Key math:** Linear: `M = I − k·J`; Nonlinear: solve `G(V) = V − U_n − k·f(t+k,V) = 0`; Newton: `dG = I − k·Jf(t+k,V)`, `delta = dG⁻¹G`, `V ← V − delta`; initial guess = explicit Euler predictor; stop when `‖G(V)‖ < tol`
- **Reference:** hw6/hw6-prob2b.py

---

## Not yet started

| Module | Files remaining |
|---|---|
| `src/fourier/` | `dft_matrix.py`, `dft_matrix_odd.py`, `dft_matrix2.py`, `fourier_interpolant.py`, `algo3_pfpp.py`, `acyclic_conv.py` |
| `src/chebyshev/` | `cheb_diff_matrix.py`, `cheb_bvp.py`, `legendre_coeff_diff.py`, `legendre_pseudospectral_bvp.py` |
| `src/integral_eq/` | `nystrom_bie.py`, `fredholm_trapezoidal.py`, `fredholm_second_kind.py` |
| `src/pde/` | `heat_adi.py`, `heat_nonlinear_be.py`, `schrodinger_explicit.py`, `schrodinger_cn_newton.py`, `advection_char.py` |
| `src/iterative/` | (no files defined yet) |

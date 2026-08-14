# Progress Log

---

## `finite_diff/` — unified finite-difference interfaces

Standard grid convention: `N` is the number of intervals, so every coordinate grid is `x_0, ..., x_N` with `N+1` points.

---

### `finite_diff/fd_1d_bvp.py`
- **Exports:** `fd_bvp_1d(N, a, b, c, f, x_left, x_right, bc_left, bc_right) -> (x, A, F)`
- **Method:** One full-grid system for `a(x)u'' + b(x)u' + c(x)u = f`; scalar and variable coefficients are supported.
- **Boundary data:** Both sides use `(alpha, beta, gamma)` for `alpha*u + beta*u' = gamma`, covering Dirichlet, Neumann, Robin, and mixed cases without separate solvers.
- **Key math:** Interior centered differences are second order; boundary derivatives use second-order one-sided differences.

### `finite_diff/richardson_extrap.py`
- **Exports:** `estimate_p(approximation, f, x, h) -> float` · `richardson(approximation, f, x, h, p=None) -> scalar`
- **Method:** Richardson extrapolation; stencil injected as callable `approximation(f, x, h)`
- **Key math:** `S_rich = (2^p · S(h/2) − S(h)) / (2^p − 1)` cancels leading error; if `p=None`, estimated from ratio `log|S(h)−S(h/2)| / |S(h/2)−S(h/4)|| / log 2`

### `iterative/newton.py` (general-purpose, not BVP-specific)
- **Exports:** `newton_solve(F_func, J_func, theta0, max_iter, tol, damping) -> (theta, iters)`
- **Method:** General damped Newton solver; `F` and `J` injected as callables
- **Key math:** `J·delta = F`; `theta ← theta − damping·delta`; stop when `‖delta‖ < tol`; `damping=0.5` matches the pendulum BVP reference

### `finite_diff/poisson_2d.py`
- **Exports:** `poisson_2d_matrix(N, x_range, y_range, stencil) -> (x_interior, y_interior, A)`
- **Method:** One configurable 2D Poisson builder; `stencil=5` selects the second-order five-point stencil and `stencil=9` selects the compact nine-point stencil.
- **Key math:** With `N` intervals per direction there are `N-1` interior points per direction and `(N-1)^2` unknowns.

---

## `ode/`

---

### `ode/forward_euler.py`
- **Exports:** `forward_euler_step(f, t, U, k) -> U_new` · `forward_euler_solve(f, t0, U0, t_final, N)`
- **Method:** Forward (explicit) Euler — one function evaluation per step
- **Key math:** `U_{n+1} = U_n + k·f(t, U_n)`; O(k) local truncation error, O(1) cost

### `ode/rk23_adaptive.py`
- **Exports:** `rk23_adaptive_step(f, t, U, k, tol, safety) -> (U_new, Z, k_new)` · `rk23_solve(f, t0, U0, t_final, k_initial, tol, safety) -> (t_values, U_values)`
- **Method:** Bogacki-Shampine RK23 with embedded error control
- **Key math:** 3rd-order update `U_new = U + k(2/9·Y1 + 1/3·Y2 + 4/9·Y3)`; 2nd-order estimate `Z = U + k(7/24·Y1 + 1/4·Y2 + 1/3·Y3 + 1/8·Y4)`; `k_new = safety·k·(tol/‖U_new−Z‖)^(1/3)`; FSAL: `Y4` at accepted step reused as `Y1` next step
- **Reference:** hw5/hw5-prob3a.py, hw5/hw5-prob3b.py

### `ode/rk4_stability.py`
- **Exports:** `rk4_stability_region(re_range, im_range, N) -> (Re, Im, mask)`
- **Method:** Absolute stability region of classical RK4
- **Key math:** `p(z) = 1 + z + z²/2 + z³/6 + z⁴/24`; `mask = |p(z)| ≤ 1` on complex grid; returns meshgrid arrays ready for `contourf`
- **Reference:** hw5/hw5-prob2.py

### `ode/backward_euler.py`
- **Exports:** `be_matrix(J, k) -> M` · `be_step_newton(f, Jf, t, U, k, tol, max_iter) -> V`
- **Method:** Backward Euler — linear matrix builder and nonlinear Newton stepper
- **Key math:** Linear: `M = I − k·J`; Nonlinear: solve `G(V) = V − U_n − k·f(t+k,V) = 0`; Newton: `dG = I − k·Jf(t+k,V)`, `delta = dG⁻¹G`, `V ← V − delta`; initial guess = explicit Euler predictor; stop when `‖G(V)‖ < tol`
- **Reference:** hw6/hw6-prob2b.py

---

## Not yet started

| Module | Files remaining |
|---|---|
| `fourier/` | `dft_matrix.py`, `dft_matrix_odd.py`, `dft_matrix2.py`, `fourier_interpolant.py`, `algo3_pfpp.py`, `acyclic_conv.py` |
| `chebyshev/` | `cheb_diff_matrix.py`, `cheb_bvp.py`, `legendre_coeff_diff.py`, `legendre_pseudospectral_bvp.py` |
| `integral_eq/` | `nystrom_bie.py`, `fredholm_trapezoidal.py`, `fredholm_second_kind.py` |
| `pde/` | `heat_adi.py`, `heat_nonlinear_be.py`, `schrodinger_explicit.py`, `schrodinger_cn_newton.py`, `advection_char.py` |
| `iterative/` | (no files defined yet) |

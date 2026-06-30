# Numerical Methods

A personal collection of numerical algorithms implemented from scratch in Python.
The goal is mathematical transparency — variable names match the math, loops reveal the structure,
and each function maps directly to the underlying formula.

Not a library. A working notebook.

---

## What's Here

| Module | Status | Algorithms |
|---|---|---|
| `finite_diff/` | Done | 1D/2D BVPs, 5-pt and 9-pt Poisson, Neumann BCs, Richardson extrapolation, Newton nonlinear solver |
| `ode/` | Done | Forward Euler, Backward Euler (linear + Newton), RK23 adaptive, RK4 stability region |
| `integration/` | Done | Trapezoidal, Simpson, Newton-Cotes, adaptive quadrature, Gauss-Legendre, 2D tensor product |
| `interpolation/` | Done | Lagrange interpolation, Chebyshev / equispaced nodes, interval maps |
| `iterative/` | Done | Jacobi, Gauss-Seidel, Newton, GMRES, projection methods |
| `fourier/` | In progress | Spectral differentiation D_N, Fourier interpolant, FFT-based operators |
| `chebyshev/` | Planned | Chebyshev differentiation, Legendre spectral BVP |
| `integral_eq/` | Planned | Nyström, Fredholm 1st and 2nd kind |
| `pde/` | Planned | ADI heat equation, nonlinear BE + Newton, Schrödinger CN, advection |

---

## Install

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Tests

```bash
pytest
```

---

## Example

```python
import numpy as np
from interpolation.lagrange import lagrange_interpolate

f = lambda x: 1 / (1 + 25 * x**2)
x_eval = np.linspace(-1, 1, 200)

x_nodes, f_nodes, x_eval, p_eval = lagrange_interpolate(
    f, -1, 1, N=12, node_type="chebyshev", x_eval=x_eval
)
```

```python
from finite_diff.fd_1d_bvp import fd_matrix_1d
import numpy as np

x, A = fd_matrix_1d(m=50, c=1.0, b=0.0, x_left=0.0, x_right=1.0)
f = np.sin(np.pi * x)
# fold in BCs and solve: A @ u = f
```

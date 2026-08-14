# NumericalMethodsLab

A collection of numerical algorithms implemented from scratch in Python.
The goal is mathematical transparency — variable names match the math, loops reveal the structure,
and each function maps directly to the underlying formula.

Not a library. A working lab notebook.

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
import numpy as np
from finite_diff import fd_bvp_1d

f = lambda x: np.sin(np.pi * x)
x, A, F = fd_bvp_1d(
    N=50,
    a=1.0,
    b=0.0,
    c=0.0,
    f=f,
    x_left=0.0,
    x_right=1.0,
    bc_left=(1.0, 0.0, 0.0),
    bc_right=(1.0, 0.0, 0.0),
)
U = np.linalg.solve(A, F)
```

---

## License

MIT — see [LICENSE](LICENSE).

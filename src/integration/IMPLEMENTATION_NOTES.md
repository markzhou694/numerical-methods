# Numerical Integration — Implementation Notes

Source: NM1 slides `slides_1124.pdf`, `slides_1119.pdf`, `slides_1201.pdf`.

Each function below follows the project conventions:
- plain functions, no classes
- variable names match the math
- explicit loops unless vectorization is structurally transparent
- intermediate variables for every named formula quantity
- no hardcoded test problems inside `src/`

---

## 1. Newton-Cotes Quadrature (`newton_cotes.py`)

### Math

Given n+1 equispaced nodes tᵢ = a + i·(b-a)/n on [a,b]:

```
Î(f) = (b-a) * Σᵢ₌₀ⁿ  λᵢ · f(tᵢ)

where  λᵢ = 1/(b-a) * ∫ₐᵇ Lᵢ(t) dt
```

Lᵢ is the i-th Lagrange cardinal basis on the nodes (see `src/interpolation/lagrange.py`).
The rule is exact for all polynomials of degree ≤ n.

Special cases:
- n=0 (midpoint):    Î(f) = (b-a) · f((a+b)/2)
- n=1 (trapezoidal): Î(f) = (b-a)/2 · (f(a) + f(b))
- n=2 (Simpson):     Î(f) = (b-a)/6 · (f(a) + 4f((a+b)/2) + f(b))

### Suggested signatures

```python
def newton_cotes_weights(n):
    """
    Compute Newton-Cotes weights for n+1 equispaced nodes on [0,1].

    Parameters
    ----------
    n : int — number of subintervals (degree of rule)

    Returns
    -------
    t : (n+1,) array — nodes tᵢ = i/n on [0,1]
    w : (n+1,) array — weights λᵢ summing to 1
    """
    # nodes on [0,1]; map to [a,b] outside this function
    # build weights by integrating each Lagrange basis over [0,1]
    # use scipy.integrate.quad or the closed-form for small n


def newton_cotes_integrate(f, a, b, n):
    """
    Integrate f on [a,b] with the n-th degree Newton-Cotes rule.

    Parameters
    ----------
    f    : callable  f(x) -> scalar or array
    a, b : float     endpoints
    n    : int       degree (0=midpoint, 1=trapezoidal, 2=Simpson, ...)

    Returns
    -------
    I_hat : float   — quadrature estimate
    """
    # 1. get weights on [0,1]
    # 2. map to [a,b]: t_phys = a + t*(b-a)
    # 3. I_hat = (b-a) * sum(w * f(t_phys))
```

### Notes
- For weight computation reuse `lagrange_eval` from `src/interpolation/lagrange.py`
  to evaluate Lᵢ on a fine grid, then integrate numerically — keeps the weight
  formula structurally identical to the formula in the slides.
- Weights for n=1 and n=2 can be hardcoded to double-check: w=[0.5, 0.5] and
  w=[1/6, 4/6, 1/6].

---

## 2. Composite Trapezoidal Rule (`trapezoidal.py`)

### Math

Split [a,b] into m equal subintervals of width h = (b-a)/m.

```
T(h) = h · [ ½f(a) + f(a+h) + f(a+2h) + ... + f(b-h) + ½f(b) ]
     = h · ( ½(f(a) + f(b)) + Σᵢ₌₁ᵐ⁻¹ f(a + i·h) )
```

Error: T(h) - ∫f = (b-a)·h²/12 · f''(τ)  for some τ ∈ [a,b].  (O(h²) globally)

Richardson extrapolation using two step sizes h and h/2:
```
T_Richardson = (4·T(h/2) - T(h)) / 3       # O(h⁴), same as Simpson
```

### Suggested signature

```python
def trapezoidal(f, a, b, m):
    """
    Composite trapezoidal rule on [a,b] with m subintervals.

    Parameters
    ----------
    f    : callable  f(x) -> scalar
    a, b : float     endpoints
    m    : int       number of subintervals

    Returns
    -------
    T : float — quadrature estimate
    """
    h = (b - a) / m
    x = a + np.arange(m + 1) * h       # m+1 nodes

    # T(h) = h * (½f(a) + interior sum + ½f(b))
    T = h * (0.5 * f(x[0]) + np.sum(f(x[1:-1])) + 0.5 * f(x[-1]))
    return T
```

### Notes
- Reuse `src/finite_diff/richardson_extrap.py::richardson` with `stencil_fn =
  lambda h: trapezoidal(f, a, b, int((b-a)/h))` to get the O(h⁴) Richardson
  correction without rewriting the extrapolation logic.

---

## 3. Composite Simpson's Rule (`simpson.py`)

### Math (composite, m must be even)

h = (b-a)/m, nodes xᵢ = a + i·h for i = 0,...,m:

```
S(h) = h/3 · [ f(x₀) + 4f(x₁) + 2f(x₂) + 4f(x₃) + ... + 4f(xₘ₋₁) + f(xₘ) ]
```

Pattern of coefficients: 1, 4, 2, 4, 2, ..., 2, 4, 1.
Error: O(h⁴) globally.

### Suggested signature

```python
def simpson(f, a, b, m):
    """
    Composite Simpson's rule on [a,b] with m subintervals (m must be even).

    Parameters
    ----------
    f    : callable  f(x) -> scalar
    a, b : float     endpoints
    m    : int       number of subintervals (even)

    Returns
    -------
    S : float — quadrature estimate
    """
    assert m % 2 == 0, "m must be even for composite Simpson"
    h = (b - a) / m
    x = a + np.arange(m + 1) * h

    # build weight vector: 1, 4, 2, 4, 2, ..., 4, 1
    w = np.ones(m + 1)
    w[1:-1:2] = 4       # odd indices
    w[2:-2:2] = 2       # even indices (interior)

    S = (h / 3) * np.dot(w, f(x))
    return S
```

---

## 4. Adaptive Quadrature (`adaptive_quad.py`)

### Math

Idea: apply a low-degree rule (e.g., trapezoidal) on each subinterval; estimate
local error by comparing two levels of refinement; subdivide where error is large.

Error estimator for interval [c,d] with midpoint m = (c+d)/2:
```
err_est = |T([c,d]) - (T([c,m]) + T([m,d]))|
```

If err_est < tol · (d-c)/(b-a), accept the interval; otherwise split and recurse.

(The slides call this "a posteriori refinement" — partition where the function is
difficult, not where we guess it might be.)

### Suggested signature

```python
def adaptive_trapezoidal(f, a, b, tol):
    """
    Adaptive composite trapezoidal rule.

    Parameters
    ----------
    f    : callable  f(x) -> scalar
    a, b : float     endpoints
    tol  : float     absolute error tolerance

    Returns
    -------
    I : float   — quadrature estimate
    n_evals : int — number of function evaluations used
    """
    # recursive helper: _adapt(c, d, fc, fd, tol, depth)
    # base case: depth > max_depth or err_est < tol*(d-c)/(b-a)
    # recursive case: split at m, call _adapt on [c,m] and [m,d]
    # count f-evals to demonstrate efficiency vs. fixed grid
```

### Notes
- The recursion can blow the call stack for oscillatory f; a stack-based iterative
  version is safer in practice (push/pop intervals manually).
- For a first implementation, the recursive form is more readable and matches the
  slides' description.

---

## 5. Gauss-Legendre Quadrature (`gauss_legendre.py`)

### Math

Choose n+1 nodes τᵢ = roots of the (n+1)-st Legendre polynomial Pₙ₊₁ on [-1,1].
Weights λᵢ > 0 are set so the rule is exact for all polynomials of degree ≤ 2n+1.

```
∫₋₁¹ f(ξ) dξ  ≈  Σᵢ₌₀ⁿ λᵢ · f(τᵢ)
```

For a general interval [a,b] via affine map ξ = to_standard(x,a,b):
```
∫ₐᵇ f(x) dx  =  (b-a)/2 · Σᵢ₌₀ⁿ λᵢ · f(from_standard(τᵢ, a, b))
```

The (n+1) nodes and weights are available from NumPy without re-deriving them:
`tau, lam = np.polynomial.legendre.leggauss(n+1)`.

### Suggested signature

```python
def gauss_legendre(f, a, b, n):
    """
    (n+1)-point Gauss-Legendre quadrature on [a,b], exact for P_deg ≤ 2n+1.

    Parameters
    ----------
    f    : callable  f(x) -> scalar
    a, b : float     endpoints
    n    : int       degree parameter; uses n+1 nodes

    Returns
    -------
    I_hat : float — quadrature estimate
    """
    from interpolation.interval_map import from_standard

    # nodes and weights on [-1,1]
    tau, lam = np.polynomial.legendre.leggauss(n + 1)

    # map nodes to [a,b]
    x_phys = from_standard(tau, a, b)

    # Jacobian of the affine map
    jac = (b - a) / 2.0

    I_hat = jac * np.dot(lam, f(x_phys))
    return I_hat
```

### Notes
- This directly reuses `from_standard` from `src/interpolation/interval_map.py`.
- The key insight from the slides: by choosing nodes optimally (not equispaced),
  n+1 nodes give twice the order of Newton-Cotes — no extra cost.
- Convergence for smooth f is spectral (exponential in n); for rough f use adaptive
  trapezoidal instead.

---

## 6. 2D Tensor-Product Quadrature (`tensor_product_2d.py`)

### Math

On [a₁,b₁] × [a₂,b₂], use the same 1D rule in each variable independently:

```
∫∫ f(x,y) dx dy  ≈  Σᵢ Σⱼ  wᵢ · wⱼ · f(xᵢ, yⱼ)
```

where {xᵢ, wᵢ} and {yⱼ, wⱼ} are 1D quadrature nodes/weights.

The "mother hat" basis function from the slides:
```
φ(x) = max(1 - |x|, 0)        (piecewise linear, support [-1,1])
```
Used to construct bilinear 2D basis φᵢⱼ(x,y) = φᵢ(x)·φⱼ(y), which gives the
bilinear quadrature formula Î(f) = Σᵢ,ⱼ f(xᵢ,yⱼ)·λᵢ·λⱼ.

Cost: O(n²) in 2D, O(nᵈ) in d dimensions — "curse of dimensionality."

### Suggested signature

```python
def tensor_product_2d(f, a1, b1, a2, b2, quad_fn, n):
    """
    2D tensor-product quadrature using a 1D rule applied in each dimension.

    Parameters
    ----------
    f         : callable  f(x, y) -> scalar
    a1, b1    : float     x-interval endpoints
    a2, b2    : float     y-interval endpoints
    quad_fn   : callable  quad_fn(g, a, b, n) -> (nodes, weights)
                          — a 1D quadrature function returning nodes and weights
    n         : int       rule parameter passed to quad_fn

    Returns
    -------
    I_hat : float — 2D quadrature estimate
    """
    x_nodes, w_x = quad_fn(a1, b1, n)
    y_nodes, w_y = quad_fn(a2, b2, n)

    I_hat = 0.0
    for i, xi in enumerate(x_nodes):
        for j, yj in enumerate(y_nodes):
            # φᵢⱼ contribution: wᵢ · wⱼ · f(xᵢ, yⱼ)
            I_hat += w_x[i] * w_y[j] * f(xi, yj)

    return I_hat
```

### Notes
- Explicit double loop keeps the tensor-product structure visible.
- The `quad_fn` interface: make `gauss_legendre` and `trapezoidal` return
  `(nodes, weights)` as an alternate mode so they can plug into `tensor_product_2d`
  without extra wrappers.

---

## Suggested file layout for `src/integration/`

```
src/integration/
├── __init__.py
├── IMPLEMENTATION_NOTES.md       ← this file
├── newton_cotes.py               ← weights + integrate (uses interval_map)
├── trapezoidal.py                ← composite trapezoidal; T(h) formula
├── simpson.py                    ← composite Simpson; weight pattern
├── adaptive_quad.py              ← adaptive trapezoidal (recursive)
├── gauss_legendre.py             ← leggauss nodes; reuse from_standard
└── tensor_product_2d.py          ← double loop over 1D nodes/weights
```

## Cross-module reuse map

| New file                | Reuses from `src/`                             |
|-------------------------|------------------------------------------------|
| `newton_cotes.py`       | `interpolation.lagrange.lagrange_eval`         |
| `gauss_legendre.py`     | `interpolation.interval_map.from_standard`     |
| `tensor_product_2d.py`  | any 1D quad function returning (nodes, weights)|
| `adaptive_quad.py`      | `trapezoidal` (as the base estimator)          |
| (any)                   | `finite_diff.richardson_extrap.richardson`     |

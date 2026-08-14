import numpy as np

# Adaptive composite trapezoidal quadrature on [a, b].
#
# A posteriori refinement: estimate the local error on [c, d] by comparing
# the coarse panel against the two refined half-panels:
#
#     err_est = | T_coarse([c,d]) - ( T([c,m]) + T([m,d]) ) |
#
# where m = (c+d)/2 and T denotes the single-panel trapezoidal rule.
#
# Accept interval [c, d] when:  err_est < tol * (d-c) / (b-a)
# (scales the tolerance proportionally to the subinterval's length fraction).
#
# Uses an explicit stack instead of recursion to avoid call-stack overflow
# on oscillatory or nearly-singular integrands.
#
# Usage:
#   I, n_evals = adaptive_trapezoidal(f, a, b, tol=1e-6)


def adaptive_trapezoidal(f, a: float, b: float, tol: float, n_max: float = 1e6):
    """
    Adaptive composite trapezoidal rule on [a, b].

    Parameters
    ----------
    f    : callable  f(x) -> scalar
    a, b : float     endpoints
    tol  : float     absolute error tolerance

    Returns
    -------
    I       : float — quadrature estimate
    n_evals : int   — number of f-evaluations used
    """
    n_evals = [0]

    def feval(x):
        n_evals[0] += 1
        return f(x)

    def trap_panel(c, d, fc, fd):
        # single-panel trapezoidal rule using already-known endpoint values
        return 0.5 * (d - c) * (fc + fd)

    fa     = feval(a)
    fb     = feval(b)
    T_full = trap_panel(a, b, fa, fb)

    # stack entries: (c, d, fc, fd, T_coarse)
    stack = [(a, b, fa, fb, T_full)]
    I = 0.0

    while stack:
        if n_evals[0] >= n_max:
           raise RuntimeError(
            f"Maximum number of function evaluations reached: {n_max}"
        )
        c, d, fc, fd, T_cd = stack.pop()

        m  = 0.5 * (c + d)
        fm = feval(m)

        T_left  = trap_panel(c, m, fc, fm)
        T_right = trap_panel(m, d, fm, fd)

        # err_est = |T_coarse - (T_left + T_right)|
        err_est   = abs(T_cd - (T_left + T_right))/3
        local_tol = tol * (d - c) / (b - a)

        if err_est < local_tol:
            I += T_left + T_right
        else:
            # subdivide: push both halves onto the stack
            stack.append((c, m, fc, fm, T_left))
            stack.append((m, d, fm, fd, T_right))

    return I, n_evals[0]


# test function: f(x) = tanh(20x) on [0, 1]
if __name__ == "__main__":
    f = lambda x: np.tanh(20 * x)
    a = 0.0
    b = 1.0
    tol = 1e-7

    I, n_evals = adaptive_trapezoidal(f, a, b, tol)
    print(f"Integral estimate: {I:.10f}, f-evals: {n_evals}")
    # normal trapezoidal rule with 2000 panels
    N = 3000
    h = (b - a) / N
    x = np.linspace(a, b, N + 1)
    T = np.trapezoid(f(x), x)
    print(f"Normal trapezoidal rule with {N} panels: {T:.10f}")

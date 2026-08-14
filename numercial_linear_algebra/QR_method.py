import numpy as np

def find_eigenvalues(A: np.ndarray, num_iters: int = 1000, tol: float = 1e-10):
    """
    Computes the eigenvalues of a matrix A using the QR method.

    The QR method iteratively decomposes the matrix A into a product of an orthogonal matrix Q and an upper triangular matrix R,
    and then updates A as R @ Q. This process converges to a form where the eigenvalues can be read off from the diagonal of A.

    Parameters:
    A (numpy.ndarray): The input square matrix.
    num_iters (int): The maximum number of iterations to perform.
    tol (float): The convergence tolerance.

    Returns:
    eigenvalues (numpy.ndarray): The computed eigenvalues of the matrix A.
    """
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix A must be square.")

    for _ in range(num_iters):
        # Perform QR decomposition
        Q, R = np.linalg.qr(A)
        # Update A
        A = R @ Q

        # Check for convergence by examining the off-diagonal elements
        off_diagonal_norm = np.linalg.norm(A - np.diag(np.diagonal(A)))
        if off_diagonal_norm < tol:
            break

    return np.diagonal(A)
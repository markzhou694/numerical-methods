import numpy as np


def find_eigenvalues(A: np.ndarray, num_iters: int = 1000, tol: float = 1e-10, QR_method = np.linalg.qr):
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
        Q, R = QR_method(A)
        # Update A
        A = R @ Q

        # Check for convergence by examining the off-diagonal elements
        off_diagonal_norm = np.linalg.norm(A - np.diag(np.diagonal(A)))
        if off_diagonal_norm < tol:
            break

    return np.diagonal(A)

if __name__ == "__main__":
    # Example usage
    from QR import householder_qr, modified_gram_schmidt_qr
    from finite_diff import fd_bvp_1d
    A = fd_bvp_1d(N=5,a = 2)[1]
    eigenvalue = find_eigenvalues(A)
    eigenvalue_2 = find_eigenvalues(A, QR_method= householder_qr)
    eigenvalue_3 = find_eigenvalues(A, QR_method= modified_gram_schmidt_qr)
    print("Eigenvalues:", eigenvalue)
    print("Eigenvalues by householder:", eigenvalue_2)
    print("Eigenvalues by mgs:", eigenvalue_3)


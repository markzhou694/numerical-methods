import numpy as np

def power_method(A: np.ndarray, num_iters: int = 1000, tol: float = 1e-10):
    """
    Computes the dominant eigenvalue and corresponding eigenvector of a matrix A using the power method.

    The desired square matrix A should have a unique dominant eigenvalue (the eigenvalue with the largest absolute value). 
    The power method iteratively refines an initial guess for the dominant eigenvector and estimates the corresponding eigenvalue.

    Parameters:
    A (numpy.ndarray): The input square matrix.
    num_iters (int): The maximum number of iterations to perform.
    tol (float): The convergence tolerance.

    Returns:
    eigenvalue (float): The dominant eigenvalue of the matrix A.
    eigenvector (numpy.ndarray): The corresponding eigenvector of the dominant eigenvalue.
    """
    n, m = A.shape
    if n != m:
        raise ValueError("Matrix A must be square.")

    # Initialize a random vector, # won't converge if the initial vector is orthogonal to the dominant eigenvector
    b_k = np.random.rand(n)

    for _ in range(num_iters):
        # Calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # Re-normalize the vector
        b_k1 = b_k1 / np.linalg.norm(b_k1)

        # Check for convergence
        if np.linalg.norm(b_k1 - b_k) < tol:
            break

        b_k = b_k1

    # Rayleigh quotient to estimate the dominant eigenvalue
    eigenvalue = np.dot(b_k.T, np.dot(A, b_k)) / np.dot(b_k.T, b_k)
    
    return eigenvalue, b_k


if __name__ == "__main__":
    # Example usage
    A = np.array([[4, 1], [2, 3]])
    eigenvalue, eigenvector = power_method(A)
    print("Dominant Eigenvalue:", eigenvalue)
    print("Corresponding Eigenvector:", eigenvector)
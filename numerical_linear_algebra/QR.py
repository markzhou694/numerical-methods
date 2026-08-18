import numpy as np

def householder_qr(A):
    """
    Compute the QR factorization of A using Householder reflections.

    Parameters
    ----------
    A : ndarray
        Input matrix of shape (n, m).

    Returns
    -------
    Q : ndarray
        Orthogonal matrix of shape (n, n).
    R : ndarray
        Upper triangular / upper trapezoidal matrix.

    The method constructs Householder reflectors H_i such that

        R = H_k ... H_2 H_1 A

    and therefore

        A = Q R,

    where

        Q = H_1 H_2 ... H_k.

    Each Householder reflector eliminates the entries below the
    diagonal in one column.
    """

    A = A.astype(float).copy()

    n, m = A.shape

    # Initialize Q as identity
    Q = np.eye(n)

    # Only need to eliminate below min(n, m) diagonal entries
    for i in range(min(n, m)):

        # Take the part of column i below and including diagonal
        x = A[i:, i].reshape(-1, 1)

        # If x is already zero, nothing to do
        if np.linalg.norm(x) == 0:
            continue

        # First coordinate vector e1
        e1 = np.zeros((len(x), 1))
        e1[0, 0] = 1

        # Choose sign to avoid cancellation
        sign = 1 if x[0, 0] >= 0 else -1

        v = x + sign * np.linalg.norm(x) * e1

        # Normalize is optional, but makes the formula simpler
        v = v / np.linalg.norm(v)

        # Full Householder matrix
        H_i = np.eye(n)

        # Apply the reflection only to the remaining submatrix
        H_i[i:, i:] -= 2 * (v @ v.T)

        # Update R
        A = H_i @ A

        # Accumulate Q
        Q = Q @ H_i.T

    return Q, A



def modified_gram_schmidt_qr(A):
    """
    Compute the reduced QR factorization of A using
    Modified Gram-Schmidt.

    Parameters
    ----------
    A : ndarray
        Input matrix of shape (n, m), usually with n >= m.

    Returns
    -------
    Q : ndarray
        Matrix of shape (n, m) whose columns are orthonormal.

    R : ndarray
        Upper triangular matrix of shape (m, m).

    The algorithm processes one column at a time and immediately
    removes its components in the directions of the previously
    computed orthonormal vectors.

    A = Q @ R
    """

    A = A.astype(float).copy()

    n, m = A.shape

    # Reduced QR:
    # Q has m orthonormal columns
    # R is m x m upper triangular
    Q = np.zeros((n, m))
    R = np.zeros((m, m))

    for i in range(m):

        # Start from column a_i
        a = A[:, i].copy()

        # Remove components along q_0, ..., q_{i-1}
        for j in range(i):

            R[j, i] = Q[:, j].T @ a

            a = a - R[j, i] * Q[:, j]

        # Length of the remaining orthogonal component
        R[i, i] = np.linalg.norm(a)

        # Check for linearly dependent columns
        if R[i, i] == 0:
            raise ValueError("Columns of A are linearly dependent.")

        # Normalize to obtain q_i
        Q[:, i] = a / R[i, i]

    return Q, R
    
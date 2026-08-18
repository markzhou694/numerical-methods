import numpy as np


def LU_factorization(A: np.ndarray, pivot: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform LU factorization of a square matrix A using Doolittle's method.

    Parameters
    ----------
    A : np.ndarray
        The input square matrix to be factored.

    Returns
    -------
    L : np.ndarray
        Lower triangular matrix with unit diagonal.
    U : np.ndarray
        Upper triangular matrix.
    P : np.ndarray
        Permutation matrix (identity if pivoting is not used).
    """
    if pivot:

        n = A.shape[0]
        P = np.eye(n)
        L = np.zeros_like(A)
        U = A.copy()

        for j in range(n):
            # Pivoting: find the index of the maximum element in the current column
            max_index = np.argmax(np.abs(U[j:n, j])) + j
            if max_index != j:
                # Swap rows in U
                U[[j, max_index], :] = U[[max_index, j], :]
                # Swap rows in P
                P[[j, max_index], :] = P[[max_index, j], :]
                # Swap rows in L (only the part below the diagonal)
                if j > 0:
                    L[[j, max_index], :j] = L[[max_index, j], :j]

            for i in range(j + 1, n):
                L[i, j] = U[i, j] / U[j, j]
                U[i, :] -= L[i, j] * U[j, :]

        np.fill_diagonal(L, 1)  # Set the diagonal of L to 1
        return L, U, P

    else:
        n = A.shape[0]
        L = np.eye(n)
        U = np.zeros_like(A)

        for j in range(n):
            for i in range(j + 1):
                U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
            for i in range(j + 1, n):
                L[i, j] = (A[i, j] - np.dot(L[i, :j], U[:j, j])) / U[j, j]

        return L, U , np.eye(n)  # Return identity matrix as P since pivoting is not implemented



if __name__ == "__main__": 
    A = np.array([[4, 3,3,3], [6, 3,12,12] ,[75,34,3.12,11],[4, 4, 4, 4]], dtype=float)
    L, U, P = LU_factorization(A, pivot=True)
    print("L:\n", L)
    print("U:\n", U)
    print("P:\n", P)
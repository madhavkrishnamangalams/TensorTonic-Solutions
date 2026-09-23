import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transposed matrix as a NumPy array.
    """
    # Write code here
    A = np.asarray(A)
    N, D = A.shape
    out = np.zeros((D, N), dtype = A.dtype)
    for i in range(N):
        for j in range(D):
            out[j,i] = A[i,j]
    return out

import numpy as np


def householder_reduction(A):
    """Perform Golub-Kahan bidiagonalization of matrix A using series
    of orthogonal transformations.

    Args:
        A (numpy.ndarray): 2-dim array representing input matrix of size m by n, where m > n.

    Returns:
        Three 2-dim numpy arrays which correspond to matrices  U, B and V.T
        such that A = UB(V.T), U and V are orthogonal and B is upper bidiagonal.
    """


    # initialize matrices
    B = np.copy(A)
    m, n = B.shape
    U = np.eye(m)
    V = np.eye(n)
    U_temp = np.eye(m)
    V_temp = np.eye(n)

    for k in range(n):

        # zero out elements under diagonal element in k-th column
        u = np.copy(B[k:m, k])
        u[0] += np.sign(u[0]) * np.linalg.norm(u)
        u = u / np.linalg.norm(u)
        U_temp[k:m, k:m] = np.eye(m - k) - 2 * np.outer(u, u)
        # update matrix U
        U[k:m, :] = np.matmul(U_temp[k:m, k:m], U[k:m, :])
        B[k:m, k:n] = np.matmul(U_temp[k:m, k:m], B[k:m, k:n])

        # zero out elements to the right of right neighbour of diagonal entry in k-th row
        if k <= n - 2:
            v = np.copy(B[k, (k + 1): n])
            v[0] += np.sign(v[0]) * np.linalg.norm(v)
            v = v / np.linalg.norm(v)
            V_temp[k + 1:n, k + 1:n] = np.eye(n - k - 1) - 2 * np.outer(v, v)
            # update matrix V.T
            V[:, k + 1:n] = np.matmul(V[:, k + 1:n], V_temp[k + 1:n, k + 1:n].T)
            B[k:m, (k + 1):n] = np.matmul(B[k:m, (k + 1):n], V_temp[k + 1:n, k + 1: n].T)

    return U.T, B, V.T

import numpy as np


def householder_reduction(A):
    """Perform Golub-Kahan bidiagonalization of matrix A using series
    of orthogonal transformations.

    Args:
        A (numpy.ndarray): 2-dim array representing input matrix of size m by n, where m > n.

    Returns:
        Three 2-dim numpy arrays which correspond to matrices  U, B and V
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

    return U.T, B, V


def two_dim_evs(A):
    tr = A[0, 0] + A[1, 1]
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    d = np.sqrt(tr ** 2 - 4 * det)
    return (tr + d) / 2, (tr - d) / 2


def golub_kahan_step(B, U, V, p, q):
    m, n = B.shape
    C = np.zeros((2, 2))
    C[0, 0] = np.dot(B[p:n - q, n - q - 2], B[p:n - q, n - q - 2])
    C[1, 0] = C[0, 1] = np.dot(B[p:n - q, n - q - 2], B[p:n - q, n - q - 1])
    C[1, 1] = np.dot(B[p:n - q, n - q - 1], B[p:n - q, n - q - 1])

    lambda1, lambda2 = two_dim_evs(C)
    mu = lambda1 if (np.abs(lambda1 - C[1, 1]) < np.abs(lambda2 - C[1, 1])) else lambda2
    alpha, beta = B[p, p] ** 2 - mu, B[p, p] * B[p, p + 1]

    R = np.zeros((2, 2))
    for k in range(p, n - q - 1):
        r = np.sqrt(alpha ** 2 + beta ** 2)
        c, s = alpha / r, -beta / r

        R[0, 0] = c
        R[0, 1] = s
        R[1, 0] = -s
        R[1, 1] = c
        B[:, k:k + 2] = np.matmul(B[:, k:k + 2], R)
        V[k:k + 2, :] = np.matmul(R.T, V[k:k + 2, :])

        alpha, beta = B[k, k], B[k + 1, k]
        r = np.sqrt(alpha ** 2 + beta ** 2)
        c, s = alpha / r, -beta / r

        R[0, 0] = c
        R[0, 1] = s
        R[1, 0] = -s
        R[1, 1] = c

        B[k:k + 2, :] = np.matmul(R.T, B[k:k + 2, :])
        U[:, k:k + 2] = np.matmul(U[:, k:k + 2], R)
        if k < n - q - 2:
            alpha, beta = B[k, k + 1], B[k, k + 2]
    return U, B, V


def givens_push(U, B, p, q):
    m, n = B.shape
    for i in range(p, n - q - 1):
        if B[i, i] == 0:
            for j in range(i + 1, n - q):
                alpha, beta = B[i, j], B[j, j]
                r = np.sqrt(alpha ** 2 + beta ** 2)
                c, s = beta / r, -alpha / r

                temp = c * B[i] + s * B[j]
                B[j] = -s * B[i] + c * B[j]
                B[i] = temp

                temp = c * U[:, i] + s * U[:, j]
                U[:, j] = -s * U[:, i] + c * U[:, j]
                U[:, i] = temp
    return U, B


def svd(A):
    m, n = A.shape
    epsilon = 1e-3
    U, B, V = householder_reduction(A)
    V = V.T
    while True:
        for i in range(n - 1):
            if np.abs(B[i, i + 1]) <= epsilon * (np.abs(B[i, i]) + np.abs(B[i + 1, i + 1])):
                B[i, i + 1] = 0

        q = 0
        while q < n - 1 and B[n - q - 2, n - q - 1] == 0:
            q += 1

        q = q if q != (n - 1) else (q + 1)

        if q == n:
            break

        r = 0
        while r < n - q - 1 and B[n - q - r - 2, n - q - r - 1]:
            r += 1
        r += 1

        p = n - q - r
        if any([B[i, i] for i in range(p, n - q)]):
            U, B, V = golub_kahan_step(B, U, V, p, q)
        else:
            U, B = givens_push(U, B, p, q)

    for i in range(n):
        if B[i, i] < 0:
            U[:, i] *= -1
            B[i, i] *= -1
    return U, B, V

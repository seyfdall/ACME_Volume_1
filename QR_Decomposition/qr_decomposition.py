# qr_decomposition.py
"""Volume 1: The QR Decomposition.
<Name> Dallin Seyfried
<Class> Math 345 002
<Date> 10/18/22
"""

import numpy as np
from scipy import linalg as la

# Problem 1
def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """

    # Setup initial variables
    m,n = np.shape(A)
    Q = np.copy(A)
    R = np.zeros((n, n))

    # Cycle through to compute reduced QR Decomposition
    for i in range(n):
        R[i, i] = la.norm(Q[:, i])
        Q[:, i] = Q[:, i] / R[i, i]
        for j in range(i + 1, n):
            R[i, j] = np.transpose(Q[:, j]) @ Q[:, i]
            Q[:, j] = Q[:, j] - R[i, j] * Q[:, i]
    return Q, R

# Problem 2
def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    return abs(np.prod(np.diag(qr_gram_schmidt(A)[1])))


# Problem 3
def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    # Generate the Q, R, and y to be used with back substitution
    Q, R = qr_gram_schmidt(A)
    y = np.transpose(Q) @ b

    # Back substitute Algorithm
    n = len(y)
    x = np.zeros(n).astype('float')
    for i in range(n):
        j = n - (i + 1)
        x[j] = (y[j] - np.dot(x, R[j, :])) / R[j, j]

    return x


# Problem 4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    # Setup initial values and the lambda function
    m, n = np.shape(A)
    R = np.copy(A)
    Q = np.identity(m)

    sign = lambda x: 1 if x >= 0 else -1

    # Cycle through the matrices
    for k in range(n):
        U = np.copy(R[k:, k])
        U[0] = U[0] + sign(U[0]) * la.norm(U)
        U = U / la.norm(U)
        R[k:, k:] = R[k:, k:] - 2 * np.outer(U, np.transpose(U) @ R[k:, k:])
        Q[k:, :] = Q[k:, :] - 2 * np.outer(U, np.transpose(U) @ Q[k:, :])

    return np.transpose(Q), R


# Problem 5
def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    # Setup initial values and the lambda function
    m, n = np.shape(A)
    H = np.copy(A)
    Q = np.identity(m)

    sign = lambda x: 1 if x >= 0 else -1

    # Cycle to put the matrix in Hessenberg form and find Q such that H = QHQ^T
    for k in range(n - 2):
        U = np.copy(H[k + 1:, k])
        U[0] = U[0] + sign(U[0]) * la.norm(U)
        U = U / la.norm(U)
        H[k + 1:, k:] = H[k + 1:, k:] - 2 * np.outer(U, np.transpose(U) @ H[k + 1:, k:])
        H[:, k + 1:] = H[:, k + 1:] - 2 * np.outer(H[:, k + 1:] @ U, np.transpose(U))
        Q[k + 1:, :] = Q[k + 1:, :] - 2 * np.outer(U, np.transpose(U) @ Q[k + 1:, :])

    return H, np.transpose(Q)


def test_prob1():
    """Test For The QR Decomposition for Modified Gram-Schmidt"""
    A = np.random.random((6, 4))
    Q, R = la.qr(A, mode="economic")
    print(A.shape, Q.shape, R.shape)

    Q_1, R_1 = qr_gram_schmidt(A)
    print(A.shape, Q_1.shape, R_1.shape)


def test_prob2():
    """Test for calculating the determinant of A"""
    A = np.random.random((6, 6))
    print(la.det(A))
    print(abs_det(A))


def test_prob3():
    """Test Solving linear system"""
    A = np.random.random((6, 6))
    b = np.random.random(6)
    print(solve(A, b))
    print(la.solve(A, b))


def test_prob4():
    """Test Houeholder"""
    A = np.random.random((6, 6))
    print(qr_householder(A))


def test_prob5():
    """Test Houeholder"""
    A = np.random.random((8, 8))
    print('\n\n', hessenberg(A)[1])
    print('\n\n', la.hessenberg(A, calc_q=True)[1])

# linear_systems.py
"""Volume 1: Linear Systems.
<Name> Dallin Seyfried
<Class> Volume 1 Math 345 Section 2
<Date> 10/11/22
"""
import time

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as spla
import pytest

# Problem 1
def ref(A):
    """Reduce the square matrix A to REF. You may assume that A is invertible
    and that a 0 will never appear on the main diagonal. Avoid operating on
    entries that you know will be 0 before and after a row operation.

    Parameters:
        A ((n,n) ndarray): The square invertible matrix to be reduced.

    Returns:
        ((n,n) ndarray): The REF of A.
    """
    # Copy the original array
    ref_A = np.copy(A).astype(float)

    # Cycle through each column and row skipping entries of zero and reducing rows
    for col in range(len(ref_A)):
        for row in range(col + 1, len(ref_A[0])):
            if ref_A[row, col] != 0:
                ref_A[row, col:] -= (ref_A[row, col] / ref_A[col, col]) * ref_A[col, col:]
    return ref_A


# Problem 2
def lu(A):
    """Compute the LU decomposition of the square matrix A. You may
    assume that the decomposition exists and requires no row swaps.

    Parameters:
        A ((n,n) ndarray): The matrix to decompose.

    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """
    # Get the shape of A, and initialize L and U
    m, n = np.shape(A)
    U = np.copy(A).astype(float)
    L = np.identity(len(A))

    # Cycle through each element of L and U and update them
    for j in range(n):
        for i in range(j + 1, m):
            L[i, j] = U[i, j] / U[j, j]
            U[i, j:] = U[i, j:] - L[i, j] * U[j, j:]

    return L, U


# Problem 3
def solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may again assume that no row swaps are required.

    Parameters:
        A ((n,n) ndarray)
        b ((n,) ndarray)

    Returns:
        x ((m,) ndarray): The solution to the linear system.
    """

    # LU Decomposition on A
    L, U = lu(A)
    b = np.array(b).astype("float")
    n = len(b)

    # Find y
    y = np.zeros(n).astype('float')
    for i in range(n):
        y[i] = (b[i] - np.dot(y, L[i, :])) / L[i, i]

    # Find x
    n = len(y)
    x = np.zeros(n).astype('float')
    for i in range(n):
        j = n - (i + 1)
        x[j] = (y[j] - np.dot(x, U[j, :])) / U[j, j]

    return x


# Problem 4
def prob4():
    """Time different scipy.linalg functions for solving square linear systems.

    For various values of n, generate a random nxn matrix A and a random
    n-vector b using np.random.random(). Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Invert A with la.inv() and left-multiply the inverse to b.
        2. Use la.solve().
        3. Use la.lu_factor() and la.lu_solve() to solve the system with the
            LU decomposition.
        4. Use la.lu_factor() and la.lu_solve(), but only time la.lu_solve()
            (not the time it takes to do the factorization).

    Plot the system size n versus the execution times. Use log scales if
    needed.
    """

    # Generate domain and codomain arrays
    domain = [2**x for x in range(1, 12)]
    approach_1_times = [0]
    approach_2_times = [0]
    approach_3_times = [0]
    approach_4_times = [0]

    # Cycle through each time in the domain
    for x in domain:
        b = np.random.random(x)
        A = np.random.random((x, x))

        # Approach 1 with la.inv
        start = time.perf_counter()
        A_inv = la.inv(A)
        np.dot(A_inv, b)
        finish = time.perf_counter()
        approach_1_times.append(finish - start)

        # Approach 2 with la.solve
        start = time.perf_counter()
        la.solve(A, b)
        finish = time.perf_counter()
        approach_2_times.append(finish - start)

        # Approach 3 with la.lu_factor and la.lu_solve
        start = time.perf_counter()
        lu, piv = la.lu_factor(A)
        la.lu_solve((lu, piv), b)
        finish = time.perf_counter()
        approach_3_times.append(finish - start)

        # Approach 4 similar to 3 but time only lu_solve
        lu, piv = la.lu_factor(A)
        start = time.perf_counter()
        la.lu_solve((lu, piv), b)
        finish = time.perf_counter()
        approach_4_times.append(finish - start)

    domain.insert(0, 0)

    # Plot the times
    plt.loglog(domain, approach_1_times, 'g-', label="Approach 1", base=2)
    plt.loglog(domain, approach_2_times, 'r-', label="Approach 2", base=2)
    plt.loglog(domain, approach_3_times, 'b-', label="Approach 3", base=2)
    plt.loglog(domain, approach_4_times, 'y-', label="Approach 4", base=2)
    plt.xlabel("n")
    plt.ylabel("Times")
    plt.title("Approach Times")
    plt.legend(loc="upper left")
    plt.show()


# Problem 5
def prob5(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """

    # Building the B matrix following the DIA Format
    diagonals = [[1 for i in range(n - 1)], [-4 for i in range(n)], [1 for i in range(n - 1)]]
    offsets = [-1, 0, 1]
    B = sparse.diags(diagonals, offsets, shape=(n,n)).toarray()

    # Building the sparse block matrix A following the BSR Format
    A = sparse.block_diag((B for i in range(n)))
    if n > 1:
        A.setdiag([1 for i in range(n**2 - n)], n)
        A.setdiag([1 for i in range(n**2 - n)], -n)

    return sparse.coo_matrix(A)

# Problem 6
def prob6():
    """Time regular and sparse linear system solvers.

    For various values of n, generate the (n**2,n**2) matrix A described of
    prob5() and vector b of length n**2. Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Convert A to CSR format and use scipy.sparse.linalg.spsolve()
        2. Convert A to a NumPy array and use scipy.linalg.solve().

    In each experiment, only time how long it takes to solve the system (not
    how long it takes to convert A to the appropriate format). Plot the system
    size n**2 versus the execution times. As always, use log scales where
    appropriate and use a legend to label each line.
    """

    # Generate domain and codomain arrays
    domain = [2 ** x for x in range(1, 8)]
    approach_1_times = [0]
    approach_2_times = [0]

    # Cycle through each option
    for n in domain:
        b = np.random.random(n**2)
        A = prob5(n)

        # Approach 1 with CSR Format
        Acsr = A.tocsr()
        start = time.perf_counter()
        spla.spsolve(Acsr, b)
        finish = time.perf_counter()
        approach_1_times.append(finish - start)

        # Approach 2 with la.solve
        Anum = A.toarray()
        start = time.perf_counter()
        la.solve(Anum, b)
        finish = time.perf_counter()
        approach_2_times.append(finish - start)

    domain.insert(0, 0)

    # Plot the times
    plt.loglog(domain, approach_1_times, 'g-', label="CSR", base=2)
    plt.loglog(domain, approach_2_times, 'r-', label="Numpy", base=2)
    plt.xlabel("n")
    plt.ylabel("Times")
    plt.title("Regular and Sparse linear Solvers Times")
    plt.legend(loc="upper left")
    plt.show()


def test_ref():
    """Testing Problem 1 - reduced echelon form"""
    A = np.array([[1, 1, 1, 1],
                  [1, 4, 2, 3],
                  [4, 7, 8, 9],
                  [0, 0, 0, 1]], dtype=np.float)
    ref_A = ref(A)
    print("Ref_A")
    print(ref_A)
    print("NP array")
    print(np.array([[1, 1, 1, 1],
                      [0, 3, 1, 2],
                      [0, 0, 3, 3],
                      [0, 0, 0, 1]], dtype=np.float))


def test_LU():
    """Testing Problem 2 - LU Decomposition"""
    A = np.array([[1, 1, 1, 1],
                  [1, 4, 2, 3],
                  [4, 7, 8, 9],
                  [0, 0, 0, 1]], dtype=np.float)
    L, U = lu(A)
    A_test = L @ U
    print("\nA")
    print(A)
    print("L * U")
    print(A_test)


def test_LU_solve():
    """Testing Problem 3 - Solve LU Decomposition"""
    A = np.array([[3, 1, -2], [1.5, 2, -5], [2, -4, 1]], dtype=np.float)
    b = np.array([1.1, 3, -2], dtype=np.float)
    x = solve(A, b)
    print('\n', x)


def test_scipy_graph():
    """Testing Problem 4 - Time the scipy stuff"""
    prob4()


def test_scipy_sparse_matrix():
    """Testing Problem 5 - Scipy Block Matrix stuff"""
    A = prob5(5)
    plt.spy(A, markersize=1)
    plt.show()


def test_reg_sparse_solvers():
    """Testing Problem 6 - Write a function that times regular
    and sparse linear system solvers"""
    prob6()



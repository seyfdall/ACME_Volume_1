# iterative_solvers.py
"""Volume 1: Iterative Solvers.
<Name> Dallin Seyfried
<Class> 001
<Date> 04/17/2023
"""

import numpy as np
import matplotlib.pyplot as plt


# Helper function
def diag_dom(n, num_entries=None):
    """Generate a strictly diagonally dominant (n, n) matrix.

    Parameters:
        n (int): The dimension of the system.
        num_entries (int): The number of nonzero values.
            Defaults to n^(3/2)-n.

    Returns:
        A ((n,n) ndarray): A (n, n) strictly diagonally dominant matrix.
    """
    if num_entries is None:
        num_entries = int(n**1.5) - n
    A = np.zeros((n,n))
    rows = np.random.choice(np.arange(0,n), size=num_entries)
    cols = np.random.choice(np.arange(0,n), size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)
    for i in range(num_entries):
        A[rows[i], cols[i]] = data[i]
    for i in range(n):
        A[i,i] = np.sum(np.abs(A[i])) + 1
    return A


# Problems 1 and 2
def jacobi(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Jacobi Method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        b ((n ,) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
    """
    # Break A down into its component matrices
    n = A.shape[0]
    D_inv = np.diag(1 / np.diagonal(A))

    # Iterate until maxiters or tolerance is reached
    x_new = np.zeros(n)
    abs_err = []
    for k in range(maxiter):
        x_old = x_new.copy()
        x_new = x_old + D_inv @ (b - A @ x_old)
        if np.linalg.norm(x_new - x_old) < tol:
            break
        # Track absolute error at each step
        abs_err.append(np.linalg.norm(A @ x_new - b, ord=np.inf))

    # Plot convergence of jacobi
    if plot:
        plt.semilogy(abs_err)
        plt.title("Convergence of Jacobi Method")
        plt.xlabel("Iteration")
        plt.ylabel("Absolute Error of Approximation")
        plt.tight_layout()
        plt.show()

    return x_new


# Test Problem 1 & 2
def test_jacobi():
    n = 10
    A = diag_dom(n)
    b = np.random.random(n)
    x = jacobi(A, b, plot=True)
    print('\n')
    print(A @ x)
    print(b)
    print(np.allclose(A @ x, b))


# Problem 3
def gauss_seidel(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Gauss-Seidel Method.

    Parameters:
        A ((n, n) ndarray): A square matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.
        plot (bool): If true, plot the convergence rate of the algorithm.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    # Break A down into its component matrices
    n = A.shape[0]
    D_inv = 1 / np.diagonal(A)

    # Iterate until maxiters or tolerance is reached
    x_new = np.zeros(n)
    abs_err = []
    for k in range(maxiter):
        x_old = x_new.copy()
        # Use Gauss-Siedel's method to simplify iteration
        for i in range(n):
            x_new[i] = x_old[i] + D_inv[i] * (b[i] - A[i, :] @ x_old)
        if np.linalg.norm(x_new - x_old) < tol:
            break
        # Track absolute error at each step
        abs_err.append(np.linalg.norm(A @ x_new - b, ord=np.inf))

    # Plot the convergence of Guass-Siedel
    if plot:
        plt.semilogy(abs_err)
        plt.title("Convergence of Gauss-Siedel Method")
        plt.xlabel("Iteration")
        plt.ylabel("Absolute Error of Approximation")
        plt.tight_layout()
        plt.show()

    return x_new


# Test Problem 3
def test_problem_3():
    n = 10
    A = diag_dom(n)
    b = np.random.random(n)
    x = gauss_seidel(A, b, plot=True)
    print('\n')
    print(A @ x)
    print(b)
    print(np.allclose(A @ x, b))


# Problem 4
def gauss_seidel_sparse(A, b, tol=1e-8, maxiter=100):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse CSR matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    raise NotImplementedError("Problem 4 Incomplete")


# Problem 5
def sor(A, b, omega, tol=1e-8, maxiter=100):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse matrix.
        b ((n, ) Numpy Array): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def hot_plate(n, omega, tol=1e-8, maxiter=100, plot=False):
    """Generate the system Au = b and then solve it using sor().
    If show is True, visualize the solution with a heatmap.

    Parameters:
        n (int): Determines the size of A and b.
            A is (n^2, n^2) and b is one-dimensional with n^2 entries.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The iteration tolerance.
        maxiter (int): The maximum number of iterations.
        plot (bool): Whether or not to visualize the solution.

    Returns:
        ((n^2,) ndarray): The 1-D solution vector u of the system Au = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of computed iterations in SOR.
    """
    raise NotImplementedError("Problem 6 Incomplete")


# Problem 7
def prob7():
    """Run hot_plate() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiter = 1000 with A and b generated with n=20. Plot the iterations
    computed as a function of omega.
    """
    raise NotImplementedError("Problem 7 Incomplete")

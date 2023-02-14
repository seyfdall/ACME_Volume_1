# condition_stability.py
"""Volume 1: Conditioning and Stability.
<Name> Dallin Seyfried
<Class> 001
<Date> 2/7/23
"""
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
from scipy import linalg as la


# Problem 1
def matrix_cond(A):
    """Calculate the condition number of A with respect to the 2-norm."""
    sing_vals = la.svdvals(A)

    # If the smallest is 0 return np.inf
    if sing_vals[-1] == 0:
        return np.inf

    return sing_vals[0] / sing_vals[-1]


def test_matrix_cond():
    A = np.array([[10, 14], [13, 12]])
    assert abs(matrix_cond(A) - np.linalg.cond(A)) < 1e-4, "Matrix Condition fails for A"

    B = la.qr(A)[0]
    assert abs(matrix_cond(B) - 1) < 1e-4, "Matrix Condition fails for Orthonormal matrix"


# Problem 2
def prob2():
    """Randomly perturb the coefficients of the Wilkinson polynomial by
    replacing each coefficient c_i with c_i*r_i, where r_i is drawn from a
    normal distribution centered at 1 with standard deviation 1e-10.
    Plot the roots of 100 such experiments in a single figure, along with the
    roots of the unperturbed polynomial w(x).

    Returns:
        (float) The average absolute condition number.
        (float) The average relative condition number.
    """
    w_roots = np.arange(1, 21)

    # Get the exact Wilkinson polynomial coefficients using SymPy.
    x, i = sy.symbols('x i')
    w = sy.poly_from_expr(sy.product(x-i, (i, 1, 20)))[0]
    w_coeffs = np.array(w.all_coeffs())
    perturbed_roots = np.array([])
    perturbed_coeffs = np.array([])
    h = np.array([])

    # Perturb the coefficients using the normal distribution and find new roots
    for _ in range(100):
        new_h = np.random.normal(1, 1e-10, len(w_coeffs))
        new_coeffs = w_coeffs * new_h
        h = np.concatenate((h, new_h))
        perturbed_coeffs = np.concatenate((perturbed_coeffs, new_coeffs))
        perturbed_roots = np.concatenate((perturbed_roots, np.roots(np.poly1d(new_coeffs))))

    # Plot the roots in the complex plane
    plt.scatter(perturbed_roots.real, perturbed_roots.imag, marker='.', label="perturbed")
    plt.scatter(w_roots.real, w_roots.imag, marker="o", label="Original")
    plt.ylabel("Imaginary Axis")
    plt.xlabel("Real Axis")
    plt.legend()
    plt.show()

    # Return absolute and relative condition numbers
    perturbed_roots = np.reshape(perturbed_roots, (100, 20))
    abs_cond = la.norm(perturbed_roots - w_roots, np.inf) / la.norm(h, np.inf)
    rel_cond = abs_cond * la.norm(w_coeffs, np.inf) / la.norm(w_roots, np.inf)

    return abs_cond, rel_cond


# Test problem 2
def test_prob2():
    print(prob2())


# Helper function
def reorder_eigvals(orig_eigvals, pert_eigvals):
    """Reorder the perturbed eigenvalues to be as close to the original eigenvalues as possible.
    
    Parameters:
        orig_eigvals ((n,) ndarray) - The eigenvalues of the unperturbed matrix A
        pert_eigvals ((n,) ndarray) - The eigenvalues of the perturbed matrix A+H
        
    Returns:
        ((n,) ndarray) - the reordered eigenvalues of the perturbed matrix
    """
    n = len(pert_eigvals)
    sort_order = np.zeros(n).astype(int)
    dists = np.abs(orig_eigvals - pert_eigvals.reshape(-1,1))
    for _ in range(n):
        index = np.unravel_index(np.argmin(dists), dists.shape)
        sort_order[index[0]] = index[1]
        dists[index[0],:] = np.inf
        dists[:,index[1]] = np.inf
    return pert_eigvals[sort_order]


# Problem 3
def eig_cond(A):
    """Approximate the condition numbers of the eigenvalue problem at A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) The absolute condition number of the eigenvalue problem at A.
        (float) The relative condition number of the eigenvalue problem at A.
    """
    # Randomly draw for real and imaginary distributions to form H
    reals = np.random.normal(0, 1e-10, A.shape)
    imags = np.random.normal(0, 1e-10, A.shape)
    H = reals + 1j*imags

    # Find Eigenvalues of A and A + H
    A_eigs = la.eigvals(A)
    A_H_eigs = reorder_eigvals(A_eigs, la.eigvals(A + H))

    # Compute the absolute and relative condition numbers
    abs_cond = la.norm(A_eigs - A_H_eigs, ord=2) / la.norm(H, ord=2)
    rel_cond = la.norm(A, ord=2) / la.norm(A_eigs, ord=2) * abs_cond

    return abs_cond, rel_cond


# Test problem 3
def test_eig_cond():
    A = np.random.rand(3, 3)
    print(eig_cond(A))


# Problem 4
def prob4(domain=[-100, 100, -100, 100], res=50):
    """Create a grid [x_min, x_max] x [y_min, y_max] with the given resolution. For each
    entry (x,y) in the grid, find the relative condition number of the
    eigenvalue problem, using the matrix   [[1, x], [y, 1]]  as the input.
    Use plt.pcolormesh() to plot the condition number over the entire grid.

    Parameters:
        domain ([x_min, x_max, y_min, y_max]):
        res (int): number of points along each edge of the grid.
    """
    # Create grid
    x = np.linspace(domain[0], domain[1], res)
    y = np.linspace(domain[2], domain[3], res)
    X, Y = np.meshgrid(x, y)
    Z = np.empty_like(X)

    # Find and store the relative condition numnbers at each point in the meshgrid
    for i in range(len(x)):
        for j in range(len(y)):
            A = np.array([[1, x[i]], [y[j], 1]])
            Z[j, i] = eig_cond(A)[1]

    # Plot the colors
    plt.title("Relative Condition Number of 2x2 matrices")
    plt.pcolormesh(X, Y, Z, cmap='gray_r')
    plt.colorbar()
    plt.show()


# Test problem 4
def test_prob4():
    prob4(res=200)


# Problem 5
def prob5(n):
    """Approximate the data from "stability_data.npy" on the interval [0,1]
    with a least squares polynomial of degree n. Solve the least squares
    problem using the normal equation and the QR decomposition, then compare
    the two solutions by plotting them together with the data. Return
    the mean squared error of both solutions, ||Ax-b||_2.

    Parameters:
        n (int): The degree of the polynomial to be used in the approximation.

    Returns:
        (float): The forward error using the normal equations.
        (float): The forward error using the QR decomposition.
    """
    # Load stability data
    xk, yk = np.load("stability_data.npy").T
    A = np.vander(xk, n + 1)

    # Compute least squares using la.inv
    x_1 = la.inv(A.T @ A) @ A.T @ yk

    # Compute least squares using qr decomp
    Q, R = la.qr(A, mode="economic")
    x_2 = la.solve_triangular(R, Q.T @ yk)

    # Plot the polynomials together with the raw data points
    plt.scatter(xk, yk, label="original points")
    plt.plot(xk, np.polyval(x_1, xk), label="la.inv")
    plt.plot(xk, np.polyval(x_2, xk), label="qr decomp")
    plt.legend()
    plt.title("Least Squares Approximations")
    plt.ylim((0, 4))
    plt.show()


# Test problem 5
def test_prob_5():
    prob5(14)


# Problem 6
def prob6():
    """For n = 5, 10, ..., 50, compute the integral I(n) using SymPy (the
    true values) and the subfactorial formula (may or may not be correct).
    Plot the relative forward error of the subfactorial formula for each
    value of n. Use a log scale for the y-axis.
    """
    x = sy.symbols('x')
    fact_vals = []
    domain = range(5, 55, 5)

    for i in domain:
        # Calculate true value
        i = int(i)
        true_val = sy.integrate(x**i * sy.exp(x - 1), (x, 0, 1))

        # Calculate using sy.subfactorial
        fact_val = (-1)**i * (sy.subfactorial(i) - sy.factorial(i) / np.e)
        fact_vals.append(abs(fact_val - true_val))

    # Plot the relative forward error
    plt.plot(domain, fact_vals)
    plt.yscale("log")
    plt.title("Relative Forward Error")
    plt.show()


# Test problem 6
def test_prob_6():
    prob6()

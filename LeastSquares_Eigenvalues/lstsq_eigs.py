# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
<Name>
<Class>
<Date>
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg
import cmath

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la


# Problem 1
def least_squares(A, b):
	"""Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
	Q, R = la.qr(A, mode="economic")
	x = la.solve_triangular(R, np.transpose(Q) @ b)
	return x


# Problem 2
def line_fit():
	"""Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """
	year, index = np.load("housing.npy").T

	# Compute least squares solution
	A = np.column_stack((year, np.ones_like(year)))
	x = least_squares(A, index)
	output = [y * x[0] + x[1] for y in year]

	# Graph the points with the least squares solution
	plt.scatter(year, index, label="Actual Prices")
	plt.plot(year, output, label="Least Squares Line", color="r")
	plt.title("Search Times")
	plt.legend(loc="upper left")
	plt.xlabel("Year after 2000")
	plt.ylabel("Index")
	plt.show()


# Problem 3
def polynomial_fit():
	"""Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """

	# Load data and initialize the domain
	year, index = np.load("housing.npy").T
	max_year = max(year)
	min_year = min(year)
	domain = np.linspace(min_year, max_year, 33)

	# Cycle through each polynomial computing the least
	# squares answer and plotting the answer
	j = 1
	for i in [3, 6, 9, 12]:
		A = np.vander(year, i)
		x = la.lstsq(A, index)
		plt.subplot(2, 2, j)
		plt.scatter(year, index)
		plt.plot(year, np.polyval(x[0], domain), color="r")
		plt.xlabel("Year after 2000")
		plt.ylabel("Index")
		plt.title("Polynomial of degree " + str(i))
		j += 1

	plt.suptitle("Polynomial Fit")
	plt.tight_layout()
	plt.show()


def plot_ellipse(a, b, c, d, e):
	"""Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
	theta = np.linspace(0, 2 * np.pi, 200)
	cos_t, sin_t = np.cos(theta), np.sin(theta)
	A = a * (cos_t ** 2) + c * cos_t * sin_t + e * (sin_t ** 2)
	B = b * cos_t + d * sin_t
	r = (-B + np.sqrt(B ** 2 + 4 * A)) / (2 * A)

	plt.plot(r * cos_t, r * sin_t)
	plt.gca().set_aspect("equal", "datalim")


# Problem 4
def ellipse_fit():
	"""Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
	x_k, y_k = np.load("ellipse.npy").T

	# Build A from the ellipse equation and compute least squares
	A = np.column_stack((x_k ** 2, x_k, x_k * y_k, y_k, y_k ** 2))
	b = np.ones(A.shape[0])
	a, b, c, d, e = la.lstsq(A, b)[0]

	# Graph the solution
	plot_ellipse(a, b, c, d, e)
	plt.title("Ellipse Plot")
	plt.xlabel("X")
	plt.ylabel("Y")
	plt.scatter(x_k, y_k, color="r", label="Original Points")
	plt.show()


# Problem 5
def power_method(A, N=20, tol=1e-12):
	"""Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
	# Initialize starting x
	m, n = np.shape(A)
	x = np.random.random(n)
	x = x / la.norm(x)

	# Cycle through approximately N times to refine the eigenvector x
	for k in range(N - 1):
		x_1 = x
		x = A @ x
		x = x / la.norm(x)
		if la.norm(x - x_1) < tol:
			break

	return np.transpose(x) @ A @ x, x


# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
	"""Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
	# Initial setup with building the Hessenberg matrix S
	m, n = np.shape(A)
	S = la.hessenberg(A)
	for k in range(N):
		Q, R = la.qr(S)
		S = R @ Q
	eigs = []
	i = 0

	while i < n:
		if i == n - 1 or abs(S[i + 1][i]) < tol:
			# Compute eigenvalue for 1 x 1 case
			eigs.append(S[i][i])
		else:
			# Compute eigenvalues via Quadratic Formula for 2 x 2 case
			a, b, c, d = S[i][i], S[i][i + 1], S[i + 1][i], S[i + 1][i + 1]
			new_eig_1 = ((a + d) + cmath.sqrt((a + d) ** 2 - 4 * (a * d - b * c))) / 2
			new_eig_2 = ((a + d) - cmath.sqrt((a + d) ** 2 - 4 * (a * d - b * c)))/ 2
			eigs.append(new_eig_1)
			eigs.append(new_eig_2)
			i += 1
		i += 1

	return eigs


def test_prob_1():
	A = np.random.random((6, 6))
	b = np.random.random(6)
	print(least_squares(A, b))


def test_prob_2():
	line_fit()


def test_prob_3():
	polynomial_fit()


def test_prob_4():
	ellipse_fit()


def test_prob_5():
	A = np.random.random((10, 10))
	eigs, vecs = la.eig(A)
	loc = np.argmax(eigs)
	lamb, x = eigs[loc], vecs[:, loc]
	my_lamb, my_x = power_method(A)
	# print(np.allclose(A @ x, lamb * x))
	# print(np.allclose(A @ x, my_lamb * my_x))


def test_prob_6():
    A = np.random.random((5, 5))
    # print("\n")
    # print(la.eig(A)[0])
    # print(qr_algorithm(A))

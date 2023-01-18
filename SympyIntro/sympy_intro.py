# sympy_intro.py
"""Python Essentials: Introduction to SymPy.
<Name> Dallin Seyfried
<Class> 347 001
<Date> 1/17/2023
"""
from time import perf_counter

import numpy as np
import sympy as sy
import matplotlib.pyplot as plt


# Problem 1
def prob1():
    """Return an expression for

        (2/5)e^(x^2 - y)cosh(x+y) + (3/7)log(xy + 1).

    Make sure that the fractions remain symbolic.
    """
    # Return the above expression
    x, y = sy.symbols('x, y')
    return sy.Rational(2, 5) * (sy.E ** (x**2 - y)) * sy.cosh(x + y) + sy.Rational(3, 7) * sy.log(x*y + 1)


# Problem 2
def prob2():
    """Compute and simplify the following expression.

        product_(i=1 to 5)[ sum_(j=i to 5)[j(sin(x) + cos(x))] ]
    """
    # Setup symbols
    x, y, i, j = sy.symbols('x, y, i, j')
    # Create Expression, simplify it and return it
    expression = sy.product(sy.summation(j * (sy.sin(x) + sy.cos(x)), (j, i, 5)), (i, 1, 5))
    return sy.simplify(expression)


# Problem 3
def prob3(N):
    """Define an expression for the Maclaurin series of e^x up to order N.
    Substitute in -y^2 for x to get a truncated Maclaurin series of e^(-y^2).
    Lambdify the resulting expression and plot the series on the domain
    y in [-2,2]. Plot e^(-y^2) over the same domain for comparison.
    """
    # Generate the maclaurin series with substituted -y^2 and lambdify the expression
    x, y, n = sy.symbols('x, y, n')
    maclaurin = sy.summation(x**n / sy.factorial(n), (n, 0, N))
    sub_maclaurin = maclaurin.subs(x, -y**2)
    f = sy.lambdify(y, sub_maclaurin)
    g = sy.lambdify(y, sy.E**(-y**2))

    # Plot the lambdify function
    domain = np.linspace(-2, 2, 50)
    range_f = f(domain)
    range_g = g(domain)
    plt.plot(domain, range_f, label="-y^2 Maclaurin")
    plt.plot(domain, range_g, label="x Maclaurin")
    plt.title(f"Maclaurin with N = {N}")
    plt.legend()
    plt.show()


# Problem 4
def prob4():
    """The following equation represents a rose curve in cartesian coordinates.

    0 = 1 - [(x^2 + y^2)^(7/2) + 18x^5 y - 60x^3 y^3 + 18x y^5] / (x^2 + y^2)^3

    Construct an expression for the nonzero side of the equation and convert
    it to polar coordinates. Simplify the result, then solve it for r.
    Lambdify a solution and use it to plot x against y for theta in [0, 2pi].
    """
    # Construct, substitute polar coordinates, simplify and solve the expression for r
    x, y, r, theta = sy.symbols('x, y, r, theta')
    rose = 1 - ((x**2 + y**2)**sy.Rational(7/2) + 18*x**5*y - 60*x**3*y**3 + 18*x*y**5) / ((x**2 + y**2)**3)
    polar_rose = rose.subs({x:r*sy.cos(theta), y:r*sy.sin(theta)})
    simp_polar_rose = sy.simplify(polar_rose)
    f = sy.solve(simp_polar_rose, r)[0] # Picking the first solution
    g = sy.lambdify(theta, f)

    # Plot the first solution with x vs y
    domain = np.linspace(0, 2 * np.pi, 200)
    plt.plot(g(domain) * np.cos(domain), g(domain) * np.sin(domain))
    plt.title("x_theta vs y_theta")
    plt.show()


# Problem 5
def prob5():
    """Calculate the eigenvalues and eigenvectors of the following matrix.

            [x-y,   x,   0]
        A = [  x, x-y,   x]
            [  0,   x, x-y]

    Returns:
        (dict): a dictionary mapping eigenvalues (as expressions) to the
            corresponding eigenvectors (as SymPy matrices).
    """
    # Construct the above matrix
    x, y, lambdy = sy.symbols('x, y, lambdy')
    A = sy.Matrix([
        [x - y, x, 0],
        [x, x-y, x],
        [0, x, x-y]
    ])

    # Find the characteristic polynomial and solve for the eigenvalues
    char_poly = sy.det(A - sy.eye(3) * lambdy)
    eigs = sy.solve(char_poly, lambdy)

    # Construct the map of eigenvalues to their eigenvectors
    eigen_map = dict()
    for eigenvalue in eigs:
        eigen_mat = A - sy.eye(3) * eigenvalue
        eigen_vecs = eigen_mat.nullspace()
        eigen_map[eigenvalue] = eigen_vecs

    return eigen_map


# Problem 6
def prob6():
    """Consider the following polynomial.

        p(x) = 2*x^6 - 51*x^4 + 48*x^3 + 312*x^2 - 576*x - 100

    Plot the polynomial and its critical points over [-5,5]. Determine which
    points are maxima and which are minima. Plot the maxima in one color and the
    minima in another color. Return the minima and maxima (x values) as two
    separate sets.

    Returns:
        (set): the local minima.
        (set): the local maxima.
    """
    x = sy.symbols('x')
    poly = 2*x**6 - 51*x**4 + 48*x**3 + 312*x**2 - 576*x -100
    lamb_poly = sy.lambdify(x, poly)

    der_poly = sy.diff(poly, x)
    critical_points = sy.solve(der_poly, x)
    der_2_poly = sy.diff(der_poly, x)

    lamb_der_2_poly = sy.lambdify(x, der_2_poly, 'numpy')
    maxima = np.array([point for point in critical_points if lamb_der_2_poly(point) < 0])
    minima = np.array([point for point in critical_points if lamb_der_2_poly(point) > 0])

    domain = np.linspace(-5, 5, 200)
    plt.plot(domain, lamb_poly(domain), label="p(x)")
    plt.plot(maxima, lamb_poly(maxima), 'bo', label="maxima")
    plt.plot(minima, lamb_poly(minima), 'ro', label="minima")
    plt.title("p(x) with critical points")
    plt.legend()
    plt.show()

    return set(minima), set(maxima)


# Problem 7
def prob7():
    """Calculate the volume integral of f(x,y,z) = (x^2 + y^2 + z^2)^2 over the
    sphere of radius r. Lambdify the resulting expression and plot the integral
    value for r in [0,3]. Return the value of the integral when r = 2.

    Returns:
        (float): the integral of f over the sphere of radius 2.
    """
    raise NotImplementedError("Problem 7 Incomplete")


# Test Problem 1
# def prob1_test():
# prob3(20)
# prob4()
# prob6()

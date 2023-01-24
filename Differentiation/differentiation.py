# differentiation.py
"""Volume 1: Differentiation.
<Name> Dallin Seyfried
<Class> 323 002
<Date> 1/24/2023
"""

import sympy as sy
from matplotlib import pyplot as plt
import numpy as np


# Define a function f globally
x = sy.symbols('x')
f = (sy.sin(x) + 1) ** sy.sin(sy.cos(x))
f_lamb = sy.lambdify(x, f)


# Problem 1
def prob1():
    """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""
    # Setup function and its derivative
    f_prime = sy.diff(f, x)
    f_prime_lamb = sy.lambdify(x, f_prime)
    return f_prime_lamb


# Test problem 1
def prob1_test():
    f_prime_lamb = prob1()

    # Plot the function and its derivative
    domain = np.linspace(-np.pi, np.pi, 100)
    ax = plt.gca()
    ax.spines["bottom"].set_position("zero")
    plt.plot(domain, f_lamb(domain), label="Original")
    plt.plot(domain, f_prime_lamb(domain), label="Derivative")
    plt.legend()
    plt.title("Problem 1")
    plt.show()


# Problem 2
def fdq1(f, x, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""
    return (f(x + h) - f(x)) / h


# Test fdq1
def fdq1_test():
    # Plot the function and its derivative
    domain = np.linspace(-np.pi, np.pi, 100)
    range = fdq1(f_lamb, np.linspace(-np.pi, np.pi, 100))
    ax = plt.gca()
    ax.spines["bottom"].set_position("zero")
    plt.plot(domain, f_lamb(domain), label="Original")
    plt.plot(domain, range, label="Derivative Approximation")
    plt.legend()
    plt.title("Problem fdq1")
    plt.show()


def fdq2(f, x, h=1e-5):
    """Calculate the second order forward difference quotient of f at x."""
    return (-3*f(x) + 4*f(x+h) - f(x+2*h)) / (2*h)


# Test fdq2
def fdq2_test():
    # Plot the function and its derivative
    domain = np.linspace(-np.pi, np.pi, 100)
    range = fdq2(f_lamb, np.linspace(-np.pi, np.pi, 100))
    ax = plt.gca()
    ax.spines["bottom"].set_position("zero")
    plt.plot(domain, f_lamb(domain), label="Original")
    plt.plot(domain, range, label="Derivative Approximation")
    plt.legend()
    plt.title("Problem fdq2")
    plt.show()


def bdq1(f, x, h=1e-5):
    """Calculate the first order backward difference quotient of f at x."""
    return (f(x) - f(x - h)) / h


# Test bdq1
def bdq1_test():
    # Plot the function and its derivative
    domain = np.linspace(-np.pi, np.pi, 100)
    range = bdq1(f_lamb, np.linspace(-np.pi, np.pi, 100))
    ax = plt.gca()
    ax.spines["bottom"].set_position("zero")
    plt.plot(domain, f_lamb(domain), label="Original")
    plt.plot(domain, range, label="Derivative Approximation")
    plt.legend()
    plt.title("Problem bdq1")
    plt.show()


def bdq2(f, x, h=1e-5):
    """Calculate the second order backward difference quotient of f at x."""
    return (3*f(x) - 4*f(x-h) + f(x-2*h)) / (2*h)


# Test bdq2
def bdq2_test():
    # Plot the function and its derivative
    domain = np.linspace(-np.pi, np.pi, 100)
    range = bdq2(f_lamb, np.linspace(-np.pi, np.pi, 100))
    ax = plt.gca()
    ax.spines["bottom"].set_position("zero")
    plt.plot(domain, f_lamb(domain), label="Original")
    plt.plot(domain, range, label="Derivative Approximation")
    plt.legend()
    plt.title("Problem bdq2")
    plt.show()


def cdq2(f, x, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""
    return (f(x+h) - f(x-h)) / (2*h)


# Test cdq2
def cdq2_test():
    # Plot the function and its derivative
    domain = np.linspace(-np.pi, np.pi, 100)
    range = cdq2(f_lamb, np.linspace(-np.pi, np.pi, 100))
    ax = plt.gca()
    ax.spines["bottom"].set_position("zero")
    plt.plot(domain, f_lamb(domain), label="Original")
    plt.plot(domain, range, label="Derivative Approximation")
    plt.legend()
    plt.title("Problem cdq2")
    plt.show()


def cdq4(f, x, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""
    return (f(x-2*h) - 8*f(x-h) + 8*f(x+h) - f(x+2*h)) / (12*h)


# Test cdq4
def cdq4_test():
    # Plot the function and its derivative
    domain = np.linspace(-np.pi, np.pi, 100)
    range = cdq4(f_lamb, np.linspace(-np.pi, np.pi, 100))
    ax = plt.gca()
    ax.spines["bottom"].set_position("zero")
    plt.plot(domain, f_lamb(domain), label="Original")
    plt.plot(domain, range, label="Derivative Approximation")
    plt.legend()
    plt.title("Problem cdq4")
    plt.show()


# Problem 3
def prob3(x0):
    """Let f(x) = (sin(x) + 1)^(sin(cos(x))). Use prob1() to calculate the
    exact value of f'(x0). Then use fdq1(), fdq2(), bdq1(), bdq2(), cdq1(),
    and cdq2() to approximate f'(x0) for h=10^-8, 10^-7, ..., 10^-1, 1.
    Track the absolute error for each trial, then plot the absolute error
    against h on a log-log scale.

    Parameters:
        x0 (float): The point where the derivative is being approximated.
    """
    f_prime_lamb = prob1()
    exact = f_prime_lamb(x0)
    domain = np.logspace(-8, 0, 9)

    # Calculate errors
    fdq1_errors = np.abs(np.array([fdq1(f_lamb, x0, domain[i]) for i in range(len(domain))]) - exact)
    fdq2_errors = np.abs(np.array([fdq2(f_lamb, x0, domain[i]) for i in range(len(domain))]) - exact)
    bdq1_errors = np.abs(np.array([bdq1(f_lamb, x0, domain[i]) for i in range(len(domain))]) - exact)
    bdq2_errors = np.abs(np.array([bdq2(f_lamb, x0, domain[i]) for i in range(len(domain))]) - exact)
    cdq2_errors = np.abs(np.array([cdq2(f_lamb, x0, domain[i]) for i in range(len(domain))]) - exact)
    cdq4_errors = np.abs(np.array([cdq4(f_lamb, x0, domain[i]) for i in range(len(domain))]) - exact)

    # Plot the errors
    plt.loglog(domain, fdq1_errors, label="fdq1 errors")
    plt.loglog(domain, fdq2_errors, label="fdq1 errors")
    plt.loglog(domain, bdq1_errors, label="bdq1 errors")
    plt.loglog(domain, bdq2_errors, label="bdq2 errors")
    plt.loglog(domain, cdq2_errors, label="cdq2 errors")
    plt.loglog(domain, cdq4_errors, label="cdq4 errors")
    plt.title("Problem 3")
    plt.xlabel("h")
    plt.ylabel("Absolute Error")
    plt.show()


# Problem 4
def prob4():
    """The radar stations A and B, separated by the distance 500m, track a
    plane C by recording the angles alpha and beta at one-second intervals.
    Your goal, back at air traffic control, is to determine the speed of the
    plane.

    Successive readings for alpha and beta at integer times t=7,8,...,14
    are stored in the file plane.npy. Each row in the array represents a
    different reading; the columns are the observation time t, the angle
    alpha (in degrees), and the angle beta (also in degrees), in that order.
    The Cartesian coordinates of the plane can be calculated from the angles
    alpha and beta as follows.

    x(alpha, beta) = a tan(beta) / (tan(beta) - tan(alpha))
    y(alpha, beta) = (a tan(beta) tan(alpha)) / (tan(beta) - tan(alpha))

    Load the data, convert alpha and beta to radians, then compute the
    coordinates x(t) and y(t) at each given t. Approximate x'(t) and y'(t)
    using a first order forward difference quotient for t=7, a first order
    backward difference quotient for t=14, and a second order centered
    difference quotient for t=8,9,...,13. Return the values of the speed at
    each t.
    """
    # functions for finding coordinates with distance = 500 m
    find_x_coords = lambda a, b: 500 * np.tan(b) / (np.tan(b) - np.tan(a))
    find_y_coords = lambda a, b: 500 * np.tan(b) * np.tan(a) / (np.tan(b) - np.tan(a))

    data = np.load('plane.npy')

    alpha = np.deg2rad(data[:, 1])
    beta = np.deg2rad(data[:, 2])

    x_coords = find_x_coords(alpha, beta)
    y_coords = find_y_coords(alpha, beta)

    # First order forward difference quotient
    x_t_prime = [x_coords[1] - x_coords[0]]

    # First order backward difference quotient
    x_t_prime.append(x_coords[i] - x_coords[i - 1] for i in range(1, len(x_coords) - 1))

    # Second order centered difference quotient

    pass

prob4()


# Problem 5
def jacobian_cdq2(f, x, h=1e-5):
    """Approximate the Jacobian matrix of f:R^n->R^m at x using the second
    order centered difference quotient.

    Parameters:
        f (function): the multidimensional function to differentiate.
            Accepts a NumPy (n,) ndarray and returns an (m,) ndarray.
            For example, f(x,y) = [x+y, xy**2] could be implemented as follows.
            >>> f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2])
        x ((n,) ndarray): the point in R^n at which to compute the Jacobian.
        h (float): the step size in the finite difference quotient.

    Returns:
        ((m,n) ndarray) the Jacobian matrix of f at x.
    """
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def cheb_poly(x, n):
    """Compute the nth Chebyshev polynomial at x.

    Parameters:
        x (jax.ndarray): the points to evaluate T_n(x) at.
        n (int): The degree of the polynomial.
    """
    raise NotImplementedError("Problem 6 Incomplete")

def prob6():
    """Use JAX and cheb_poly() to create a function for the derivative
    of the Chebyshev polynomials, and use that function to plot the derivatives
    over the domain [-1,1] for n=0,1,2,3,4.
    """
    raise NotImplementedError("Problem 6 Incomplete")


# Problem 7
def prob7(N=200):
    """
    Let f(x) = (sin(x) + 1)^sin(cos(x)). Perform the following experiment N
    times:

        1. Choose a random value x0.
        2. Use prob1() to calculate the "exact" value of f′(x0). Time how long
            the entire process takes, including calling prob1() (each
            iteration).
        3. Time how long it takes to get an approximation of f'(x0) using
            cdq4(). Record the absolute error of the approximation.
        4. Time how long it takes to get an approximation of f'(x0) using
            JAX (calling grad() every time). Record the absolute error of
            the approximation.

    Plot the computation times versus the absolute errors on a log-log plot
    with different colors for SymPy, the difference quotient, and JAX.
    For SymPy, assume an absolute error of 1e-18.
    """
    raise NotImplementedError("Problem 7 Incomplete")

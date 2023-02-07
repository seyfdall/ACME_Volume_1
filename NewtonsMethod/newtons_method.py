# newtons_method.py
"""Volume 1: Newton's Method.
<Name> Dallin Seyfried
<Class> 001
<Date> 1/31/2023
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton as eye_of_newt
import scipy.linalg as la


# Problems 1, 3, and 5
def newton(f, x0, Df, tol=1e-5, maxiter=15, alpha=1.):
    """Use Newton's method to approximate a zero of the function f.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.
        alpha (float): Backtracking scalar (Problem 3).

    Returns:
        (float or ndarray): The approximation for a zero of f.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    post = x0
    converged = False
    is_matrix = not np.isscalar(f(x0))

    # If matrix solve using la.solve
    if is_matrix:
        i = 0
        for _ in range(maxiter):
            i += 1
            pre = post
            post = pre - alpha * la.solve(Df(pre), f(pre))
            if la.norm(post - pre) < tol:
                converged = True
                break
    else:
        # Cycle up to maxiter times and return if converged
        i = 0
        for _ in range(maxiter):
            i += 1
            # Including alpha for backtracking
            pre = post
            post = pre - alpha * f(pre) / Df(pre)
            if np.linalg.norm(post - pre) < tol:
                converged = True
                break

    return post, converged, i


# Function to test newton() for problem 1
def prob1_test():
    f = lambda x: np.exp(x) - 2
    Df = lambda x: np.exp(x)
    print(f"Newton: {newton(f, 2.0, Df)}")
    print(f"Scipy: {eye_of_newt(f, 2.0, Df)}")

    f = lambda x: x**4 - 3
    Df = lambda x: 4*x**3
    print(f"Newton: {newton(f, 0.5, Df)}")
    print(f"Scipy: {eye_of_newt(f, 0.5, Df)}")


def prob5_test():
    # f(x,y) = [x - y, x^2 - 3], x0 = [2, 1]
    f = lambda x: np.array([x[0] - x[1], x[0]**2 - 3])
    Df = lambda x: np.array([
        [1, -1],
        [2*x[0], 0]
    ])
    x0 = np.array([2, 1])
    print(f"Newton: {newton(f, x0, Df)}")


def prob3_test():
    f = lambda x: np.sign(x) * np.power(np.abs(x), 1./3)
    Df = lambda x: np.power(np.abs(x), -2./3) / 3.
    print(newton(f,.01,Df,alpha=1))    # this should be (-327.679..., False, 15)
    print(newton(f,.01,Df,alpha=.4))   # this should be (6.400...e-7, True, 6)


# Problem 2
def prob2(N1, N2, P1, P2):
    """Use Newton's method to solve for the constant r that satisfies

                P1[(1+r)**N1 - 1] = P2[1 - (1+r)**(-N2)].

    Use r_0 = 0.1 for the initial guess.

    Parameters:
        P1 (float): Amount of money deposited into account at the beginning of
            years 1, 2, ..., N1.
        P2 (float): Amount of money withdrawn at the beginning of years N1+1,
            N1+2, ..., N1+N2.
        N1 (int): Number of years money is deposited.
        N2 (int): Number of years money is withdrawn.

    Returns:
        (float): the value of r that satisfies the equation.
    """
    # Initial Guess
    r0 = 0.1
    # Function f
    f = lambda r: P1*((1 + r)**N1 - 1) - P2*(1 - (1 + r)**(-N2))
    # Derivative of f
    Df = lambda r: P1*N1*(1 + r)**(N1 - 1) - P2*N2*(1 + r)**(-N2 - 1)

    # Compute the newton's method and return r
    results = newton(f, r0, Df)
    return results[0]


# Function to test problem 2
def prob2_test():
    N1, N2, P1, P2 = 30, 20, 2000, 8000
    print(prob2(N1, N2, P1, P2))


# Problem 4
def optimal_alpha(f, x0, Df, tol=1e-5, maxiter=15):
    """Run Newton's method for various values of alpha in (0,1].
    Plot the alpha value against the number of iterations until convergence.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): a value for alpha that results in the lowest number of
            iterations.
    """
    # Define linspace of alphas to try and iterations tested on
    alphas = np.linspace(0.01, 1, 100)
    iterations = [newton(f, x0, Df, tol, maxiter, alpha)[2] for alpha in alphas]

    # Find the most effective alpha
    min_iter = min(iterations)
    min_index = iterations.index(min_iter)

    # Plot the values
    plt.plot(alphas, iterations)
    plt.title("Optimal Alpha")
    plt.ylabel("Iterations")
    plt.xlabel("Alpha Value")
    plt.tight_layout()
    plt.show()

    return alphas[min_index]


def prob4_test():
    f = lambda x: np.sign(x) * np.power(np.abs(x), 1./3)
    Df = lambda x: np.power(np.abs(x), -2./3) / 3.
    print(optimal_alpha(f,.01,Df)) # should return alpha closer to .3 than .4


# Problem 6
def prob6():
    """Consider the following Bioremediation system.

                              5xy − x(1 + y) = 0
                        −xy + (1 − y)(1 + y) = 0

    Find an initial point such that Newton’s method converges to either
    (0,1) or (0,−1) with alpha = 1, and to (3.75, .25) with alpha = 0.55.
    Return the intial point as a 1-D NumPy array with 2 entries.
    """
    # Define search space
    x_search = np.linspace(-0.25, 0, 50)
    y_search = np.linspace(0, 0.25, 50)
    tol = 1e-5

    # Define f and Df
    f = lambda x: np.array([4. * x[0] * x[1] - x[0], -x[0] * x[1] + 1. - x[1] ** 2.])
    Df = lambda x: np.array([[4. * x[1] - 1, 4. * x[0]], [-x[1], -x[0] - 2. * x[1]]])

    # Search through with alpha = 1 looking for (0, 1) or (0, -1)
    for x in x_search:
        for y in y_search:
            vect, converged, iters = newton(f, np.array([x, y]), Df, alpha=1)
            if np.allclose(np.abs(vect), np.array([0,1])):
                # Search through with alpha = 0.55 looking for (3.75, 0.25)
                vect, converged, iters = newton(f, np.array([x, y]), Df, alpha=0.55)
                if np.allclose(vect, np.array([3.75, 0.25])):
                    return np.array([x, y])


# Problem 7
def plot_basins(f, Df, zeros, domain, res=1000, iters=15):
    """Plot the basins of attraction of f on the complex plane.

    Parameters:
        f (function): A function from C to C.
        Df (function): The derivative of f, a function from C to C.
        zeros (ndarray): A 1-D array of the zeros of f.
        domain ([r_min, r_max, i_min, i_max]): A list of scalars that define
            the window limits and grid domain for the plot.
        res (int): A scalar that determines the resolution of the plot.
            The visualized grid has shape (res, res).
        iters (int): The exact number of times to iterate Newton's method.
    """
    # Construct a mesh grid
    x_real = np.linspace(domain[0], domain[1], res)
    x_imag = np.linspace(domain[2], domain[3], res)
    X_real, X_imag = np.meshgrid(x_real, x_imag)
    X_0 = X_real + 1j*X_imag

    X_k = X_0
    # Run Newton's Method
    for _ in range(iters):
        X_iter = X_k
        X_k = X_iter - f(X_iter) / Df(X_iter)

    # Compute res x res array Y
    Y = np.argmin(np.array([np.abs(X_k - zero) for zero in zeros]), axis=0)

    # Plot and visualize the basins
    plt.pcolormesh(x_real, x_imag, Y, cmap="brg")
    plt.title("Basins")
    plt.show()


# Test Plot Basins
def plot_basins_test():
    # Plot x^3 - 1
    f = lambda x: x**3 - 1
    Df = lambda x: 3*x**2
    zeros = np.array([1, -0.5 + 1j*np.sqrt(3)/2, -0.5 - 1j*np.sqrt(3)/2])
    domain = [-1.5, 1.5, -1.5, 1.5]
    plot_basins(f, Df, zeros, domain)

    # Plot x^3 - x
    f = lambda x: x**3 - x
    Df = lambda x: 3*x**2 - 1
    zeros = np.array([0, 1, -1])
    domain = [-1.5, 1.5, -1.5, 1.5]
    plot_basins(f, Df, zeros, domain)

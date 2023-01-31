# newtons_method.py
"""Volume 1: Newton's Method.
<Name> Dallin Seyfried
<Class> 001
<Date> 1/31/2023
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import newton as eye_of_newt

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
            post = np.linalg.solve(Df(pre), f(pre))
            if np.linalg.norm(post - pre) < tol:
                converged = True
                break
    else:
        # Cycle up to maxiter times and return if converged
        i = 0
        for _ in range(maxiter):
            i += 1
            pre = post
            # Including alpha for backtracking
            post = pre - alpha * f(pre) / Df(pre)
            if np.linalg.norm(post - pre) < tol:
                converged = True
                break

    return post, converged, i


# Function to test newton() for problem 1
def prob1_test():
    f = lambda x: np.exp(x) - 2
    Df = lambda x: np.exp(x)
    print(f"Newton: {newton(f, 0.6, Df)}")
    print(f"Scipy: {eye_of_newt(f, 0.6, Df)}")


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


def prob5_test():
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
    x_search = np.linspace(-0.25, 0, 100, endpoint=False)
    y_search = np.linspace(0.00001, 0.25, 100)

    # Define f and Df


    # Search through with alpha = 1 looking for (0, 1) or (0, -1)

    # Search through with alpha = 0.55 looking for (3.75, 0.25


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
    raise NotImplementedError("Problem 7 Incomplete")

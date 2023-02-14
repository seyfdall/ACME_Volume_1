# montecarlo_integration.py
"""Volume 1: Monte Carlo Integration.
<Name> Dallin Seyfried
<Class> 001
<Date> 2/14/2023
"""

import numpy as np
from scipy import linalg as la
from scipy.stats.mvn import mvnun

# Problem 1
def ball_volume(n, N=10000):
    """Estimate the volume of the n-dimensional unit ball.

    Parameters:
        n (int): The dimension of the ball. n=2 corresponds to the unit circle,
            n=3 corresponds to the unit sphere, and so on.
        N (int): The number of random points to sample.

    Returns:
        (float): An estimate for the volume of the n-dimensional unit ball.
    """
    # Get N points in n-D domains
    points = np.random.uniform(-1, 1, (n, N))

    # Determine how many points are within the circle
    lengths = la.norm(points, axis=0, ord=2)
    num_within = np.count_nonzero(lengths < 1)

    # Estimate the Ball's area
    return 2**n * (num_within / N)


# Test ball volume - problem 1
def test_ball_volume():
    print(ball_volume(3))


# Problem 2
def mc_integrate1d(f, a, b, N=10000):
    """Approximate the integral of f on the interval [a,b].

    Parameters:
        f (function): the function to integrate. Accepts and returns scalars.
        a (float): the lower bound of interval of integration.
        b (float): the lower bound of interval of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over [a,b].

    Example:
        # >>> f = lambda x: x**2
        # >>> mc_integrate1d(f, -4, 2)    # Integrate from -4 to 2.
        23.734810301138324              # The true value is 24.
    """
    # Get N points in the interval [a,b]
    points = np.random.uniform(a, b, N)

    return (b - a) / N * np.sum(f(points))


# Test mc_integrate - problem 2
def test_mc_integrate1d():
    f = lambda x: x**2
    f_1 = lambda x: np.sin(x)
    assert abs(24 - mc_integrate1d(f, -4, 2)) < 0.5, "Integrate for x**2 failed"
    assert abs(0 - mc_integrate1d(f_1, -4, 2)) < 0.5, "Integrate for x**2 failed"


# Problem 3
def mc_integrate(f, mins, maxs, N=10000):
    """Approximate the integral of f over the box defined by mins and maxs.

    Parameters:
        f (function): The function to integrate. Accepts and returns
            1-D NumPy arrays of length n.
        mins (list): the lower bounds of integration.
        maxs (list): the upper bounds of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over the domain.

    Example:
        # Define f(x,y) = 3x - 4y + y^2. Inputs are grouped into an array.
        # >>> f = lambda x: 3*x[0] - 4*x[1] + x[1]**2

        # Integrate over the box [1,3]x[-2,1].
        # >>> mc_integrate(f, [1, -2], [3, 1])
        53.562651072181225              # The true value is 54.
    """
    # Calculate the volume
    maxs = np.array(maxs)
    mins = np.array(mins)
    volume = np.prod(maxs - mins)

    # Get N points in n-D domains and scale them
    points = np.random.uniform(0, 1, (len(maxs), N))

    # Use array broadcasting to shift the points
    for i in range(len(maxs)):
        points[i] = points[i] * (maxs[i] - mins[i]) + mins[i]

    return volume / N * np.sum(f(points))


# Test mc_integrate - problem 3
def test_mc_integrate():
    # Test x^2 + y^2
    maxs = [1, 1]
    mins = [0, 0]
    f = lambda x: x[0]**2 + x[1]**2
    print(mc_integrate(f, mins, maxs))
    assert abs(2/3 - mc_integrate(f, mins, maxs)) < 0.5, "Error on x**2 + y**2"

    # Test x + y - wz^2
    maxs = [-1, -2, -3, -4]
    mins = [1, 2, 3, 4]
    f_2 = lambda x: x[0] + x[1] - x[3] * x[2]**2
    print(mc_integrate(f_2, mins, maxs, N=int(1e7)))
    assert abs(0 - mc_integrate(f_2, mins, maxs, N=int(1e7))) < 0.5, "Error on x + y - wz**2"


# Problem 4
def prob4():
    """Let n=4 and Omega = [-3/2,3/4]x[0,1]x[0,1/2]x[0,1].
    - Define the joint distribution f of n standard normal random variables.
    - Use SciPy to integrate f over Omega.
    - Get 20 integer values of N that are roughly logarithmically spaced from
        10**1 to 10**5. For each value of N, use mc_integrate() to compute
        estimates of the integral of f over Omega with N samples. Compute the
        relative error of estimate.
    - Plot the relative error against the sample size N on a log-log scale.
        Also plot the line 1 / sqrt(N) for comparison.
    """
    # Setup for the function
    n = 4
    mins = np.array([-3/2, 0, 0, 0])
    maxs = np.array([3/4, 1, 1/2, 1])
    f = lambda x: np.exp(-x.T @ x / 2) / ((2 * np.pi)**n/2)

    # The distribution has mean 0 and covariance I (the nxn identity)
    means, cov = np.zeros(4), np.eye(4)

    # Compute the integral with SciPy
    true_val = mvnun(mins, maxs, means, cov)[0]

    


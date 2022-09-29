# linear_transformations.py
"""Volume 1: Linear Transformations.
<Name>
<Class>
<Date>
"""

from random import random
import numpy as np
from matplotlib import pyplot as plt
import time


# Problem 1
def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    # Create the stretch matrix and return the matrix multiplication of them
    stretch_matrix = np.array([[a, 0], [0, b]])
    return np.matmul(stretch_matrix, A)

def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    # Create the shear matrix and return the matrix multiplication of them
    shear_matrix = np.array([[1, a], [b, 1]])
    return np.matmul(shear_matrix, A)

def reflect(A, a, b):
    """Reflect the points in A about the line that passes through the origin
    and the point (a,b).

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    # Create the reflect matrix and return the matrix multiplication of them
    mat = np.array([[a**2 - b**2, 2*a*b], [2*a*b, b**2 - a**2]])
    coef = 1 / (a**2 + b**2)
    t = mat * coef
    return np.matmul(t, A)

def rotate(A, theta):
    """Rotate the points in A about the origin by theta radians.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    Return:
        ((2,n) ndarray): Transformed matrix
    """
    # Create the rotation matrix and return the matrix multiplication of them
    rotate_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.matmul(rotate_matrix, A)


# Problem 2
def solar_system(T, x_e, x_m, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).

    Parameters:
        T (float): The final time.
        x_e (float): The earth's initial x coordinate.
        x_m (float): The moon's initial x coordinate.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    # Setup initial variables
    p_e = np.array([x_e, 0])
    p_m = np.array([x_m, 0])

    earth_coords_x = [x_e]
    earth_coords_y = [0]
    moon_coords_x = [x_m]
    moon_coords_y = [0]

    for t in np.linspace(0, T, num=100):

        # Step 1 compute p_e(T) by rotating p_e counterclockwise by omega_e radians
        p_e_t = rotate(p_e, t * omega_e)

        # Step 2 calculate position of moon relative to earth at time t
        moon_relative_earth = rotate(p_m - p_e, t * omega_m)

        # Step 3 compute p_m(T) translate
        p_m_t = p_e_t + moon_relative_earth

        # Append coordinates to arrays
        moon_coords_x.append(p_m_t[0])
        moon_coords_y.append(p_m_t[1])
        earth_coords_x.append(p_e_t[0])
        earth_coords_y.append((p_e_t[1]))

    plt.plot(moon_coords_x, moon_coords_y, 'r--', markersize=1.5)
    plt.plot(earth_coords_x, earth_coords_y, '.b', markersize=3.0)
    plt.gca().set_aspect("equal")
    plt.show()


def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]

# Problem 3
def prob3():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """

    # Initialize tracking arrays
    mat_vec_times = [0]
    mat_mat_times = [0]
    n = [i for i in range(50, 250, 50)]

    for i in n:
        A = random_matrix(i)
        x = random_vector(i)

        # Matrix-Matrix multiplication first
        start = time.time()
        matrix_matrix_product(A, A)
        finish = time.time()
        mat_mat_times.append(finish - start)

        # Matrix-Vector
        start = time.time()
        matrix_vector_product(A, x)
        finish = time.time()
        mat_vec_times.append(finish - start)

    # Create and subplot the Matrix-Vector graph
    ax1 = plt.subplot(121)
    ax1.plot(range(0, 250, 50), mat_vec_times, 'g-')
    plt.xlabel("n")
    plt.ylabel("Seconds")
    plt.title("Matrix-Vector Multiplication")

    # Create and subplot the sin(2x) graph
    ax2 = plt.subplot(122)
    ax2.plot(range(0, 250, 50), mat_mat_times, 'r--')
    plt.xlabel("n")
    plt.title("Matrix-Matrix Multiplication")
    plt.show()

# Problem 4
def prob4():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    # Initialize tracking arrays
    mat_vec_prod_times = []
    mat_mat_prod_times = []
    mat_vec_dot_times = []
    mat_mat_dot_times = []
    n = [2**i for i in range(9)]

    for i in n:
        A = random_matrix(i)
        x = random_vector(i)

        # Matrix-Matrix Product
        start = time.perf_counter()
        matrix_matrix_product(A, A)
        finish = time.perf_counter()
        mat_mat_prod_times.append(finish - start)

        # Matrix-Vector Product
        start = time.perf_counter()
        matrix_vector_product(A, x)
        finish = time.perf_counter()
        mat_vec_prod_times.append(finish - start)

        # Matrix-Matrix np.dot
        start = time.perf_counter()
        np.dot(A, A)
        finish = time.perf_counter()
        mat_mat_dot_times.append(finish - start)

        # Matrix-Vector np.dot
        start = time.perf_counter()
        np.dot(A, x)
        finish = time.perf_counter()
        mat_vec_dot_times.append(finish - start)

    # Create and subplot the linear time graph
    ax1 = plt.subplot(121)
    ax1.plot(range(0, 270, 30), mat_mat_prod_times, 'g-', label="Mat_Mat_Prod")
    ax1.plot(range(0, 270, 30), mat_vec_prod_times, 'r-', label="Mat_Vec_Prod")
    ax1.plot(range(0, 270, 30), mat_mat_dot_times, 'b-', label="Mat_Mat_Dot")
    ax1.plot(range(0, 270, 30), mat_vec_dot_times, 'm-', label="Mat_Vec_Dot")
    plt.xlabel("n")
    plt.ylabel("Seconds")
    plt.title("Linear Graph Multiplication")
    plt.legend(loc="upper left")

    # Create and subplot the log-log graph
    ax2 = plt.subplot(122)
    ax2.set_xlim((1, 250))
    ax2.set_ylim((2**-20, 1))
    ax2.loglog(range(0, 270, 30), mat_mat_prod_times, 'g-', label="Mat_Mat_Prod", base=2)
    ax2.loglog(range(0, 270, 30), mat_vec_prod_times, 'r-', label="Mat_Vec_Prod", base=2)
    ax2.loglog(range(0, 270, 30), mat_mat_dot_times, 'b-', label="Mat_Mat_Dot", base=2)
    ax2.loglog(range(0, 270, 30), mat_vec_dot_times, 'm-', label="Mat_Vec_Dot", base=2)
    plt.xlabel("n")
    plt.ylabel("Seconds")
    plt.title("Log Log Graph")
    plt.legend(loc="upper left")
    plt.show()


def showplot(H):
    # This function displays the image produce by the collection of coordinates given in H
    plt.plot(H[0,:],H[1,:],'k.',markersize=3.5)
    plt.axis([-1.5,1.5,-1.5,1.5])
    plt.gca().set_aspect("equal")
    plt.show()


def test_stretch():
    """Problem 1 Stretch Unit test"""
    data = np.load("horse.npy")
    showplot(data)
    showplot(stretch(data, 2, 2))


def test_shear():
    """Problem 1 Shear Unit test"""
    data = np.load("horse.npy")
    showplot(data)
    showplot(shear(data, 2, 2))


def test_reflection():
    """Problem 1 Reflection Unit Test"""
    data = np.load("horse.npy")
    showplot(data)
    showplot(reflect(data, 2, 2))


def test_rotation():
    """Problem 1 Rotate Unit Test"""
    data = np.load("horse.npy")
    showplot(data)
    showplot(rotate(data, 180))

def test_solar_system():
    """Problem 2 Solar System Unit Test"""
    solar_system(3*np.pi/2, 10, 11, 1, 13)

def test_prob_3():
    """Problem 3 Test"""
    prob3()

def test_prob_4():
    """Problem 4 Test"""
    prob4()
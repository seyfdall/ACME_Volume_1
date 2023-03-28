# profiling.py
"""Python Essentials: Profiling.
<Name> Dallin Seyfried
<Class> 001
<Date> 3/21/2023
"""

# Note: for problems 1-4, you need only implement the second function listed.
# For example, you need to write max_path_fast(), but keep max_path() unchanged
# so you can do a before-and-after comparison.
import math
import time
import matplotlib.pyplot as plt
from numba import jit
import numpy as np

# Problem 1
def max_path(filename="triangle.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    def path_sum(r, c, total):
        """Recursively compute the max sum of the path starting in row r
        and column c, given the current total.
        """
        total += data[r][c]
        if r == len(data) - 1:          # Base case.
            return total
        else:                           # Recursive case.
            return max(path_sum(r+1, c,   total),   # Next row, same column
                       path_sum(r+1, c+1, total))   # Next row, next column

    return path_sum(0, 0, 0)            # Start the recursion from the top.


def max_path_fast(filename="triangle_large.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                for line in infile.readlines()]

    # Cycle from the next to bottom row working up
    n = len(data)
    for i in range(n - 1)[::-1]:
        m = len(data[i])
        for j in range(m):
            left = data[i + 1][j]
            right = data[i + 1][j + 1]
            if right >= left:
                data[i][j] = data[i][j] + right
            else:
                data[i][j] = data[i][j] + left

    return data[0][0]


# Test Max_Path
def test_max_path():
    print(max_path())
    print(max_path_fast("triangle.txt"))


# Problem 2
def primes(N):
    """Compute the first N primes."""
    primes_list = []
    current = 2
    while len(primes_list) < N:
        isprime = True
        for i in range(2, current):     # Check for nontrivial divisors.
            if current % i == 0:
                isprime = False
        if isprime:
            primes_list.append(current)
        current += 1
    return primes_list


def primes_fast(N):
    """Compute the first N primes."""
    # Make this a set if too slow
    primes_list = [2]
    current = 3
    nprimes = 1
    while nprimes < N:
        is_prime = True
        max = int(current**0.5) + 1
        for i in primes_list:     # Check for nontrivial divisors.
            if current % i == 0:
                is_prime = False
                break
            # Check square for root condition
            elif i > max:
                break
        if is_prime:
            nprimes += 1
            primes_list.append(current)
        current += 2
    return primes_list


# Test Primes
def test_primes_fast():
    print('\n')
    print(primes(100))
    print(primes_fast(100))


# Problem 3
def nearest_column(A, x):
    """Find the index of the column of A that is closest to x.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    distances = []
    for j in range(A.shape[1]):
        distances.append(np.linalg.norm(A[:,j] - x))
    return np.argmin(distances)


def nearest_column_fast(A, x):
    """Find the index of the column of A that is closest in norm to x.
    Refrain from using any loops or list comprehensions.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    # Modify the shape of x so that it can be array-broadcasted with A
    x.shape = (len(x), 1)
    return np.argmin(np.linalg.norm(A - x, axis=0))


# Test nearest_column_fast
def test_nearest_column():
    print('\n')
    A = np.array([[1, 2, 3, 4], [1, 1, 3, 5]])
    x = np.array([1, 1])
    print(nearest_column(A, x))
    print(nearest_column_fast(A, x))


# Problem 4
def name_scores(filename="names.txt"):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    total = 0
    for i in range(len(names)):
        name_value = 0
        for j in range(len(names[i])):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for k in range(len(alphabet)):
                if names[i][j] == alphabet[k]:
                    letter_value = k + 1
            name_value += letter_value
        total += (names.index(names[i]) + 1) * name_value
    return total


def name_scores_fast(filename='names.txt'):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    alphabet = {chr(ind): ind - 64 for ind in range(65, 91)}
    total = 0
    for index, name in enumerate(names):
        name_value = 0
        for letter in name:
            name_value += alphabet[letter]
        total += (index + 1) * name_value
    return total


# Test Problem 4
def test_names_scores():
    print('\n')
    print(name_scores())
    print(name_scores_fast())


# Problem 5
def fibonacci():
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    # Yield the first 2 terms in the Fibonacci sequence
    a = b = 1
    while True:
        yield a
        a, b = b, a + b


# Test fibonacci
def test_fibonacci():
    fib = fibonacci()
    print('\n')
    print(next(fib))
    print(next(fib))
    print(next(fib))
    print(next(fib))
    print(next(fib))


def fibonacci_digits(N=1000):
    """Return the index of the first term in the Fibonacci sequence with
    N digits.

    Returns:
        (int): The index.
    """
    # Start the generator
    fib = fibonacci()
    index = 0
    while True:
        index += 1
        next_val = next(fib)

        # Use math.log to efficiently find number of digits
        if math.log10(next_val) + 1 >= N:
            return index


# Test fibonacci_digits
def test_fibonacci_digits():
    print(fibonacci_digits())


# Problem 6
def prime_sieve(N):
    """Yield all primes that are less than N."""
    # Start the list already pruned to the 3rd prime
    integers = [i for i in range(3, N, 2) if i % 3 != 0]
    curr_prime = 3

    yield 2

    # Cycle and generate next primes
    while integers:
        yield curr_prime
        curr_prime = integers.pop(0)
        for num in integers:
            if num % curr_prime == 0:
                integers.remove(num)

    yield curr_prime


# Test prime_sieve
def test_prime_sieve():
    prim = prime_sieve(100000)
    test = next(prim)
    while test != 99991:
        test = next(prim)


# Problem 7
def matrix_power(A, n):
    """Compute A^n, the n-th power of the matrix A."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product


@jit
def matrix_power_numba(A, n):
    """Compute A^n, the n-th power of the matrix A, with Numba optimization."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i, k] * A[k, j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product


def prob7(n=10):
    """Time matrix_power(), matrix_power_numba(), and np.linalg.matrix_power()
    on square matrices of increasing size. Plot the times versus the size.
    """
    # Compile matrix power numba
    matrix_power_numba(np.random.random((2, 2)), 2)

    # Time lists for functions
    mat_pow_times = []
    mat_numba_times = []
    linalg_times = []
    sizes = [2**i for i in range(2, 8)]

    # Cycle for powers of 2 up to 2^7
    for m in sizes:
        # Generate random matrix
        A = np.random.random((m, m))

        # Time matrix_power()
        start = time.perf_counter()
        matrix_power(A, n)
        end = time.perf_counter()
        mat_pow_times.append(end - start)

        # Time matrix_power()
        start = time.perf_counter()
        matrix_power_numba(A, n)
        end = time.perf_counter()
        mat_numba_times.append(end - start)

        # Time matrix_power()
        start = time.perf_counter()
        np.linalg.matrix_power(A, n)
        end = time.perf_counter()
        linalg_times.append(end - start)

    # Plot the times on a log log scale
    plt.loglog(sizes, mat_pow_times, label="matrix_power")
    plt.loglog(sizes, mat_numba_times, label="matrix_power_numba")
    plt.loglog(sizes, linalg_times, label="np.linalg")
    plt.ylabel("Times")
    plt.xlabel("Size")
    plt.tight_layout()
    plt.legend()
    plt.show()


# Test prob7
def test_prob7():
    prob7()

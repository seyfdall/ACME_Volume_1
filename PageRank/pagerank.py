# solutions.py
"""Volume 1: The Page Rank Algorithm.
<Name> Dallin Seyfried
<Class> 001
<Date> 03/07/2023
"""

import numpy as np


# Problems 1-2
class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        (fill this out after completing DiGraph.__init__().)
    """
    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        A = A.astype("float64")
        self.n = len(A)

        # Remove sinks in graph and normalize columns
        for i in range(self.n):
            col_sum = np.sum(A[:, i])
            if col_sum == 0:
                A[:, i] += 1 / len(A)
            else:
                A[:, i] /= col_sum

        self.A_hat = A

        # Set labels if valid
        if labels is None:
            self.labels = list(range(self.n))
        elif len(labels) != self.n:
            raise ValueError("Number of labels is not equal to number of nodes in graph")
        else:
            self.labels = labels

    # Problem 2
    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # Use np.linalg.solve to find p
        A = np.eye(self.n) - epsilon * self.A_hat
        b = ((1 - epsilon) / self.n) * np.ones(self.n)
        p = np.linalg.solve(A, b)
        return dict(zip(self.labels, p))

    # Problem 2
    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # Create B and then find it's first eigenvalue
        B = epsilon * self.A_hat + ((1 - epsilon) / self.n) * np.ones((self.n, self.n))
        eigs, vecs = np.linalg.eig(B)
        p = vecs[:, 0] / np.sum(vecs[:, 0])
        return dict(zip(self.labels, p))

    # Problem 2
    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # Initialize starting x and B
        B = epsilon * self.A_hat + ((1 - epsilon) / self.n) * np.ones((self.n, self.n))
        x = np.random.random(self.n)
        x = x / np.linalg.norm(x)

        # Cycle through approximately maxiter times to refine the eigenvector x
        for k in range(maxiter):
            x_1 = x
            x = B @ x
            x = x / np.linalg.norm(x)
            if np.linalg.norm(x - x_1) < tol:
                break

        x = x / np.sum(x)
        return dict(zip(self.labels, x))


# Test the Digraph class
def test_DiGraph():
    A = np.array([
        [0, 0, 0, 0],
        [1, 0, 1, 0],
        [1, 0, 0, 1],
        [1, 0, 1, 0]
    ])
    dig = DiGraph(A, ['a', 'b', 'c', 'd'])
    print("\n")
    print(dig.linsolve())
    print(dig.eigensolve())
    print(dig.itersolve())
    print(get_ranks(dig.linsolve()))


# Problem 3
def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """
    # Get lables and values of dictionary
    labels = list(d.keys())
    values = list(d.values())

    # Use argsort to get new order, reverse it and return it
    sort = np.argsort(values)
    sorted_labels = [labels[ind] for ind in sort]
    sorted_labels.reverse()
    return sorted_labels


# Problem 4
def rank_websites(filename="web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks(). If two webpages have the same rank,
    resolve ties by listing the webpage with the larger ID number first.

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.
    """
    # '' possible - A should be shape(630, 630) - data.read().strip()
    with open(filename, 'r') as infile:
        contents = infile.read()


# Problem 5
def rank_ncaa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """
    raise NotImplementedError("Problem 5 Incomplete")


# Problem 6
def rank_actors(filename="top250movies.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node a points to
    node b with weight w if actor a and actor b were in w movies together but
    actor b was listed first. Use NetworkX to compute the PageRank values of
    the actors, then rank them with get_ranks().

    Each line of the file has the format
        title/actor1/actor2/actor3/...
    meaning actor2 and actor3 should each have an edge pointing to actor1,
    and actor3 should have an edge pointing to actor2.
    """
    raise NotImplementedError("Problem 6 Incomplete")

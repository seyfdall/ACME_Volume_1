# image_segmentation.py
"""Volume 1: Image Segmentation.
<Name>
<Class>
<Date>
"""
import math

import numpy as np
from scipy import linalg as la
from imageio.v2 import imread
from matplotlib import pyplot as plt
from scipy import sparse


# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    return np.diag(np.sum(A, axis=0)) - A


# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    # Compute Laplacian and eigenvalues
    L = laplacian(A)
    eigs = np.real(la.eigvals(L))
    eigs.sort()
    eig_vals = list(eigs.copy())

    # Compute the connectivity of the graph
    j = 0
    for i in range(len(eig_vals)):
        if eig_vals[j] < tol:
            eig_vals.remove(eig_vals[j])
            j -= 1
        j += 1
    alg_con = eig_vals[0]

    # Compute the number of connected components of the graph
    connected = 0
    for eig in eigs:
        if eig < tol:
            connected += 1

    return connected, alg_con


# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(int), R[mask]


# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        self.image = imread(filename)
        self.scaled = self.image / 255

        # If image is rgb compute brightness with scaled
        if len(self.image.shape) == 3:
            self.brightness = self.scaled.mean(axis=2)
        else:
            self.brightness = self.image
        self.brightness = np.ravel(self.brightness)
        self.height = len(self.image)
        self.width = len(self.image[0])

    # Problem 3
    def show_original(self):
        """Display the original image."""
        # If image is rgb disply normally otherwise use cmap
        if len(self.image.shape) == 3:
            plt.imshow(self.image)
        else:
            plt.imshow(self.image, cmap="gray")
        plt.axis("off")
        plt.show()

    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""
        A = sparse.lil_matrix((self.height * self.width, self.height * self.width))
        D = np.zeros(self.height * self.width)

        # Cycle through computing neighbors and distances and use those to build D and A matrices
        for i in range(self.height * self.width):
            neighbors, distances = get_neighbors(i, r, self.height, self.width)
            # D[i] = self.brightness[i]
            weights = np.exp((-abs(self.brightness[i] - self.brightness[neighbors]) / sigma_B2 - distances / sigma_X2))
            A[i, neighbors] = weights
            D[i] = sum(weights)

        A = sparse.csc_matrix(A)
        return A, D

    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        # Compute the matrix for the boolean mask
        L = sparse.csgraph.laplacian(A)
        D_half = sparse.diags(D ** -0.5)
        res = D_half @ L @ D_half

        # Compute the eigenvectors for D^(1/2)*L*D^(1/2)
        eig = sparse.linalg.eigsh(res, which="SM", k=2)[1][:, 1]
        eig = eig.reshape((self.height, self.width))
        mask = eig > 0
        return mask

    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""

        # Generate the mask using previous functions
        A, D = self.adjacency(r, sigma_B, sigma_X)
        mask = self.cut(A, D)
        if len(self.image.shape) == 3:
            mask = np.dstack([mask, mask, mask])

        # Plot the original image
        plt.subplot(131)
        if len(self.image.shape) == 3:
            plt.imshow(self.image)
        else:
            plt.imshow(self.image, cmap="gray")
        plt.axis("off")
        plt.title("Original")

        # Plot the Positive image
        plt.subplot(132)
        plt.imshow(mask * self.image)
        plt.axis("off")
        plt.title("Positive")

        # Plot the Negative image
        plt.subplot(133)
        plt.imshow(-(mask - 1) * self.image)
        plt.axis("off")
        plt.title("Negative")
        plt.suptitle("Image Segmentation")
        plt.show()


# if __name__ == '__main__':
#     ImageSegmenter("dream_gray.png").segment()
#     ImageSegmenter("dream.png").segment()
#     ImageSegmenter("monument_gray.png").segment()
#     ImageSegmenter("monument.png").segment()


def test_prob_1():
    A = np.random.random((4, 4))
    print(laplacian(A))


def test_prob_2():
    A = np.random.random((4, 4))
    B = np.array([[0, 3, 0, 0, 0, 0],
                  [3, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 1, 0, 2, .5],
                  [0, 0, 0, 2, 0, 1],
                  [0, 0, 0, .5, 1, 0]])
    print(connectivity(A))
    print(connectivity(B))


def test_prob_3():
    img_seg = ImageSegmenter("dream_gray.png")
    img_seg.show_original()


def test_prob_4():
    img_seg = ImageSegmenter("dream.png")
    img_seg.adjacency()

def test_prob_6():
    img_seg = ImageSegmenter("dream.png")
    img_seg.segment()

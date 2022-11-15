"""Volume 1: The SVD and Image Compression."""


import numpy as np
from scipy import linalg as la
from imageio import imread
from matplotlib import pyplot as plt
from scipy import sparse


# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """

    # Create AHA and eigenvalues/vectors - and I said HEYEAHYAEAHYAEAHYAEAH HEYEAHYEAH
    A_herm = A.conj().T
    eigenvalues, eigenvectors = la.eig(A_herm @ A)
    singular_vals = eigenvalues ** 0.5

    # Sort everything by singular values
    order = np.argsort(singular_vals)[::-1]
    eigenvectors = eigenvectors[:, order]
    singular_vals = singular_vals[order]

    # Get number of positive singular values and construct appropriate matrices
    r = np.sum(singular_vals > tol)
    sig = singular_vals[:r]
    V = eigenvectors[:, :r]
    U = np.dot(A, V) / sig
    return U, sig, V.conj().T


# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    # Compute space, E and S
    space = np.linspace(0, 2 * np.pi, 200)
    E = np.array([[1, 0, 0], [0, 0, 1]])
    S = np.row_stack((np.cos(space), np.sin(space)))

    # Compute the SVD
    U, s, Vh = la.svd(A)
    Sig = np.diag(s)

    # (a) S subplot
    plt.subplot(221)
    plt.plot(S[0, :], S[1, :])
    plt.plot(E[0, :], E[1, :])
    plt.title("(a) S")
    plt.axis("equal")

    # (b) VhS subplot
    VhS = np.dot(Vh, S)
    VhE = np.dot(Vh, E)
    plt.subplot(222)
    plt.plot(VhS[0, :], VhS[1, :])
    plt.plot(VhE[0, :], VhE[1, :])
    plt.title("(b) VhS")
    plt.axis("equal")

    # (c) SigVhS subplot
    SigVhS = np.dot(Sig, VhS)
    SigVhE = np.dot(np.dot(Sig, Vh), E)
    plt.subplot(223)
    plt.plot(SigVhS[0, :], SigVhS[1, :])
    plt.plot(SigVhE[0, :], SigVhE[1, :])
    plt.title("(c) SigVhS")
    plt.axis("equal")

    # (b) USigVhS subplot
    USigVhS = np.dot(U, SigVhS)
    USigVhE = np.dot(np.dot(U, np.dot(Sig, Vh)), E)
    plt.subplot(224)
    plt.plot(USigVhS[0, :], USigVhS[1, :])
    plt.plot(USigVhE[0, :], USigVhE[1, :])
    plt.title("(d) USigVhS")
    plt.axis("equal")

    plt.suptitle("Visualizing the SVD")
    plt.tight_layout()
    plt.show()


# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    # Raise error if s is greater than r
    if s > np.linalg.matrix_rank(A):
        raise ValueError("s is greater than the number of nonzero singular values")

    # Compute svd_approx
    U, S, Vh = la.svd(A)
    U_hat = U[:, :s]
    S_hat = S[:s]
    Vh_hat = Vh[:s, :] # e i e i o

    # Compute A_s approximation
    A_s = U_hat * S_hat @ Vh_hat
    return A_s, U_hat.size + S_hat.size + Vh_hat.size


# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    U, S, Vh = la.svd(A)

    # Find the right singular value representing the error
    if err <= S[-1]:
        raise ValueError("Epsilon is less than or equal to the smallest singular value of A")

    # Go back through problem 3 and compute the A_s approximation
    s = 0
    while S[s] > err:
        s += 1

    # Compute svd_approx
    U, S, Vh = la.svd(A)
    U_hat = U[:, :s]
    S_hat = S[:s]
    Vh_hat = Vh[:s, :]  # e i e i o

    # Compute A_s approximation
    A_s = U_hat * S_hat @ Vh_hat
    return A_s, U_hat.size + S_hat.size + Vh_hat.size


# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    image = imread(filename) / 255

    # Display for the color image
    if len(image.shape) == 3:
        red_image, r_s = svd_approx(image[:, :, 0], s)
        green_image, g_s = svd_approx(image[:, :, 1], s)
        blue_image, b_s = svd_approx(image[:, :, 2], s)

        image_s = np.dstack((red_image, green_image, blue_image))

        plt.subplot(121)
        plt.imshow(image)
        plt.axis("off")
        plt.title("Original image")

        plt.subplot(122)
        plt.imshow(image_s)
        plt.axis("off")
        plt.title("Rank " + str(s) + " Approximation")

        plt.suptitle("Approximation can be stored with " + str(image.size - (r_s + g_s + b_s)) + " fewer entries than the original")
        plt.show()

    # Display for the grayscale image
    else:
        image_s, s_s = svd_approx(image, s)

        plt.subplot(121)
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.title("Original image")

        plt.subplot(122)
        plt.imshow(image_s, cmap="gray")
        plt.axis("off")
        plt.title("Rank " + str(s) + " Approximation")

        plt.suptitle("Approximation can be stored with " + str(image.size - s_s) + " fewer entries than the original")
        plt.show()


def test_compact_svd():
    A = np.array([[5, 1, 7, 5, 2, 1, 6, 9, 1, 8, ],
                  [2, 9, 8, 8, 5, 7, 5, 3, 9, 8, ],
                  [8, 8, 4, 6, 9, 2, 9, 2, 6, 4, ],
                  [2, 8, 2, 5, 1, 7, 6, 9, 2, 5, ],
                  [9, 4, 6, 5, 6, 9, 8, 4, 9, 3, ]])
    U, s, Vh = compact_svd(A)
    print(U.shape, s.shape, Vh.shape)
    print(U.T @ U)
    print(np.allclose(U.T @ U, np.identity(5)))
    print(np.allclose(U @ np.diag(s) @ Vh, A))
    print(np.linalg.matrix_rank(A) == len(s))


def test_svd_approx():
    A = np.array([[5, 1, 7, 5, 2, 1, 6, 9, 1, 8, ],
                  [2, 9, 8, 8, 5, 7, 5, 3, 9, 8, ],
                  [8, 8, 4, 6, 9, 2, 9, 2, 6, 4, ],
                  [2, 8, 2, 5, 1, 7, 6, 9, 2, 5, ],
                  [9, 4, 6, 5, 6, 9, 8, 4, 9, 3, ]])
    A_s, entry_count = svd_approx(A, 3)
    print(A_s)
    print(entry_count)


def test_lowest_rank_approx():
    A = np.array([[5, 1, 7, 5, 2, 1, 6, 9, 1, 8, ],
                  [2, 9, 8, 8, 5, 7, 5, 3, 9, 8, ],
                  [8, 8, 4, 6, 9, 2, 9, 2, 6, 4, ],
                  [2, 8, 2, 5, 1, 7, 6, 9, 2, 5, ],
                  [9, 4, 6, 5, 6, 9, 8, 4, 9, 3, ]])
    A_s, entry_count = lowest_rank_approx(A, 7)
    print(A_s)
    print(entry_count)


def test_visualize_svd():
    A = np.array([[3, 1], [1, 3]])
    visualize_svd(A)


def test_compress_image():
    compress_image("hubble_gray.jpg", 45)
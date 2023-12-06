"""
SubTask 2A: Random Matrix (random_eigen.py)
Take the dimensions of matrix n as input.
Produce a random n x n invertible matrix A. For the purpose of demonstrating, every cell of A will be an integer.
Perform Eigen Decomposition using NumPy's library function 
Reconstruct A from eigenvalues and eigenvectors (refer to Section 2.7).
Check if the reconstruction worked properly. (np.allclose will come in handy.)
You should be able to explain how your code ensures that the way you generated A ensures invertibility. 
"""
import numpy as np

np.random.seed(42)
np.set_printoptions(precision=3, suppress=True, linewidth=120)


def random_invertible_matrix(n):
    A = np.random.randint(1, 100, size=(n, n))
    while np.linalg.det(A) == 0:
        A = np.random.randint(1, 100, size=(n, n))
    return A


def strictly_diagonally_dominant_matrix(n):
    """
    See: https://en.wikipedia.org/wiki/Diagonally_dominant_matrix
    A diagonally dominant matrix is guaranteed to be invertible.
    This function avoids the use of while loop and is more efficient.
    """
    A = np.random.randint(-50, 50, size=(n, n))
    np.fill_diagonal(A, 0)
    sums = np.sum(np.abs(A), axis=1)
    sign_flips = np.random.choice([-1, 1], size=n)
    diag_entries = sums + np.random.randint(1, 10, size=n)
    diag_entries = diag_entries * sign_flips
    np.fill_diagonal(A, diag_entries)
    return A


n = int(input("Enter the dimension of the matrix: "))
# A = random_invertible_matrix(n)

A = strictly_diagonally_dominant_matrix(n)
assert np.linalg.det(A) != 0, "A is not invertible"


print("Matrix A is: \n", np.array_str(A), "\n")

eigenvalues, eigenvectors = np.linalg.eig(A)
print("Eigenvalues: \n", np.array_str(eigenvalues), "\n")
print("Eigenvectors: \n", np.array_str(eigenvectors), "\n")

A_reconstructed = eigenvectors @ np.diag(eigenvalues) @ np.linalg.inv(eigenvectors)

# use np.allclose to check if the reconstruction worked properly
print("Is the reconstruction correct? ", np.allclose(A, A_reconstructed), "\n")
print("Reconstructed A: \n", np.array_str(A_reconstructed))

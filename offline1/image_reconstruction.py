"""
Task 3: Image Reconstruction using Singular Value Decomposition 
(image_reconstruction.py)
Take a photo of a book's cover within your vicinity. Let's assume it is named image.jpg. 
Use OpenCV or similar frameworks to read image.jpg. Transform it to grayscale using functions such as cv2.cvtColor(). If you wish, resize to lower dimensions (~500) for faster computation. 
The grayscale image will be an n x m matrix A. 
Perform Singular Value Decomposition using NumPy's library function.
Given a matrix A and an integer k, write a function low_rank_approximation(A, k) that returns the k-rank approximation of A.
Now vary the value of k from 1 to min(n, m) (take at least 10 such values in the interval). In each case, plot the resultant k-rank approximation as a grayscale image. Observe how the images vary with k.  You can find a sample intended output in the shared folder. 
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

def low_rank_approximation(A, k):
    U, S, V = np.linalg.svd(A, full_matrices=True)
    A_k = U[:, :k] @ np.diag(S[:k]) @ V[:k, :]
    return A_k

raw_img = cv2.imread("image.jpg")
# rescale to 400x300
img = cv2.resize(raw_img, (600, 800))
# convert to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# convert to numpy array
array = np.array(img)
print("Shape of array: ", array.shape)

# perform SVD
U, S, V = np.linalg.svd(array, full_matrices=True)

# Vary k from 1 to min(n, m) and plot the resultant k-rank approximation as a grayscale image
fig = plt.figure(figsize=(10, 10))
for idx, k in enumerate(range(10, 101, 10), start=1):
    print("k = ", k)
    # A_k = U[:, :k] @ np.diag(S[:k]) @ V[:k, :] # This is more efficient since it doesn't require performing SVD
    A_k = low_rank_approximation(array, k) # But spec says to write a function 😞
    fig.add_subplot(3, 5, idx)
    plt.imshow(A_k, cmap='gray')
    plt.title("k = " + str(k))

end = min(array.shape[0], array.shape[1])
for idx, k in enumerate(range(200, end + 1, 100), start=11):
    print("k = ", k)
    # A_k = U[:, :k] @ np.diag(S[:k]) @ V[:k, :] # This is more efficient since it doesn't require performing SVD
    A_k = low_rank_approximation(array, k) # But spec says to write a function
    fig.add_subplot(3, 5, idx)
    plt.imshow(A_k, cmap='gray')
    plt.title("k = " + str(k))

# turn off axis and labels
for ax in fig.axes:
    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
plt.tight_layout()

# save as pdf
plt.savefig("image_reconstruction.pdf", bbox_inches='tight', pad_inches=0.5, dpi=500)
print("Plot saved as image_reconstruction.pdf")
# plt.show()
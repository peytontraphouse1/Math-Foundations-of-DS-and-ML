"""
Homework 2c
Math 528: Mathematical Foundations of ML and DS
Instructor: Mitchel Colebank
Date edited: 2/5/2026
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --------------------------------------------------
# Load images and stack as column vectors
# --------------------------------------------------

# Each image is reshaped from 28x28 to a 28^2 vector
fname = "HW2/mnist_4/img_"
digits_stack = np.zeros((28 * 28, 12))

for i in range(1, 13):  # MATLAB: for i = 1:12
    fname_load = f"{fname}{i}.jpg"
    img = Image.open(fname_load).convert("L")  # grayscale
    h = np.array(img, dtype=float)
    digits_stack[:, i - 1] = h.reshape(-1)     # vectorize image

# --------------------------------------------------
# Compute outer product and center the data
# --------------------------------------------------

# Outer product (pixel-pixel relationships)
X = digits_stack @ digits_stack.T

# Mean across second dimension (axis=1 in Python)
mean_dig = np.mean(X, axis=1)

# Center the data
Xcentered = X - mean_dig[:, None]   # ensure correct broadcasting

# SVD of centered data
Ux, Sx, Vxt = np.linalg.svd(Xcentered, full_matrices=True)

# --------------------------------------------------
# Visualize the mean image
# --------------------------------------------------

mean_dig_square = mean_dig.reshape((28, 28))

plt.figure(1)
plt.imshow(mean_dig_square, cmap="gray")
plt.axis("off")
plt.title("Mean image")
plt.show()

# --------------------------------------------------
# Visualize the first 9 eigenimages
# --------------------------------------------------

plt.figure(2)
for i in range(9):  # MATLAB: for i = 1:9
    eig_img = Ux[:, i].reshape((28, 28))
    plt.subplot(3, 3, i + 1)
    plt.imshow(eig_img, cmap="gray")
    plt.axis("off")
    plt.title(f"Component {i+1}")

plt.show()

import cv2
import numpy as np
from Src.noise import add_salt_pepper_noise

## Colored Images
original_ctx = "Original Image"
original_img = "Images/5.png"
# original = None

gaussian_ctx = "Guassian Image"
gaussian_img = "Images/5_gaussian.png"
# gaussian = None

saltpepp_ctx = "Salt & Papper Image"
saltpepp_img = "Images/5_salt_pepper.png"
# saltpepp = None

# Original
original = cv2.imread(original_img, cv2.IMREAD_COLOR)
# original = cv2.cvtColor(original, cv2.IMREAD_GRAYSCALE)
original_shape = np.shape(original)
original_grey = cv2.imread(original_img, cv2.IMREAD_GRAYSCALE)

# Gaussian
gaussian = cv2.GaussianBlur(original, (7,7), 30)
# gaussian = cv2.GaussianBlur(original, (9,9), sigmaX=30, sigmaY=30)
# gaussian = cv2.GaussianBlur(original, (31,31), sigmaX=50, sigmaY=50)
# gaussian_grey = cv2.cvtColor(gaussian, cv2.IMREAD_GRAYSCALE)
gaussian_shape = np.shape(gaussian)
cv2.imwrite(gaussian_img, gaussian)
gaussian_grey = cv2.imread(gaussian_img, cv2.IMREAD_GRAYSCALE)
assert gaussian_shape == original_shape

# Salt & Pepper
saltpepp = add_salt_pepper_noise(original)
# saltpepp_grey = cv2.cvtColor(saltpepp, cv2.IMREAD_GRAYSCALE)
saltpepp_shape = np.shape(saltpepp)
cv2.imwrite(saltpepp_img, saltpepp)
saltpepp_grey = cv2.imread(saltpepp_img, cv2.IMREAD_GRAYSCALE)
assert saltpepp_shape == original_shape
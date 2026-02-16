import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from Src.helper_functions import show_window

def median_filter(image, kernel_size=5):
    h, w = image.shape
    pad = kernel_size // 2
    filtered = np.zeros_like(image)
    padded = np.pad(image, pad, mode='edge')
    for i in range(h):
        for j in range(w):
            window = padded[i:i + kernel_size, j:j + kernel_size]
            filtered[i, j] = np.median(window)
    return filtered

##########

def gaussian_filter(image, kernel_size=5, sigma=0):
    return cv2.GaussianBlur(image, (kernel_size,kernel_size), sigma)

# def gaussian_filter(image, kernel_size=5, sigma=0):
#     if sigma <= 0:
#         sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
#
#     # Δημιουργία 2D Gaussian kernel
#     center = kernel_size // 2
#     kernel = np.zeros((kernel_size, kernel_size))
#     for i in range(kernel_size):
#         for j in range(kernel_size):
#             x = i - center
#             y = j - center
#             kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
#     kernel = kernel / kernel.sum()
#     # Συνέλιξη
#     # output = np.convolve(image, kernel)
#     pad = kernel_size // 2
#     padded = np.pad(image, pad, mode='edge')
#     output = np.zeros_like(image, dtype=np.float64)
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             region = padded[i:i + kernel_size, j:j + kernel_size]
#             output[i, j] = np.sum(region * kernel)
#     return output.astype(np.uint8)

def smooth(profile, r=3):
        kernel = np.ones(2 * r + 1, dtype=np.float64) / (2 * r + 1)
        return np.convolve(profile, kernel, mode='same')

# For Report
# salt_pepper_original = cv2.imread('../Images/5_salt_pepper.png')
# salt_pepper = cv2.cvtColor(salt_pepper_original, cv2.COLOR_BGR2RGB)
# clear = []
# fig, ax = plt.subplots(ncols=2, nrows=5, figsize=(15, 15))
# ax_flat = ax.flatten()
# for i in range(0, 10):
#     n = 2 * i + 1
#     result = cv2.medianBlur(salt_pepper[250:350], n)
#     cv2.imwrite(f'../Report/gaussian/{n}.png', result)
#     show_window(f'Kernel: {n}x{n}',result)
#     clear.append(result)
#     ax_flat[i].imshow(result)
#     ax_flat[i].set_title(f'Kernel: {n}x{n}')
#     ax_flat[i].axis('off')
# plt.tight_layout()
# plt.show()
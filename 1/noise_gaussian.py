import cv2
from matplotlib import pyplot as plt
from Src.filters import gaussian_filter
from Src.helper_functions import histogram_extract, show_window, show_window_ratio
from noise_free import exercise

DEBUG = False
# DEBUG = True

kernel_size = 5
sigma = 2

original_image_ctx = "Original Image"
original_image = f"./Images/5.png"
original = cv2.imread(original_image, cv2.IMREAD_GRAYSCALE)

assert original is not None

filtered = gaussian_filter(original, kernel_size, sigma)

# if DEBUG:
#     x, y, w, h = 250, 250, 350, 350
#
#     fig_hist, ax_hist = plt.subplots(3, 5, figsize=(10, 10))
#     ax_hist = ax_hist.flatten()
#
#     fig, ax = plt.subplots(3, 5, figsize=(10, 10))
#     ax = ax.flatten()
#     kernel_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
#     for i, ksize in enumerate(kernel_sizes):
#         filtered = gaussian_filter(original, ksize, 0)
#         filtered_hist = histogram_extract(filtered)
#         subsection = filtered[y:y + h, x:x + w]
#         ax[i].imshow(subsection, cmap='gray')
#         ax[i].set_title(f'Kernel: {ksize}', color="red")
#         ax[i].axis('off')
#
#         ax_hist[i].plot(filtered_hist[2], 'r')
#         ax_hist[i].set_ylabel('# of Pixels')
#         ax_hist[i].set_xlabel('Pixel intensity')
#     plt.tight_layout()
#     plt.show()
#
#     filtered = gaussian_filter(original, kernel_size, 3)
#
#     # Histogram extraction with and without the noise
#     original_hist = histogram_extract(original)
#     filtered_hist = histogram_extract(filtered)
#
#     plt.figure()
#     plt.subplot(221)
#     plt.imshow(original[200:500, 200:500], cmap="grey")
#
#     plt.subplot(223)
#     plt.plot(original_hist[2], color='r')
#     plt.plot(original_hist[0])
#     plt.xlabel('Pixel intensity')
#     plt.ylabel('# of Pixels')
#     plt.subplot(222)
#     plt.imshow(filtered[200:500, 200:500], cmap="grey")
#
#     plt.subplot(224)
#     plt.plot(filtered_hist[2], color='r')
#     plt.plot(filtered_hist[0])
#     plt.xlabel('Pixel intensity')
#     plt.ylabel('# of Pixels')
#
#     plt.show()

# show_window_ratio('Original', original)
# show_window_ratio("Restored", filtered)

final,morph = exercise(filtered)
assert final.any()
assert morph.any()
if DEBUG:
    cv2.imwrite(f'./Report/gaussian/final_{kernel_size}.png', final)
    cv2.imwrite(f'./Report/gaussian/morph_{kernel_size}.png', morph)
import cv2
import numpy as np

def histogram_extract(img):
    histr = cv2.calcHist([img], [0], None, [256], [0, 256])
    cdf = histr.cumsum()
    cdf_norm = cdf * float(histr.max()) / cdf.max()
    return (histr, cdf, cdf_norm)

def show_window_ratio(title, image):
    cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO)
    cv2.imshow(title, image)
    cv2.waitKey(0)

def show_window(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
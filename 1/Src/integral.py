import cv2
import numpy as np

def integral(image):
    int = np.cumsum(image, axis=0, dtype=np.float32)
    int = np.cumsum(int, axis=1, dtype=np.float32)
    return np.pad(int, ((1,0),(1,0)),mode='constant')

# def integral_sum()
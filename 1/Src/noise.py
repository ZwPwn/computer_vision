import cv2
import numpy as np

def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    sp = image.copy()
    salt = np.random.random(image.shape[:2])< salt_prob
    sp[salt] = 255
    pepper = np.random.random(image.shape[:2]) < pepper_prob
    sp[pepper] = 0
    return sp

# def add_gaussian_noise(image):
#
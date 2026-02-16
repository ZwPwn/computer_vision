from os import path, listdir
import numpy as np
import cv2

def read_images(txt_path):
    img_names = [path.join(txt_path, img) for img in listdir(txt_path)]
    images = [cv2.imread(img) for img in img_names[:8]]
    return np.array(images)
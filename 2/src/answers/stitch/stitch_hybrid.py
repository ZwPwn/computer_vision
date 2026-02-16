import numpy as np
from src.answers.image_operations import crop_image
from src.answers.stitch.stitch_planar import planar_stitch
from src.answers.stitch.stitch_cylindrical import calculate_intrinsics, cylindrical_warp

def cylindrical_planar_stitch(images, kp_alg: str = 'sift', K=None):
    if K is None:
        K = calculate_intrinsics(images[0].shape[1], images[0].shape[0])
    images = [crop_image(cylindrical_warp(img, K)) for img in images]
    images = np.array(images)
    return planar_stitch(images, kp_alg=kp_alg)
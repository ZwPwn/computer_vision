import cv2
import numpy as np
from os import makedirs

from src.answers.stitch.stitch_planar import planar_stitch
from src.answers.stitch.stitch_cylindrical import cylindrical_stitch
from src.answers.stitch.stitch_hybrid import cylindrical_planar_stitch
from src.helper.io import read_images


############ OpenPano
# kp_alg = 'sift'
# K = None
# fov = 60 # Τυπικό για κάμερες
# txt_path = "./images/OpenPano_medium"
# dest_dir = "./out/OpenPano_medium/sift"
# makedirs(dest_dir, exist_ok=True)
# images = read_images(txt_path)
# p_pano = planar_stitch(images)
# cv2.imwrite(dest_dir + "/planar_pano.jpg", p_pano)
# h_pano = cylindrical_planar_stitch(images, kp_alg=kp_alg, K=K)
# cv2.imwrite(dest_dir + "/hybrid_pano.jpg", h_pano)
# c_pano = cylindrical_stitch(images, kp_alg=kp_alg, fov=fov)
# cv2.imwrite(dest_dir + f"/cylindrical_pano.jpg", c_pano)
# kp_alg = 'surf'
# K = None
# fov = 60 # Τυπικό για κάμερες
# txt_path = "./images/OpenPano_medium"
# dest_dir = "./out/OpenPano_medium/surf"
# makedirs(dest_dir, exist_ok=True)
# images = read_images(txt_path)
# p_pano = planar_stitch(images, kp_alg=kp_alg)
# cv2.imwrite(dest_dir + "/planar_pano.jpg", p_pano)
# h_pano = cylindrical_planar_stitch(images, kp_alg=kp_alg, K=K)
# cv2.imwrite(dest_dir + "/hybrid_pano.jpg", h_pano)
# c_pano = cylindrical_stitch(images, kp_alg=kp_alg, fov=fov)
# cv2.imwrite(dest_dir + f"/cylindrical_pano.jpg", c_pano)
# K = None
# fov = 60 # Τυπικό για κάμερες
# txt_path = "./images/OpenPano_medium"
# dest_dir = "./out/OpenPano_medium/orb"
# makedirs(dest_dir, exist_ok=True)
# images = read_images(txt_path)
# p_pano = planar_stitch(images)
# cv2.imwrite(dest_dir + "/planar_pano.jpg", p_pano)
# h_pano = cylindrical_planar_stitch(images, kp_alg=kp_alg, K=K)
# cv2.imwrite(dest_dir + "/hybrid_pano.jpg", h_pano)
# c_pano = cylindrical_stitch(images, kp_alg=kp_alg, fov=fov)
# cv2.imwrite(dest_dir + f"/cylindrical_pano.jpg", c_pano)
############ NIS
# kp_alg = 'sift'
# K = None
# fov = 60 # Τυπικό για κάμερες
# txt_path = "./images/NISwGP-02_Uffizi"
# dest_dir = "./out/NISwGP-02_Uffizi/sift"
# makedirs(dest_dir, exist_ok=True)
# images = read_images(txt_path)
# # p_pano = planar_stitch(images, kp_alg=kp_alg)
# # cv2.imwrite(dest_dir + "/planar_pano.jpg", p_pano)
# h_pano = cylindrical_planar_stitch(images, kp_alg=kp_alg, K=K)
# cv2.imwrite(dest_dir + "/hybrid_pano.jpg", h_pano)
# c_pano = cylindrical_stitch(images, kp_alg=kp_alg, fov=fov)
# cv2.imwrite(dest_dir + f"/cylindrical_pano.jpg", c_pano)
# kp_alg = 'surf'
# K = None
# fov = 60 # Τυπικό για κάμερες
# dest_dir = "./out/NISwGP-02_Uffizi/surf"
# makedirs(dest_dir, exist_ok=True)
# images = read_images(txt_path)
# # p_pano = planar_stitch(images, kp_alg=kp_alg)
# # cv2.imwrite(dest_dir + "/planar_pano.jpg", p_pano)
# h_pano = cylindrical_planar_stitch(images, kp_alg=kp_alg, K=K)
# cv2.imwrite(dest_dir + "/hybrid_pano.jpg", h_pano)
# c_pano = cylindrical_stitch(images, kp_alg=kp_alg, fov=fov)
# cv2.imwrite(dest_dir + f"/cylindrical_pano.jpg", c_pano)
# K = None
# fov = 60 # Τυπικό για κάμερες
# dest_dir = "./out/NISwGP-02_Uffizi/orb"
# makedirs(dest_dir, exist_ok=True)
# images = read_images(txt_path)
# # p_pano = planar_stitch(images)
# # cv2.imwrite(dest_dir + "/planar_pano.jpg", p_pano)
# h_pano = cylindrical_planar_stitch(images, kp_alg=kp_alg, K=K)
# cv2.imwrite(dest_dir + "/hybrid_pano.jpg", h_pano)
# c_pano = cylindrical_stitch(images, kp_alg=kp_alg, fov=fov)
# cv2.imwrite(dest_dir + f"/cylindrical_pano.jpg", c_pano)
#
# ############ UAV
# ## SIFT
# kp_alg = 'sift'
# K = None
# fov = 66.36
# txt_path = "images/uav_orthophotomap"
# dest_dir = "./out/uav_orthophotomap/sift"
# makedirs(dest_dir, exist_ok=True)
# images = read_images(txt_path)
# p_pano = planar_stitch(images)
# cv2.imwrite(dest_dir + "/planar_pano.jpg", p_pano)
# h_pano = cylindrical_planar_stitch(images, kp_alg=kp_alg, K=K)
# cv2.imwrite(dest_dir + "/hybrid_pano.jpg", h_pano)
# c_pano = cylindrical_stitch(images, kp_alg=kp_alg, fov=fov)
# cv2.imwrite(dest_dir + f"/cylindrical_pano.jpg", c_pano)
# ## SURF
# kp_alg = 'surf'
# K = None
# fov = 66.36
# dest_dir = "./out/uav_orthophotomap/surf"
# makedirs(dest_dir, exist_ok=True)
# images = read_images(txt_path)
# p_pano = planar_stitch(images)
# cv2.imwrite(dest_dir + "/planar_pano.jpg", p_pano)
# h_pano = cylindrical_planar_stitch(images, kp_alg=kp_alg, K=K)
# cv2.imwrite(dest_dir + "/hybrid_pano.jpg", h_pano)
# c_pano = cylindrical_stitch(images, kp_alg=kp_alg, fov=fov)
# cv2.imwrite(dest_dir + f"/cylindrical_pano.jpg", c_pano)
# ## ORB
# kp_alg = 'orb'
# K = None
# fov = 66.36
# dest_dir = "./out/uav_orthophotomap/orb"
# makedirs(dest_dir, exist_ok=True)
# images = read_images(txt_path)
# p_pano = planar_stitch(images)
# cv2.imwrite(dest_dir + "/planar_pano.jpg", p_pano)
# h_pano = cylindrical_planar_stitch(images, kp_alg=kp_alg, K=K)
# cv2.imwrite(dest_dir + "/hybrid_pano.jpg", h_pano)
# c_pano = cylindrical_stitch(images, kp_alg=kp_alg, fov=fov)
# cv2.imwrite(dest_dir + f"/cylindrical_pano.jpg", c_pano)
# for fov in range(10,60,10):
#     c_pano = cylindrical_stitch(images,fov)
#     cv2.imwrite(dest_dir + f"/cylindrical_{fov}_pano.jpg", c_pano)

############ Custom


K = np.array([
    [6.8091007065400277e+02,0.,2.7057136369831328e+03],
    [0,3.1530830738477102e+03,2.1317269090366208e+03],
    [0,0,1]
], dtype=np.float64)

kp_alg = 'sift'
txt_path = "./images/Custom_Panorama/1"
dest_dir = "./out/Custom_Panorama/1/sift"
makedirs(dest_dir, exist_ok=True)
images = read_images(txt_path)
p_pano = planar_stitch(images, kp_alg=kp_alg)
cv2.imwrite(dest_dir + "/planar_pano.jpg", p_pano)
h_pano = cylindrical_planar_stitch(images, kp_alg=kp_alg, K=K)
cv2.imwrite(dest_dir + "/hybrid_pano.jpg", h_pano)
c_pano = cylindrical_stitch(images, kp_alg=kp_alg)
cv2.imwrite(dest_dir + f"/cylindrical_pano.jpg", c_pano)

kp_alg = 'surf'
txt_path = "./images/Custom_Panorama/1"
dest_dir = "./out/Custom_Panorama/1/surf"
makedirs(dest_dir, exist_ok=True)
images = read_images(txt_path)
p_pano = planar_stitch(images, kp_alg=kp_alg)
cv2.imwrite(dest_dir + "/planar_pano.jpg", p_pano)
h_pano = cylindrical_planar_stitch(images, kp_alg=kp_alg, K=K)
cv2.imwrite(dest_dir + "/hybrid_pano.jpg", h_pano)
c_pano = cylindrical_stitch(images, kp_alg=kp_alg)
cv2.imwrite(dest_dir + f"/cylindrical_pano.jpg", c_pano)

kp_alg = 'orb'
txt_path = "./images/Custom_Panorama/1"
dest_dir = "./out/Custom_Panorama/1/orb"
makedirs(dest_dir, exist_ok=True)
images = read_images(txt_path)
p_pano = planar_stitch(images, kp_alg=kp_alg)
cv2.imwrite(dest_dir + "/planar_pano.jpg", p_pano)
h_pano = cylindrical_planar_stitch(images, kp_alg=kp_alg, K=K)
cv2.imwrite(dest_dir + "/hybrid_pano.jpg", h_pano)
c_pano = cylindrical_stitch(images, kp_alg=kp_alg)
cv2.imwrite(dest_dir + f"/cylindrical_pano.jpg", c_pano)
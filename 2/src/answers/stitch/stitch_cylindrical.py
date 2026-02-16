import cv2
import numpy as np
import math
from src.answers.bfmatcher import findHomographyFromImages
from src.answers.image_operations import crop_image

def calculate_intrinsics(width, height, fov=60):
    fov_rad = np.deg2rad(fov)
    focal_len = width / (2 * math.tan(fov_rad / 2))
    return np.array([[focal_len, 0, width / 2], [0, focal_len, height / 2], [0, 0, 1]])


def cylindrical_warp(img, K, kp_alg:str = 'sift'):
    h_img, w_img = img.shape[:2]
    foc_len = (K[0][0] + K[1][1]) / 2

    cylinder = np.zeros_like(img)

    # Batch processing σε δείγματα των 200 για εξοικονόμηση μνήμης
    CHUNK = 200
    for y_start in range(0, h_img, CHUNK):
        y_end = min(y_start + CHUNK, h_img)
        h_chunk = y_end - y_start

        # Grid for this chunk only
        x, y = np.meshgrid(np.arange(w_img), np.arange(y_start, y_end))
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        theta = (x - K[0][2]) / foc_len
        h_cyl = (y - K[1][2]) / foc_len

        # 3D points
        p = np.dstack((np.sin(theta), h_cyl, np.cos(theta)))
        del x, y, theta, h_cyl

        # Project
        p = p.reshape(-1, 3)
        image_points = p.dot(K.T)
        del p

        # Normalize
        z = image_points[:, 2]
        z[z == 0] = 1  # Avoid div by zero
        map_x = (image_points[:, 0] / z).reshape(h_chunk, w_img).astype(np.float32)
        map_y = (image_points[:, 1] / z).reshape(h_chunk, w_img).astype(np.float32)
        del image_points, z

        # Remap
        cylinder[y_start:y_end] = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
        del map_x, map_y

    return cylinder


def cylindrical_stitch(images,kp_alg:str='sift',K=None,fov=60):
    # 1. Init
    h_orig, w_orig = images[0].shape[:2]
    if K is None:
        K = calculate_intrinsics(w_orig, h_orig, fov)

    # 2. Pre-warp ALL images to clean lists
    print("Pre-warping images...")
    cyl_images = [cylindrical_warp(img, K) for img in images]

    # 3. Create Canvas
    canvas_w = int(2 * np.pi * K[0][0])
    canvas_h = h_orig + 400
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # 4. Place First Image
    y_offset = 200
    img0 = cyl_images[0]
    canvas[y_offset:y_offset + img0.shape[0], 0:img0.shape[1]] = img0

    # 5. Global Accumulators
    global_x = 0
    global_y = y_offset

    print("Stitching...")

    for i in range(1, len(cyl_images)):
        img_prev = cyl_images[i - 1]
        img_curr = cyl_images[i]

        H, pt1, pt2 = findHomographyFromImages(img_prev, img_curr, kp_alg=kp_alg)

        # Σχετικό Shift από την ομογραφία (θεωρώ τέλειο κύλινδρο χωρίς pitch / yaw)
        dx = H[0, 2]
        dy = H[1, 2]

        # Συσσορευμένο drift
        global_x += dx
        global_y += dy

        print(f"Image {i}: Rel Shift {dx:.1f} -> Global X {global_x:.1f}")

        # 1. Υπολογίζω συντεταγμένες
        h, w = img_curr.shape[:2]
        y_start = int(global_y)
        x_start = int(global_x)
        y_end = y_start + h
        x_end = x_start + w

        img_y_start = 0
        img_x_start = 0

        if y_start < 0:
            img_y_start = -y_start
            y_start = 0
        if x_start < 0:
            img_x_start = -x_start
            x_start = 0
        if y_end > canvas_h: y_end = canvas_h
        if x_end > canvas_w: x_end = canvas_w

        # 3. Επικόλλησε έγκυρη περιοχή
        h_slice = y_end - y_start
        w_slice = x_end - x_start

        if h_slice > 0 and w_slice > 0:
            # Πάρε "φέτες" για εξοικονόμηση μνήμης
            img_slice = img_curr[img_y_start:img_y_start + h_slice, img_x_start:img_x_start + w_slice]
            canvas_roi = canvas[y_start:y_end, x_start:x_end]

            # Μάσκα για τις "φέτες"
            gray_slice = cv2.cvtColor(img_slice, cv2.COLOR_BGR2GRAY)
            _, mask_slice = cv2.threshold(gray_slice, 1, 255, cv2.THRESH_BINARY)

            # Boolean indexing (για τις συγκεκριμένες φωτογραφίες χειρότερο σενάριο 128MiB / 230 MiB
            mask_bool = mask_slice > 0
            canvas_roi[mask_bool] = img_slice[mask_bool]

            # Αποθήκευσε
            canvas[y_start:y_end, x_start:x_end] = canvas_roi

    return crop_image(canvas)
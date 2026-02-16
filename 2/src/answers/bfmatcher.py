import cv2
import numpy as np
from typing import Tuple

# Στην python μπορώ να ορίσω τύπους / τα stub της opencv 3.4.2 δεν λειτουργούν σωστά.
def findHomographyFromImages(
        img1: np.ndarray,
        img2: np.ndarray,
        kp_alg: str = 'sift',
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Low level: Continuous memory (handling smaller FOVs)
    img1 = np.ascontiguousarray(img1)
    img2 = np.ascontiguousarray(img2)

    # Κλιμάκωση προς τα κάατω σε περίπτωση που έχω πολύ μεγάλη εικόνα
    MAX_DIM = 3000

    def get_scale_and_resized(img):
        h, w = img.shape[:2]
        if max(h, w) > MAX_DIM:
            scale = MAX_DIM / float(max(h, w))
            small = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            return scale, small
        return 1.0, img

    scale1, small1 = get_scale_and_resized(img1)
    scale2, small2 = get_scale_and_resized(img2)

    max_features = 1000

    # --- 1. Feature Detection ---
    if kp_alg == 'sift':
        detector = cv2.xfeatures2d.SIFT_create(nfeatures=max_features)
    elif kp_alg == 'surf':
        detector = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
    elif kp_alg == 'orb':
        detector = cv2.ORB_create(nfeatures=max_features)
    else:
        detector = cv2.xfeatures2d.SIFT_create(nfeatures=max_features)

    gray1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)

    # ΑΣΦΑΛΕΙΑ 2: Διασφάλησε συνεχή περιοχή μνήμης για το ένα κανάλι επίσης
    gray1 = np.ascontiguousarray(gray1)
    gray2 = np.ascontiguousarray(gray2)

    # print(gray1.shape, gray2.shape)
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)

    # assert για να έχω τουλάχιστον δυο ζεύγη περιγραφέων
    assert des1 is not None or des2 is not None and (len(des1) >= 4 and len(des2) >= 4)

    N1, N2 = des1.shape[0], des2.shape[0]

    # 2. Υπολογισμός απόστασης με αλγόριθμο M
    dist_matrix = None

    if kp_alg in ['sift', 'surf']:
        des1_f = des1.astype(np.float32)
        des2_f = des2.astype(np.float32)

        sum_sq1 = np.sum(des1_f ** 2, axis=1).reshape(N1, 1)
        sum_sq2 = np.sum(des2_f ** 2, axis=1).reshape(1, N2)
        dot_prod = np.matmul(des1_f, des2_f.T)

        dist_sq = np.maximum(sum_sq1 + sum_sq2 - 2 * dot_prod, 0)
        dist_matrix = np.sqrt(dist_sq)

    elif kp_alg == 'orb':
        xor_res = cv2.bitwise_xor(des1[:, None, :], des2[None, :, :])
        bits = np.unpackbits(xor_res, axis=2)
        dist_matrix = np.sum(bits, axis=2)

    # 3. Cross-Check + Lowe's Ratio (Από το paper του Lowe του 2004)
    sorted_indices = np.argsort(dist_matrix, axis=1)

    forward_idx = sorted_indices[:, 0]
    second_idx = sorted_indices[:, 1]

    backward_idx = np.argmin(dist_matrix, axis=0)

    good_matches = []
    ratio_thresh = 0.75 # Lowe's Ratio

    for i in range(N1):
        j = forward_idx[i]
        k = second_idx[i]

        dist1 = dist_matrix[i, j]
        dist2 = dist_matrix[i, k]

        # Συνθήκη 1: Lowe's Ratio Test
        if dist1 < ratio_thresh * dist2:
            # Συνθήκη 2: (Cross Check)
            if backward_idx[j] == i:
                good_matches.append(cv2.DMatch(_queryIdx=i, _trainIdx=j, _distance=dist1))

    assert len(good_matches) >= 4   # Κάνω assert για να γίνει panic σε περίπτωση που δεν βρω τουλάχιστον δυο ζεύγη

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    print(f"Good matches (Filtered) = {len(good_matches)}")

    # 4. Υποκλιμάκωση και Ομογραφία
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2) * (1.0 / scale1)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2) * (1.0 / scale2)

    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

    return H, src_pts, dst_pts


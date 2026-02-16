import cv2
import numpy as np
from src.answers.bfmatcher import findHomographyFromImages
from src.answers.image_operations import crop_image


# def stitch_two_images(img1, img2, offset=(0, 0)):
#     y_off, x_off = offset
#     h1, w1 = img1.shape[:2]
#     h2, w2 = img2.shape[:2]
#
#     # Υπολογισμός ορίων επικάλυψης για να δουλέψουμε μόνο στο slice
#     y1 = max(y_off, 0)
#     x1 = max(x_off, 0)
#     y2 = min(y_off + h1, h2)
#     x2 = min(x_off + w1, w2)
#
#     if y1 >= y2 or x1 >= x2:
#         return img2
#
#     # Αντιστοίχιση συντεταγμένων στην εικόνα επικάλυψης (img1)
#     img1_y1 = y1 - y_off
#     img1_x1 = x1 - x_off
#     img1_y2 = img1_y1 + (y2 - y1)
#     img1_x2 = img1_x1 + (x2 - x1)
#
#     # Slice μόνο στα pixels που χρειάζονται (Reference only - Zero Extra RAM)
#     img1_slice = img1[img1_y1:img1_y2, img1_x1:img1_x2]
#     roi = img2[y1:y2, x1:x2]
#
#     # Δημιουργία μάσκας μόνο για το slice
#     gray = cv2.cvtColor(img1_slice, cv2.COLOR_BGR2GRAY)
#     _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
#
#     # Painter's Algorithm με Boolean Indexing (Αντικαθιστά τα bitwise_and/or)
#     roi[mask > 0] = img1_slice[mask > 0]
#     img2[y1:y2, x1:x2] = roi
#
#     img2 = crop_image(img2)
#     return img2

def stitch_two_images(img1, img2, offset=(0, 0)):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    y_off, x_off = offset

    # Region Of Interest
    y1, y2 = max(0, y_off), min(h2, y_off + h1)
    x1, x2 = max(0, x_off), min(w2, x_off + w1)

    img1_y1 = max(0, -y_off)
    img1_y2 = img1_y1 + (y2 - y1)
    img1_x1 = max(0, -x_off)
    img1_x2 = img1_x1 + (x2 - x1)

    # 1. Create Views (Zero Memory Cost)
    dest_slice = img2[y1:y2, x1:x2]
    src_slice = img1[img1_y1:img1_y2, img1_x1:img1_x2]

    # 2. LOW-MEMORY MASK GENERATION
    gray_src = cv2.cvtColor(src_slice, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_src, 1, 255, cv2.THRESH_BINARY)

    # 3. In-Place Stitching
    dest_slice[mask == 255] = src_slice[mask == 255]

    # 4. Cleanup to free memory immediately
    del gray_src
    del mask

    # 5. Crop black borders (assuming crop_image handles the logic)
    return crop_image(img2)


def stitch_left(left_list, kp_alg: str = 'sift'):
    a = left_list[0]
    for b in left_list[1:]:
        H, pt1, pt2 = findHomographyFromImages(a, b, kp_alg=kp_alg)
        xh = np.linalg.inv(H)

        # Γωνίες 'α' μετά τον Μ//Σ
        h, w = a.shape[:2]
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
        transformed_corners = xh.dot(corners)
        transformed_corners = transformed_corners / transformed_corners[2]

        # Ελάχιστο/Μέγιστο για να βρω το μέγεθος πλαισίου
        min_x = min(0, transformed_corners[0].min())
        max_x = max(b.shape[1], transformed_corners[0].max())
        min_y = min(0, transformed_corners[1].min())
        max_y = max(b.shape[0], transformed_corners[1].max())

        # Σειρά ομογενών Μ/Σ
        translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
        xh = translation.dot(xh)

        # Τι μέγεθος απαιτείται; (να μην γίνει overflow με παραπάνω πιξελ)
        dsize = (int(np.ceil(max_x - min_x)), int(np.ceil(max_y - min_y)))

        # SANITY CHECK: Αν το μέγεθος είναι τεράστιο (Explosion), αγνόησε την εικόνα
        if dsize[0] * dsize[1] > 40000 * 40000:  # Όριο ~1.6 Gigapixels
            print(f"Skipping frame (Left): Homography explosion detected. Size {dsize}")
            continue

        # Νέο πλαίσιο εικόνας, κάνε warp την 'α'
        tmp = cv2.warpPerspective(a, xh, dsize)

        # Offset για 'β' εικόνα
        offset = (int(-min_y), int(-min_x))
        a = stitch_two_images(b, tmp, offset=offset)
        tmp = a
    return tmp


def stitch_right(leftimage, right_list, kp_alg: str = 'sift'):
    result = leftimage.copy()
    for each in right_list:
        H, pt1, pt2 = findHomographyFromImages(result, each, kp_alg=kp_alg)

        # Γωνίες μετά τον Μ/Σ
        h, w = each.shape[:2]
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T

        transformed_corners = H.dot(corners)
        transformed_corners = transformed_corners / transformed_corners[2]

        # Μέγεθος πλαισίου εικόνας
        min_x = min(0, transformed_corners[0].min())
        max_x = max(result.shape[1], transformed_corners[0].max())
        min_y = min(0, transformed_corners[1].min())
        max_y = max(result.shape[0], transformed_corners[1].max())

        # Μετατόπιση αν υπάρχει offset
        translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
        H_adjusted = translation.dot(H)

        # Ακριβές μέγεθος
        dsize = (int(np.ceil(max_x - min_x)), int(np.ceil(max_y - min_y)))

        # Warp δεξιά
        tmp = cv2.warpPerspective(each, H_adjusted, dsize)

        # Αρνητικό offset; Κανένα πρόβλημα, μετακίνησε την εικόνα
        if min_x < 0 or min_y < 0:
            # MEMORY FIX:
            # Instead of creating a new massive array (result_shifted), use 'tmp' as the canvas
            # and paste 'result' onto it. 'tmp' is already allocated and is the correct size.

            y_offset = int(-min_y)
            x_offset = int(-min_x)

            # Stitch result (overlay) onto tmp (canvas)
            # Effectively: tmp[offset] = result
            result = stitch_two_images(result, tmp, offset=(y_offset, x_offset))
        else:
            result = stitch_two_images(result, tmp)

    return result


def planar_stitch(images, kp_alg: str = 'sift'):
    mid = int((len(images) + 1) / 2)
    left_list = images[:mid]
    right_list = images[mid:]
    left = stitch_left(left_list, kp_alg=kp_alg)
    pano = stitch_right(left, right_list, kp_alg=kp_alg)
    return pano


# import cv2
# import numpy as np
# from src.answers.bfmatcher import findHomographyFromImages
# from src.answers.image_operations import crop_image
#
# # def stitch_two_images(img1, img2, offset=(0, 0)):
# #     print(f"THIS FAILS {img1.shape}")
# #     # cv2.namedWindow('1',cv2.WINDOW_KEEPRATIO)
# #     # cv2.imshow('1',img2)
# #     # cv2.waitKey(0)
# #     # cv2.destroyWindow('1')
# #     ret, mask = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY)
# #     tmp = img2[offset[0]:img1.shape[0] + offset[0], offset[1]:img1.shape[1] + offset[1]]
# #     mask = cv2.bitwise_not(mask)
# #     tmp = cv2.bitwise_and(tmp, mask)
# #     tmp = cv2.bitwise_or(img1, tmp)
# #     img2[offset[0]:img1.shape[0] + offset[0], offset[1]:img1.shape[1] + offset[1]] = tmp
# #     img2 = crop_image(img2)
# #     return img2
#
# def stitch_two_images(img1, img2, offset=(0, 0)):
#     y_off, x_off = offset
#     h1, w1 = img1.shape[:2]
#     h2, w2 = img2.shape[:2]
#
#     # Υπολογισμός ορίων επικάλυψης για να δουλέψουμε μόνο στο slice
#     y1 = max(y_off, 0)
#     x1 = max(x_off, 0)
#     y2 = min(y_off + h1, h2)
#     x2 = min(x_off + w1, w2)
#
#     if y1 >= y2 or x1 >= x2:
#         return img2
#
#     # Αντιστοίχιση συντεταγμένων στην εικόνα επικάλυψης
#     img1_y1 = y1 - y_off
#     img1_x1 = x1 - x_off
#     img1_y2 = img1_y1 + (y2 - y1)
#     img1_x2 = img1_x1 + (x2 - x1)
#
#     # Slice
#     img1_slice = img1[img1_y1:img1_y2, img1_x1:img1_x2]
#     roi = img2[y1:y2, x1:x2]
#     # Μάσκα
#     gray = cv2.cvtColor(img1_slice, cv2.COLOR_BGR2GRAY)
#     _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
#     # Painter's Algorithm
#     roi[mask > 0] = img1_slice[mask > 0]
#     img2[y1:y2, x1:x2] = roi
#
#     img2 = crop_image(img2)
#     return img2
#
# def stitch_left(left_list, kp_alg:str = 'sift'):
#     a = left_list[0]
#     for b in left_list[1:]:
#         H, pt1, pt2 = findHomographyFromImages(a, b, kp_alg=kp_alg)
#         xh = np.linalg.inv(H)
#
#         # Γωνίες 'α' μετά τον Μ//Σ
#         h, w = a.shape[:2]
#         corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
#         transformed_corners = xh.dot(corners)
#         transformed_corners = transformed_corners / transformed_corners[2]
#
#         # Ελάχιστο/Μέγιστο για να βρω το μέγεθος πλαισίου
#         min_x = min(0, transformed_corners[0].min())
#         max_x = max(b.shape[1], transformed_corners[0].max())
#         min_y = min(0, transformed_corners[1].min())
#         max_y = max(b.shape[0], transformed_corners[1].max())
#
#         # Σειρά ομογενών Μ/Σ
#         translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
#         xh = translation.dot(xh)
#
#         # Τι μέγεθος απαιτείται; (να μην γίνει overflow με παραπάνω πιξελ)
#         dsize = (int(np.ceil(max_x - min_x)), int(np.ceil(max_y - min_y)))
#
#         # Νέο πλαίσιο εικόνας, κάνε warp την 'α'
#         tmp = cv2.warpPerspective(a, xh, dsize)
#
#         # Offset για 'β' εικόνα
#         offset = (int(-min_y), int(-min_x))
#         a = stitch_two_images(b, tmp, offset=offset)
#         tmp = a
#     return tmp
#
#
#
# # def stitch_right(leftimage, right_list, kp_alg:str = 'sift'):
# #     result = leftimage.copy()
# #     for each in right_list:
# #         H, pt1, pt2 = findHomographyFromImages(result, each, kp_alg=kp_alg)
# #
# #         # Γωνίες μετά τον Μ/Σ
# #         h, w = each.shape[:2]
# #         corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]]).T
# #
# #         transformed_corners = H.dot(corners)
# #         transformed_corners = transformed_corners / transformed_corners[2]
# #
# #         # Μέγεθος πλαισίου εικόνας
# #         min_x = min(0, transformed_corners[0].min())
# #         max_x = max(result.shape[1], transformed_corners[0].max())
# #         min_y = min(0, transformed_corners[1].min())
# #         max_y = max(result.shape[0], transformed_corners[1].max())
# #
# #         # Μετατόπιση αν υπάρχει offset
# #         translation = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
# #         H_adjusted = translation.dot(H)
# #
# #         # Ακριβές μέγεθος
# #         dsize = (int(np.ceil(max_x - min_x)), int(np.ceil(max_y - min_y)))
# #
# #         # Warp δεξιά
# #         tmp = cv2.warpPerspective(each, H_adjusted, dsize)
# #
# #         # Αρνητικό offset; Κανένα πρόβλημα, μετακίνησε την εικόνα
# #         if min_x < 0 or min_y < 0:
# #             result_shifted = np.zeros((dsize[1], dsize[0], result.shape[2]), dtype=result.dtype)
# #             y_offset = int(-min_y)
# #             x_offset = int(-min_x)
# #             result_shifted[y_offset:y_offset + result.shape[0], x_offset:x_offset + result.shape[1]] = result
# #             result = stitch_two_images(result_shifted, tmp, offset=(0, 0))
# #         else:
# #             result = stitch_two_images(result, tmp)
# #
# #     return result
#
# def planar_stitch(images, kp_alg:str = 'sift'):
#     mid = int((len(images) + 1) / 2)
#     left_list = images[:mid]
#     right_list = images[mid:]
#     left = stitch_left(left_list, kp_alg=kp_alg)
#     pano = stitch_right(left, right_list, kp_alg=kp_alg)
#     return pano
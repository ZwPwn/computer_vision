import numpy as np
import cv2
import glob
from os import path,listdir

# 1. BOARD PARAMETERS
# Ensure these match the physical board you printed and measured.
SQUARES_X = 7
SQUARES_Y = 5
SQUARE_LENGTH = 0.030
MARKER_LENGTH = 0.015
DICT_TYPE = cv2.aruco.DICT_5X5_100  # The dictionary used

# 2. CONFIGURATION ---
IMAGE_FOLDER = '../../calibrate'    # Φάκελος με φωτογραφίες
CAMERA_CALIB_FILE = '../../camera_params.yaml'
CALIBRATION_FLAGS = 0

all_charuco_corners = []
all_charuco_ids = []
all_img_size = []

# 3. DICTIONARY AND BOARD OBJECTS
dictionary = cv2.aruco.getPredefinedDictionary(DICT_TYPE)
board = cv2.aruco.CharucoBoard_create(SQUARES_X, SQUARES_Y, SQUARE_LENGTH, MARKER_LENGTH, dictionary)
params = cv2.aruco.DetectorParameters_create()

# 4. DETECTION LOOP

print(f"Starting ChArUco board detection on images in '{IMAGE_FOLDER}/'...")

image_files = sorted([path.join(IMAGE_FOLDER,img) for img in listdir(IMAGE_FOLDER)])


for i, filename in enumerate(image_files):
    # Load the image
    img = cv2.imread(filename)
    if img is None:
        print(f"Warning: Could not read image {filename}. Skipping.")
        continue

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Detect the individual ArUco markers
    marker_corners, marker_ids, rejected_markers = cv2.aruco.detectMarkers(img_gray, dictionary, parameters=params)

    if marker_ids is not None:
        # 2. Interpolate the ChArUco corners
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, img_gray, board
        )

        if ret > 10:
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
            all_img_size.append(img_gray.shape[::-1])

            img_copy = img.copy()
            cv2.aruco.drawDetectedMarkers(img_copy, marker_corners, marker_ids)
            cv2.aruco.drawDetectedCornersCharuco(img_copy, charuco_corners, charuco_ids, (255, 0, 0))

            cv2.namedWindow('Detected Cornersx', cv2.WINDOW_KEEPRATIO)
            cv2.imshow('Detected Corners', img_copy)
            cv2.waitKey(10)  # Display for 10ms
            print(f"Image {i + 1}/{len(image_files)}: Found {ret} ChArUco corners.")
        else:
            print(
                f"Image {i + 1}/{len(image_files)}: Detected markers but not enough ChArUco corners for calibration. (Found {ret} corners)")
    else:
        print(f"Image {i + 1}/{len(image_files)}: No markers detected.")

cv2.destroyAllWindows()

print("\nStarting camera calibration...")
img_size = all_img_size[0]

# Perform the actual calibration using all collected data
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
    all_charuco_corners, all_charuco_ids, board, img_size, None, None, flags=CALIBRATION_FLAGS
)

# 6. Results
print(f"\n--- CALIBRATION RESULTS ---")
print(f"Reprojection Error: {ret:.4f} (Lower is better, typically < 1.0)")
print("\nCamera Matrix (Intrinsic Parameters):")
print(camera_matrix)
print("\nDistortion Coefficients:")
print(dist_coeffs)

# Optional: Save the calibration results
if ret > 0:
    cv_file = cv2.FileStorage(CAMERA_CALIB_FILE, cv2.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix", camera_matrix)
    cv_file.write("dist_coeffs", dist_coeffs)
    cv_file.release()
    print(f"\nCalibration data saved to '{CAMERA_CALIB_FILE}'")
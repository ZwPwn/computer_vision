import cv2
import numpy as np
from Src.helper_functions import show_window_ratio, show_window
from Src.integral import integral

DEBUG = False
# DEBUG = True  # Just comment this out for ""release""

# print(cv2.__version__)

def exercise(dut_array_image):
    original_integral = integral(dut_array_image)

    _, thresh = cv2.threshold(dut_array_image, 250, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    morph_close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    if DEBUG:
        show_window_ratio('morpholoy', morph_close.copy())

    text_mask = np.zeros_like(thresh)
    text_mask[thresh == 0] = 255

    text_integral = integral(text_mask)

    (totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(morph_close, 4)

    detected = cv2.cvtColor(dut_array_image.copy(),cv2.COLOR_GRAY2BGR)

    for i in range(1, totalLabels):
        sub_area = values[i, cv2.CC_STAT_AREA]
        x1 = values[i, cv2.CC_STAT_LEFT]
        y1 = values[i, cv2.CC_STAT_TOP]
        w1 = values[i, cv2.CC_STAT_WIDTH]
        h1 = values[i, cv2.CC_STAT_HEIGHT]
        bounding_box_area = w1 * h1

        pt1 = (x1, y1)
        pt2 = (x1 + w1, y1 + h1)

        sub_image = detected[y1:(y1 + h1), x1:(x1 + w1)]
        cv2.rectangle(detected, pt1, pt2, (0, 0, 255), 3)
        cv2.putText(detected, f"Region {i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        sub_integral = original_integral[y1 + h1, x1 + w1] - original_integral[y1 - 1, x1] - original_integral[
            y1, x1 - w1] + original_integral[y1 - 1, x1 - 1]
        mean_grey = sub_integral / sub_area
        sub_text_integral = text_integral[y1 + h1, x1 + w1] - text_integral[y1 - 1, x1] - text_integral[y1, x1 - 1] + \
                            text_integral[y1 - 1, x1 - 1]

        assert sub_area <= bounding_box_area

        sub_thresh = thresh[y1:(y1 + h1), x1:(x1 + w1)]
        # Κάνω morphological close για να ενώσω τα γράμματα σε λέξεις
        word_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        word_close = cv2.morphologyEx(sub_thresh, cv2.MORPH_CLOSE, word_kernel)

        # Μετράω τα connected components = λέξεις
        word_count, _, sub_values, _ = cv2.connectedComponentsWithStats(word_close, 4)
        # cv2.imwrite(f'./Report/text_segmentation/parts/{i}.png', word_close)

        print(f'-------Region {i} -------')
        print(f'Bounding box area(px): {bounding_box_area}')
        print(f'Area (px): {sub_area}')
        print(f'Words : {word_count}')
        print(f'Mean gray-level value in bounding box: {mean_grey}')
        # show_window(f'{i}.png', word_close)

    show_window_ratio('detected', detected.copy())
    # cv2.imwrite('./Report/text_segmentation/regions.png', detected)
    return (detected,morph_close)

if __name__ == "__main__":
    original_img = 'Images/5.png'
    original = cv2.imread(original_img, cv2.IMREAD_GRAYSCALE)
    detected, _ = exercise(original)
    cv2.imwrite('./Report/clean.png', detected)
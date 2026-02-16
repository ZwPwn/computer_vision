import cv2
import numpy as np
import matplotlib.pyplot as plt
from sympy.external.importtools import version_tuple

from Src.filters import smooth
from Src.helper_functions import show_window, show_window_ratio

DEBUG = True
IMAGE_PATH = '../Images/5.png'

def segment_document(binary_img, h_profile, threshold_ratio=0.05, min_width=50, min_height=20):
    img_height, img_width = binary_img.shape

    kernel_gaps = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
    threshold_h = np.max(h_profile) * threshold_ratio
    text_mask_h = h_profile > threshold_h
    morph_close = cv2.morphologyEx(threshold_h, cv2.MORPH_CLOSE, kernel_gaps)

    diff_h = np.diff(np.concatenate([[0], text_mask_h.astype(int), [0]]))
    starts_h = np.where(diff_h == 1)[0]
    ends_h = np.where(diff_h == -1)[0]

    regions = []

    for y_start, y_end in zip(starts_h, ends_h):
        row_strip = binary_img[y_start:y_end, :]
        v_profile = np.sum(row_strip, axis=0, dtype=np.uint64)

        threshold_v = np.max(v_profile) * threshold_ratio
        text_mask_v = v_profile > threshold_v

        diff_v = np.diff(np.concatenate([[0], text_mask_v.astype(int), [0]]))
        starts_v = np.where(diff_v == 1)[0]
        ends_v = np.where(diff_v == -1)[0]

        for x_start, x_end in zip(starts_v, ends_v):
            width = x_end - x_start
            height = y_end - y_start
            print(width)
            if width > min_width and height > min_height:
                regions.append((x_start, y_start, x_end, y_end))
            # regions.append((x_start, y_start, x_end, y_end))

    return regions

def find_text_regions(profile, threshold_ratio=0.05, min_gap=5):
    threshold = np.max(profile) * threshold_ratio

    text_mask = profile > threshold

    diff = np.diff(np.concatenate([[0], text_mask.astype(int), [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    regions = []
    if len(starts) > 0:
        current_start = starts[0]
        current_end = ends[0]
        for i in range(1, len(starts)):
            # Ό,τι είναι λιγότερο από το "stride" λάβε το υπόψιν
            if starts[i] - current_end < min_gap:
                current_end = ends[i]
            else:
                regions.append((current_start, current_end))
                current_start = starts[i]
                current_end = ends[i]
        regions.append((current_start, current_end))

    return regions


def segment_words_in_line(binary_img, y_start, y_end, threshold_ratio=0.02, min_word_width=5):
    line_strip = binary_img[y_start:y_end, :]
    # if DEBUG:
    #     show_window('line_strip', line_strip)
    #     show_window('binary_img', binary_img)
    vertical_profile = np.sum(line_strip, axis=0, dtype=np.uint64)
    words = find_text_regions(vertical_profile, threshold_ratio, min_gap=min_word_width)

    return words

def projection_profile_segmentation(img_path,
                                    line_threshold=0.05,
                                    word_threshold=0.02,
                                    smooth_radius=3,
                                    min_line_gap=10,
                                    min_word_gap=5):

    original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(original, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Horizontal Profile
    h_profile = np.sum(binary, axis=1, dtype=np.uint64)
    # h_smooth = smooth(h_profile, r=smooth_radius)
    h_smooth = h_profile

    # Boundaries
    lines = find_text_regions(h_smooth, threshold_ratio=line_threshold, min_gap=min_line_gap)
    # show_window('binary', binary)

    large_regions = segment_document(binary,h_smooth, threshold_ratio=0.1)
    regions_detect = cv2.cvtColor(original.copy(), cv2.COLOR_GRAY2BGR)
    for (x_start, y_start, x_end, y_end) in large_regions:
        assert x_start is not None
        assert y_start is not None
        assert x_end is not None
        assert y_end is not None
        cv2.rectangle(regions_detect, (x_start,y_start), (x_end,y_end), (0,0,255))
    show_window_ratio('Regions Detected', regions_detect)

    # Horizontal Projection
    all_words = {}
    for line_idx, (y_start, y_end) in enumerate(lines):
        words = segment_words_in_line(binary, y_start, y_end,
                                      threshold_ratio=word_threshold,
                                      min_word_width=min_word_gap)
        all_words[line_idx] = words

    return original, binary, lines, all_words, h_profile, h_smooth



def visualize_results(original, binary, lines, all_words, h_profile, h_smooth):
    result = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    for y_start, y_end in lines:
        cv2.rectangle(result, (0, y_start), (original.shape[1], y_end), (0, 0, 255), 2)
    for line_idx, words in all_words.items():
        y_start, y_end = lines[line_idx]
        for x_start, x_end in words:
            cv2.rectangle(result, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)


    fig = plt.figure(figsize=(20, 12))

    # Original Image
    ax1 = plt.subplot(3, 2, 1)
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    # Binary Image
    ax2 = plt.subplot(3, 2, 2)
    ax2.imshow(binary, cmap='gray')
    ax2.set_title('Binary Image (Inverted)', fontsize=14, fontweight='bold')
    ax2.axis('off')

    # Horizontal Profile
    ax3 = plt.subplot(3, 2, 3)
    y_positions = np.arange(len(h_profile))
    ax3.barh(y_positions, h_profile, color='blue', alpha=0.5, label='Raw Profile')
    ax3.plot(h_smooth, y_positions, 'r-', linewidth=2, label='Smoothed')

    # boundaries
    for y_start, y_end in lines:
        ax3.axhspan(y_start, y_end, alpha=0.2, color='green')

    ax3.invert_yaxis()
    ax3.set_title('Horizontal Projection Profile', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Sum of Pixel Intensities')
    ax3.set_ylabel('Y Position (Row)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Vertical Profile
    ax4 = plt.subplot(3, 2, 4)
    if len(lines) > 0:
        y_start, y_end = lines[0]
        line_strip = binary[y_start:y_end, :]
        v_profile = np.sum(line_strip, axis=0, dtype=np.uint64)
        x_positions = np.arange(len(v_profile))

        ax4.bar(x_positions, v_profile, color='blue', alpha=0.5, width=1.0)

        # Mark word boundaries
        if 0 in all_words:
            for x_start, x_end in all_words[0]:
                ax4.axvspan(x_start, x_end, alpha=0.2, color='green')

        ax4.set_title(f'Vertical Profile (Line 1)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('X Position (Column)')
        ax4.set_ylabel('Sum of Pixel Intensities')
        ax4.grid(True, alpha=0.3)

    # Segmentation Result (Lines only)
    ax5 = plt.subplot(3, 2, 5)
    result_lines = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    for y_start, y_end in lines:
        cv2.rectangle(result_lines, (0, y_start), (original.shape[1], y_end), (0, 0, 255), 2)
    ax5.imshow(cv2.cvtColor(result_lines, cv2.COLOR_BGR2RGB))
    ax5.set_title('Line Segmentation (Red)', fontsize=14, fontweight='bold')
    ax5.axis('off')

    # 6. Complete Segmentation (Lines + Words)
    ax6 = plt.subplot(3, 2, 6)
    ax6.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    ax6.set_title('Complete Segmentation (Lines: Red, Words: Green)', fontsize=14, fontweight='bold')
    ax6.axis('off')

    plt.tight_layout()
    plt.show()

    # Print statistics
    print("\n" + "=" * 60)
    print("SEGMENTATION STATISTICS")
    print("=" * 60)
    print(f"Total Lines Detected: {len(lines)}")
    print(f"Total Words Detected: {sum(len(words) for words in all_words.values())}")
    print("\nLine-by-Line Breakdown:")
    for line_idx, (y_start, y_end) in enumerate(lines):
        word_count = len(all_words.get(line_idx, []))
        print(f"  Line {line_idx + 1}: Y[{y_start:4d}:{y_end:4d}] - {word_count} words")
    print("=" * 60 + "\n")

#####################

if __name__ == "__main__":
    original, binary, lines, all_words, h_profile, h_smooth = projection_profile_segmentation(
        IMAGE_PATH,
        line_threshold=0.05,
        word_threshold=0.02,
        smooth_radius=3,
        min_line_gap=10,
        min_word_gap=5
    )

    # Visualize if DEBUG is True
    if DEBUG:
        visualize_results(original, binary, lines, all_words, h_profile, h_smooth)

    # Save result
    result = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    for y_start, y_end in lines:
        cv2.rectangle(result, (0, y_start), (original.shape[1], y_end), (0, 0, 255), 2)
    for line_idx, words in all_words.items():
        y_start, y_end = lines[line_idx]
        for x_start, x_end in words:
            cv2.rectangle(result, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    cv2.imwrite('../Report/projection/projection_profile_result.png', result)
    print("Result saved as 'projection_profile_result.png'")
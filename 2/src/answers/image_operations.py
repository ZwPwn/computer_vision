import cv2

def crop_image(img):
    h, w = img.shape[:2]
    scale=1.0
    # if h*w> 0.5*1000 * 1000:
    #     scale=0.1
    #     check_img = cv2.resize(img,(0,0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    # else:
    check_img = img
    tmp = cv2.cvtColor(check_img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(mask, 1, 2)
    assert contours is not None # Αν δεν βρεθεί έστω και μια ισουψής κάνε panic
    largest_contour = max(contours, key=cv2.contourArea)
    x,y,w_bounding,h_bounding = cv2.boundingRect(largest_contour)
    if scale != 1.0:
        x = int(x/scale)
        y = int(y/scale)
        w_bounding = int(w_bounding/scale)
        h_bounding = int(h_bounding/scale)
    x = max(0, x)
    y = max(0, y)
    w_box = min(w_bounding, w - x)
    h_box = min(h_bounding, h - y)

    ret = img[y:(y+h_box),x:(x+w_box)]
    # cv2.namedWindow('1',cv2.WINDOW_KEEPRATIO)
    # cv2.imshow('1', ret)
    # cv2.waitKey(0)
    # cv2.destroyWindow('1')
    return ret.copy()
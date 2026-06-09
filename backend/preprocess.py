import cv2

INPUT_SIZE = 224
BLUR_KERNEL = (5, 5)
USE_THRESHOLD = True


def preprocess_steps(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
    if USE_THRESHOLD:
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2)
    else:
        thresh = blurred
    resized = cv2.resize(thresh, (INPUT_SIZE, INPUT_SIZE))
    return {"gray": gray, "blurred": blurred, "thresholded": thresh, "resized": resized}


def preprocess_crop(bgr):
    return preprocess_steps(bgr)["resized"]

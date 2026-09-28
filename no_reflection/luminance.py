import cv2
import numpy as np


def rgb_to_luminance(image_bgr):
    # cast_to_float64
    image_bgr = image_bgr.astype(np.float64)
    # split_bgr_channels
    b, g, r = cv2.split(image_bgr)
    # weighted_luminance_sum
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance

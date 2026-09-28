import math

import cv2


def gaussian_blur(image, sigma):
    # kernel_size_from_sigma
    ksize = int(2 * math.ceil(3 * sigma) + 1)
    # apply_gaussian_blur
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

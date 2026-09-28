import numpy as np


def normalize_to_uint8(image):
    # cast_to_float64
    image = image.astype(np.float64)
    # compute_min_max
    min_val, max_val = image.min(), image.max()
    # guard_flat_image
    if max_val - min_val < 1e-12:
        return np.zeros_like(image, dtype=np.uint8)
    # scale_to_0_255
    normalized = (image - min_val) / (max_val - min_val) * 255.0
    return normalized.astype(np.uint8)

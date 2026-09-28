import numpy as np


def log_transform(luminance, c=None):
    # cast_to_float64
    r = luminance.astype(np.float64)
    # auto_scale_constant
    if c is None:
        c = 255.0 / np.log(1 + np.max(r))
    # log_compression
    s = c * np.log(1 + r)
    return s

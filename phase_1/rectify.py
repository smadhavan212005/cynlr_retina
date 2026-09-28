import numpy as np


def rectify_on(dog):
    # half_wave_rectify_positive
    return np.maximum(0, dog)


def rectify_off(dog):
    # half_wave_rectify_negative
    return np.maximum(0, -dog)

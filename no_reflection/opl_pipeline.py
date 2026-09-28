import cv2

from luminance import rgb_to_luminance
from log_transform import log_transform
from difference_of_gaussians import difference_of_gaussians
from rectify import rectify_on, rectify_off
from normalize import normalize_to_uint8


def opl_pipeline(image_bgr, sigma_c=1.5, sigma_s=6.0):
    # rgb_to_luminance
    luminance = rgb_to_luminance(image_bgr)
    # log_transform
    log_image = log_transform(luminance)
    # difference_of_gaussians
    dog = difference_of_gaussians(log_image, sigma_c, sigma_s)
    # rectify_on
    on_channel = rectify_on(dog)
    # rectify_off
    off_channel = rectify_off(dog)

    # greyscale_composite_base
    grey_u8 = normalize_to_uint8(luminance)
    composite = cv2.cvtColor(grey_u8, cv2.COLOR_GRAY2BGR).astype(float)

    # normalize_on_off_channels
    on_norm = normalize_to_uint8(on_channel).astype(float)
    off_norm = normalize_to_uint8(off_channel).astype(float)

    # blend_on_blue_off_red
    composite[:, :, 0] = (composite[:, :, 0] + on_norm).clip(0, 255)
    composite[:, :, 2] = (composite[:, :, 2] + off_norm).clip(0, 255)
    composite = composite.astype("uint8")

    return luminance, log_image, dog, on_channel, off_channel, composite

import os
import sys

import cv2
import numpy as np


# core_opl_primitives_same_as_phase_1

def rgb_to_luminance(image_bgr):
    # cast_to_float64
    image_bgr = image_bgr.astype(np.float64)
    # split_bgr_channels
    b = image_bgr[:, :, 0]
    g = image_bgr[:, :, 1]
    r = image_bgr[:, :, 2]
    # weighted_luminance_sum
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance


def log_transform(luminance, c=None):
    # cast_to_float64
    luminance = luminance.astype(np.float64)
    # auto_scale_constant
    if c is None:
        c = 255.0 / np.log1p(luminance.max())
    # log_compression
    return c * np.log1p(luminance)


def gaussian_blur(image, sigma):
    # kernel_size_from_sigma
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    # apply_gaussian_blur
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def difference_of_gaussians(image, sigma_c, sigma_s):
    # centre_gaussian
    center = gaussian_blur(image, sigma_c)
    # surround_gaussian
    surround = gaussian_blur(image, sigma_s)
    # centre_minus_surround
    return center - surround


def rectify_on(dog):
    return np.maximum(dog, 0)


def rectify_off(dog):
    return np.maximum(-dog, 0)


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


# stage_a_specular_suppression_before_dog

def build_specular_mask(image_bgr, s_thresh=0.15, v_thresh=0.75):
    # bgr_to_hsv
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float64)
    s = hsv[:, :, 1] / 255.0
    v = hsv[:, :, 2] / 255.0

    # low_saturation_high_value_mask
    spec_mask = (s < s_thresh) & (v > v_thresh)

    # dilate_mask_for_penumbra
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    spec_mask = cv2.dilate(spec_mask.astype(np.uint8), kernel, iterations=2)

    return spec_mask.astype(bool)


def suppress_specular(luminance_log, spec_mask, sigma_s):
    # surround_average
    surround = gaussian_blur(luminance_log, sigma_s)
    # replace_masked_pixels
    cleaned = luminance_log.copy()
    cleaned[spec_mask] = surround[spec_mask]
    return cleaned


# stage_b_contour_area_filter_after_dog

def filter_by_contour_area(dog, dog_threshold=5.0, min_area=500):
    # threshold_abs_dog
    edge_binary = (np.abs(dog) > dog_threshold).astype(np.uint8) * 255

    # find_external_contours
    contours, _ = cv2.findContours(edge_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # keep_contours_above_min_area
    filtered_mask = np.zeros_like(edge_binary)
    kept = 0
    for c in contours:
        if cv2.contourArea(c) > min_area:
            cv2.drawContours(filtered_mask, [c], -1, 255, thickness=cv2.FILLED)
            kept += 1

    print(f"Contours found: {len(contours)}, kept after area filter: {kept}")

    return filtered_mask


# full_v2_pipeline

def opl_pipeline_v2(image_bgr, sigma_c=1.5, sigma_s=6.0,
                     s_thresh=0.15, v_thresh=0.75,
                     dog_threshold=5.0, min_contour_area=500):
    # rgb_to_luminance
    luminance = rgb_to_luminance(image_bgr)
    # log_transform
    luminance_log = log_transform(luminance)

    # specular_mask_and_suppression
    spec_mask = build_specular_mask(image_bgr, s_thresh, v_thresh)
    luminance_clean = suppress_specular(luminance_log, spec_mask, sigma_s)

    # difference_of_gaussians_on_cleaned_luminance
    dog = difference_of_gaussians(luminance_clean, sigma_c, sigma_s)
    on_raw = rectify_on(dog)
    off_raw = rectify_off(dog)

    # contour_area_filter
    contour_mask = filter_by_contour_area(dog, dog_threshold, min_contour_area)
    mask_bool = contour_mask > 0
    on_filtered = on_raw * mask_bool
    off_filtered = off_raw * mask_bool

    # composite_grey_base_with_on_blue_off_red
    base = np.clip(luminance_clean, 0, 255).astype(np.uint8)
    composite = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR).astype(np.float64)
    composite[:, :, 0] += normalize_to_uint8(on_filtered).astype(np.float64)   # blue = ON
    composite[:, :, 2] += normalize_to_uint8(off_filtered).astype(np.float64)  # red = OFF
    composite = np.clip(composite, 0, 255).astype(np.uint8)

    return {
        "luminance_log": luminance_log,
        "spec_mask": spec_mask,
        "luminance_clean": luminance_clean,
        "dog": dog,
        "on_raw": on_raw,
        "off_raw": off_raw,
        "contour_mask": contour_mask,
        "on_filtered": on_filtered,
        "off_filtered": off_filtered,
        "composite": composite,
    }


# output_saving

def save_all(input_path, output_dir, **kwargs):
    # ensure_output_dir
    os.makedirs(output_dir, exist_ok=True)

    # read_input_image
    original = cv2.imread(input_path)
    if original is None:
        raise FileNotFoundError(f"Could not read input image: {input_path}")

    # run_v2_pipeline
    r = opl_pipeline_v2(original, **kwargs)

    # write_stage_images
    cv2.imwrite(os.path.join(output_dir, "01_input.png"), original)
    cv2.imwrite(os.path.join(output_dir, "02_luminance.png"), normalize_to_uint8(r["luminance_log"]))
    cv2.imwrite(os.path.join(output_dir, "03_specular_mask.png"), (r["spec_mask"].astype(np.uint8) * 255))
    cv2.imwrite(os.path.join(output_dir, "04_cleaned_luminance.png"), normalize_to_uint8(r["luminance_clean"]))
    cv2.imwrite(os.path.join(output_dir, "05_dog_raw.png"), normalize_to_uint8(np.abs(r["dog"])))
    cv2.imwrite(os.path.join(output_dir, "06_on_raw.png"), normalize_to_uint8(r["on_raw"]))
    cv2.imwrite(os.path.join(output_dir, "07_off_raw.png"), normalize_to_uint8(r["off_raw"]))
    cv2.imwrite(os.path.join(output_dir, "08_contour_mask.png"), r["contour_mask"])
    cv2.imwrite(os.path.join(output_dir, "09_on_filtered.png"), normalize_to_uint8(r["on_filtered"]))
    cv2.imwrite(os.path.join(output_dir, "10_off_filtered.png"), normalize_to_uint8(r["off_filtered"]))
    cv2.imwrite(os.path.join(output_dir, "11_composite.png"), r["composite"])

    # summarize_dog_stats
    dog = r["dog"]
    on_count = int(np.count_nonzero(r["on_filtered"]))
    off_count = int(np.count_nonzero(r["off_filtered"]))

    print(f"DoG min: {dog.min():.4f}")
    print(f"DoG max: {dog.max():.4f}")
    print(f"ON pixels (filtered): {on_count}")
    print(f"OFF pixels (filtered): {off_count}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python retinal_opl_v2.py <input_image_path>")
        sys.exit(1)

    # parse_input_path
    input_path = sys.argv[1]
    # resolve_output_dir
    output_dir = os.path.join(os.path.dirname(os.path.abspath(input_path)), "opl_v2_output")

    # run_and_save
    save_all(input_path, output_dir)
    print("Done.")

import os

import cv2
import numpy as np

from opl_pipeline import opl_pipeline
from normalize import normalize_to_uint8


def save_outputs(input_path, output_dir):
    # ensure_output_dir
    os.makedirs(output_dir, exist_ok=True)

    # read_input_image
    original = cv2.imread(input_path)
    if original is None:
        raise FileNotFoundError(f"Could not read input image: {input_path}")

    # run_opl_pipeline
    luminance, log_image, dog, on_channel, off_channel, composite = opl_pipeline(original)

    # write_stage_images
    cv2.imwrite(os.path.join(output_dir, "01_input.png"), original)
    cv2.imwrite(os.path.join(output_dir, "02_luminance.png"), normalize_to_uint8(luminance))
    cv2.imwrite(os.path.join(output_dir, "03_log_compressed.png"), normalize_to_uint8(log_image))
    cv2.imwrite(os.path.join(output_dir, "04_dog_response.png"), normalize_to_uint8(dog))
    cv2.imwrite(os.path.join(output_dir, "05_on_channel.png"), normalize_to_uint8(on_channel))
    cv2.imwrite(os.path.join(output_dir, "06_off_channel.png"), normalize_to_uint8(off_channel))
    cv2.imwrite(os.path.join(output_dir, "07_composite.png"), composite)

    # summarize_dog_stats
    dog_min, dog_max = dog.min(), dog.max()
    on_count = int(np.sum(dog > 0))
    off_count = int(np.sum(dog < 0))

    print(f"DoG min: {dog_min:.4f}")
    print(f"DoG max: {dog_max:.4f}")
    print(f"ON pixels firing (DoG > 0): {on_count}")
    print(f"OFF pixels firing (DoG < 0): {off_count}")

    if dog_min >= 0 or dog_max <= 0:
        print("WARNING: DoG response lacks both ON and OFF signal - check input/parameters.")

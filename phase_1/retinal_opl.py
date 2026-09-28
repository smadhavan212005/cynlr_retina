import os
import sys

from luminance import rgb_to_luminance
from log_transform import log_transform
from gaussian_blur import gaussian_blur
from difference_of_gaussians import difference_of_gaussians
from rectify import rectify_on, rectify_off
from opl_pipeline import opl_pipeline
from save_outputs import save_outputs

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python retinal_opl.py <input_image_path>")
        sys.exit(1)

    # parse_input_path
    input_path = sys.argv[1]
    # resolve_output_dir
    input_dir = os.path.dirname(os.path.abspath(input_path))
    output_dir = os.path.join(input_dir, "opl_output")

    # run_and_save
    save_outputs(input_path, output_dir)
    print("Done. Outputs saved to opl_output/")

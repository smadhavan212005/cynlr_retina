from gaussian_blur import gaussian_blur


def difference_of_gaussians(image, sigma_c, sigma_s):
    # centre_gaussian
    centre = gaussian_blur(image, sigma_c)
    # surround_gaussian
    surround = gaussian_blur(image, sigma_s)
    # centre_minus_surround
    dog = centre - surround
    return dog

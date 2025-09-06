import numpy as np
from scipy.ndimage import convolve


def gaussian_downsample(img, G):
    img_out = convolve(img, G, mode='constant')

    img_out = img_out[::2]
    img_out = img_out[:, ::2]

    return img_out

def build_gaussian_pyramid(img, course_size = 256):
    # Standard 5x5 Gaussian Kernel For Downsampling
    G_KERNEL = (1/256) * np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ])

    pyramid = [img]
    current_shape = img.shape
    current_img = img

    print("Initial Shape: ", current_shape)
    while (current_shape[0] >= course_size and current_shape[1] >= course_size):
        current_img = gaussian_downsample(current_img, G_KERNEL)
        current_shape = current_img.shape
        print("Current Shape: ", current_shape)
        pyramid.append(current_img)
    return pyramid







    
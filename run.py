# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio
import matplotlib.pyplot as plt
from pyramid import gaussian_downsample, build_gaussian_pyramid


def read_in_image(imname, plot=False):
    im = skio.imread(imname)
    if plot:
        plt.imshow(im)
        plt.show()
    
    # Convert to float for saving memory
    im = sk.img_as_float(im)

    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(int)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # Return in BGR format
    return np.array([b, g, r])

# Returns tuple of (aligned img, optim_dx, optim_dy)
def align(img, ref_img, dx_VALS, crop_factor = 0.1):
    min_L2 = np.linalg.norm(img - ref_img)
    optim_dx = None
    optim_dy = None

    optim_img = img

    ref_cropped = crop_center(ref_img, crop_factor)
    for dx in dx_VALS:
        for dy in dx_VALS:
            img_shifted  = np.roll(img, shift=dy, axis=0)
            img_shifted = np.roll(img_shifted, shift=dx, axis=1)

            img_cropped = crop_center(img_shifted, crop_factor)
            
            

            # Take the difference
            difference = img_cropped - ref_cropped

            # Compute metric
            L2 = np.linalg.norm(difference)

            if L2 < min_L2:
                min_L2 = L2
                optim_dx = dx
                optim_dy = dy
                optim_img = img_shifted

    return (optim_img, optim_dx, optim_dy)
    
def crop_center(img, crop_factor=0.1):
    border_x = int(crop_factor * img.shape[0])
    border_y = int(crop_factor * img.shape[1])
    return img[border_x : -border_x, border_y : -border_y]



def align_images(img_seperated, dx_VALS, metric="L2_Distance", crop_factor=0.1):
    # Green is our reference image
    ref_img = img_seperated[1]

    # Align blue to reference
    aligned_blue, blue_dx, blue_dy = align(img_seperated[0], ref_img, dx_VALS, crop_factor=crop_factor)

    # Align red to reference
    aligned_red, red_dx, red_dy = align(img_seperated[2], ref_img, dx_VALS, crop_factor=crop_factor)

    # Stack images
    im_out = np.dstack([aligned_red, ref_img, aligned_blue])

    # Logging info
    print("(x, y) Displacement to align Blue to Green: ", (blue_dx, blue_dy))
    print("(x, y) Displacement to align Red to Green: ", (red_dx, red_dy))

    return im_out




img_seperated = read_in_image(imname='data/monastery.jpg', plot=False)
plt.imshow(skio.imread('data/monastery.jpg'), cmap="grey")
plt.show()

max_displacement = 15
crop_factor = 0.1

optimal_color_image = align_images(img_seperated, dx_VALS=range(-max_displacement, max_displacement + 1), crop_factor=crop_factor)
im_out = crop_center(optimal_color_image, crop_factor=crop_factor)
plt.imshow(im_out)
plt.show()


# print(img_seperated.shape)

# # Create a figure with 1 row and 2 columns of axes
# fig, axes = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

# blue = img_seperated[0]
# blue_shifted = np.roll(blue, 100, axis=0)

# axes[0].imshow(blue, cmap="gray")
# axes[1].imshow(blue_shifted, cmap="gray")


# plt.show()



# align the images
# functions that might be useful for aligning the images include:
# np.roll, np.sum, sk.transform.rescale (for multiscale)

### ag = align(g, b)
### ar = align(r, b)
# create a color image
#im_out = np.dstack([ar, ag, b])

# save the image
#fname = '/out_path/out_fname.jpg'
#skio.imsave(fname, im_out)

# display the image
#skio.imshow(im_out)
#skio.show()
import numpy as np
import imageio
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage import color
import matplotlib.pyplot as plt


def imshow(img):
    plt.imshow(img, cmap=plt.cm.gray)
    plt.show()


def get_pixel_coord_from_images(image, background=0, binarize=False,
                                use_centroid=False, show_steps=False):
    """Get all pixel different from background.

    Parameters
    ----------
    image: np.array
        2d images

    background: int
        background pixel value

    binarize: bool
        binarize the image

    use_centroid: bool
        use only the centroid of a specific region

    show_steps:
        display intermediary images
 
    Returns
    -------
    pixels: list
        all pixels coordinate different from the background

    """
    if binarize or use_centroid:
        image = np.where(image!=background, 1, 0).astype(np.uint8)
        background = 0
        if show_steps:
            imshow(image)

    if use_centroid:
        im_dist = distance_transform_edt(image)
        if show_steps:
            imshow(im_dist)
        image = peak_local_max(im_dist, indices=False,
                               footprint=np.ones((3, 3)),
                               labels=image)
        if show_steps:
            imshow(image)

    pixels = np.argwhere(image != background)
    return pixels


if __name__ == "__main__":

    fname = r"C:\Users\skoudoro\Devel\scratches\clustering.png"
    dname = '/home/elef/Devel/robotikos/docs/examples/'
    fname = dname + 'pic_coords1.png'
    img = imageio.imread(fname)

        
    img = color.rgb2gray(img).astype(np.uint8)
    imshow(img)
    pixels = get_pixel_coord_from_images(img, background=1, use_centroid=False, show_steps=True)
    # pixels = get_pixel_coord_from_images(img, background=1, binarize=True)
    # pixels = get_pixel_coord_from_images(img)

    print(pixels)
    print(len(pixels))

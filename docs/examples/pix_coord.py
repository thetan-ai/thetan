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
        image = np.where(image != background, 1, 0).astype(np.uint8)
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


def generate_circles(n_samples=10, inner_radius=60, outer_radius=15,
                     density=150, size=(200, 200), labellize=False):
    """Generate small circle on a bigger one.

    Parameters
    ----------
    n_samples: int
        number of outer circle

    inner_radius: int
        radius of the bigger circle

    outer_radius: int
        radius of the small circle

    density: int
        number of point on each circle uniformly distributed

    size: tuple
        2d image size

    labellize: bool
        Switch on/off if you want to labellize each circle or not

    Returns
    -------
    arr: 2d array
        numpy array containing the drawing

    """
    arr = np.zeros(size)
    label_value = 1
    for theta in np.linspace(0, 2*np.pi, n_samples, endpoint=False):
        x = int(size[1]//2 + inner_radius * np.cos(theta))
        y = int(size[0]//2 + inner_radius * np.sin(theta))
        # arr[y, x] = 1
        if labellize:
            label_value += 1
        for _ in range(density):
            r = outer_radius * np.sqrt(np.random.random())
            theta = np.random.random() * 2 * np.pi
            x1 = int(x + r * np.cos(theta))
            y1 = int(y + r * np.sin(theta))
            arr[y1, x1] = label_value

    return arr


if __name__ == "__main__":
    TEST_GET_PIXEL_COORD = False
    TEST_GENERATE_CIRCLES = True

    fname = r"C:\Users\skoudoro\Devel\scratches\clustering.png"
    dname = '/home/elef/Devel/robotikos/docs/examples/'
    fname = dname + 'pic_coords1.png'
    img = imageio.imread(fname)

    # ------ Demo of get_pixel_coord_from_images -----------#
    if TEST_GET_PIXEL_COORD:
        img = color.rgb2gray(img).astype(np.uint8)
        imshow(img)
        pixels = get_pixel_coord_from_images(img, background=1,
                                             use_centroid=False,
                                             show_steps=True)
        # pixels = get_pixel_coord_from_images(img, background=1,
        #                                  binarize=True)
        # pixels = get_pixel_coord_from_images(img)

        print(pixels)
        print(len(pixels))

    # ------ Demo of create_cirles -----------#
    if TEST_GENERATE_CIRCLES:
        arr = generate_circles()
        plt.imshow(arr)
        plt.show()

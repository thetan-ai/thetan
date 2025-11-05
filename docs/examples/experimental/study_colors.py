import imageio.v2 as imageio
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from skimage.color import rgb2lab, rgb2luv
import os

fig_size = 10
font_size = 15


def read_and_resize_image(file_path, target_width=256):
    img = imageio.imread(file_path)[..., :3]
    target_height = target_width * img.shape[0] // img.shape[1]
    return cv2.resize(img, (target_width, target_height))


def plot_rgb_space(img, output_path=None, point_size=0.3, colorful=True):
    rgb = img.reshape(-1, 3)
    colors = rgb / 255 if colorful else 'k'
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(r, g, b, c=colors, s=point_size, alpha=0.5)
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    ax.set_title('RGB Color Space', fontsize=font_size)
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close(fig)


def plot_lab_space(img, output_path=None, point_size=0.3, colorful=True):
    lab = rgb2lab(img).reshape(-1, 3)
    colors = img.reshape(-1, 3) / 255 if colorful else 'k'
    l, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(l, a, b, c=colors, s=point_size, alpha=0.5)
    ax.set_xlabel('L*')
    ax.set_ylabel('a*')
    ax.set_zlabel('b*')
    ax.set_title('LAB Color Space', fontsize=font_size)
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close(fig)


def plot_luv_space(img, output_path=None, point_size=0.3, colorful=True):
    luv = rgb2luv(img).reshape(-1, 3)
    colors = img.reshape(-1, 3) / 255 if colorful else 'k'
    l, u, v = luv[:, 0], luv[:, 1], luv[:, 2]
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(l, u, v, c=colors, s=point_size, alpha=0.5)
    ax.set_xlabel('L*')
    ax.set_ylabel('u*')
    ax.set_zlabel('v*')
    ax.set_title('LUV Color Space', fontsize=font_size)
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close(fig)


def main(save_plots=False):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    images = [f for f in os.listdir('.') if f.lower().endswith(image_extensions)]

    output_dir = "output_plots"
    if save_plots and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_file in images:
        print(f"Processing: {image_file}")
        img = read_and_resize_image(image_file)

        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Resized Image: {image_file}')
        if save_plots:
            plt.savefig(os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_resized.png"))
        else:
            plt.show()
        plt.close()

        plot_rgb_space(
            img,
            output_path=os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_rgb.png") if save_plots else None,
            point_size=1,
            colorful=True,
        )
        plot_lab_space(
            img,
            output_path=os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_lab.png") if save_plots else None,
            point_size=1,
            colorful=True,
        )
        plot_luv_space(
            img,
            output_path=os.path.join(output_dir, f"{os.path.splitext(image_file)[0]}_luv.png") if save_plots else None,
            point_size=1,
            colorful=True,
        )


if __name__ == "__main__":
    main(save_plots=True)  # Set to True to save plots or False to display them

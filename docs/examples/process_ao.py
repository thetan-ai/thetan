import numpy as np
from patchify import patchify
from time import time
from thetan.berserker import berserker2, X_to_many_2D, berserker2_to_2D_labels, centroid_images
from thetan.theta import img2d_to_ijv_features
from scipy.ndimage import gaussian_gradient_magnitude, gaussian_filter, median_filter
import cv2

# Path to the video file
# video_path = r'/Users/prajwal/Desktop/P2Video/576_RW_R_11_18_2019_103_dewarped_ch4201_packbits.avi'
video_path = r'/home/elef/Downloads/576_RW_R_11_18_2019_103_dewarped_ch4201_packbits.avi'


cap = cv2.VideoCapture(video_path)

frames = []
while True:
    # Read frame from the video
    ret, frame = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        print("Finished reading the video. Exiting ...")
        break

    # Append frame to the list
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frames.append(gray_frame)

cap.release()

frames = np.asarray(frames)
frames = np.swapaxes(frames, 0, 2)
frames = np.swapaxes(frames, 0, 1)

print(frames.shape)



def prepare_2d_data(data):

    data = np.ascontiguousarray(data)

    ijv = img2d_to_ijv_features(data)

    ci, cj = np.asarray(data.shape) / 2.0

    X = ijv.copy()
    X[:, 0] -= ci
    X[:, 1] -= cj
    X[:, 2] *= 2

    return X


def patch_norm(data, patch):

    data2 = np.pad(data, patch[0]//2)
    data2_enh = patchify(data2, patch)
    data2_enhr = data2_enh.reshape((data.shape[0], data.shape[1], np.prod(patch)))
    data2_avg = data2_enhr.mean(axis=-1)
    return np.mean(data2_enhr - data2_avg[..., None], axis=-1)


def hist_local_eq(data, disk_sz = 30):

    from skimage import exposure
    from skimage.morphology import disk
    from skimage.morphology import ball
    from skimage.filters import rank
    from skimage.util import img_as_ubyte

    # Load an example image
    img = img_as_ubyte(data)

    # Global equalize
    img_rescale = exposure.equalize_hist(img)

    # Equalization
    footprint = disk(disk_sz)
    img_eq = rank.equalize(img, footprint=footprint)
    return img_eq

def process_all_frames(frames, theta_value, limit=None):

    from dipy.viz import horizon
    # horizon(images=[(frames, np.eye(4), 'frames')])

    frames_orig = frames.copy()


    # values = np.unique(frames)
    # frames = np.interp(frames, (values[1], frames.max()), (0, 255.))
    # frames2 = gaussian_filter(frames, sigma=1)
    # frames2 = median_filter(frames, size=3)

    frameon = patch_norm(frames_orig[..., 0], (15, 15))
    framelh = hist_local_eq(frames_orig[..., 0], 20)
    framelhg = gaussian_filter(framelh, sigma=1)

    from pylab import imshow, figure, show, title

    f = figure()
    title('original')
    imshow(frames_orig[..., 0])

    f2 = figure()
    title('frameon')
    imshow(frameon)

    f3 = figure()
    title('framelh')
    imshow(framelh)

    f4 = figure()
    title('framelhg')
    imshow(framelhg)


    show()

    t = 1

    # horizon(images=[(frames_orig, np.eye(4), 'frameso'), (frames, np.eye(4), 'frames'), (frames2, np.eye(4), 'frames2')])

    new_frames = np.zeros(frames.shape)

    if limit is None:
        ran = range(frames.shape[2])
    else:
        ran = range(limit)
    for i in ran:
        print('Processing frame ', i)
        data = frames[..., i].astype('f8')
        # values = np.unique(data)
        # data = np.interp(data, (values[1], data.max()), (0, 255))

        X = prepare_2d_data(data)
        # I, J, W = X_to_many_2D(X, data.shape)
        data_labels, centroids = berserker2_to_2D_labels(X, theta_value, data.shape)
        centroids_im = centroid_images(data_labels, centroids)
        new_frames[..., i] = centroids_im.copy()

        # grads = gaussian_gradient_magnitude(W, sigma=2)

    return new_frames

t = time()
new_frames = process_all_frames(frames, 40., None)
print('Duration is ', time() - t)

from dipy.viz import horizon
horizon(images=[(frames, np.eye(4), 'frames'), (new_frames, np.eye(4), 'new_frames'), (np.abs(frames - new_frames), np.eye(4), 'diff')], bg_color=(1, 1, 1))

avg_frame = np.mean(frames, axis=-1)


from pylab import imshow, figure, show, title

f = figure()
title('avg_frame')
imshow(avg_frame)
# f2 = figure()
# title('grads')
# imshow(grads)
# f3 = figure()
# title('centroids_im')
# imshow(centroids_im)
show()

t = 1
import numpy as np
from thetan.patchify import patchify
from time import time
from thetan.berserker import berserker2, X_to_many_2D, berserker2_to_2D_labels, centroid_images, prepare_data_2d, berserker2_seg_2d
from scipy.ndimage import gaussian_gradient_magnitude, gaussian_filter, median_filter
import cv2

# Path to the video file
# video_path = r'/Users/prajwal/Desktop/P2Video/576_RW_R_11_18_2019_103_dewarped_ch4201_packbits.avi'
# video_path = r'/home/elef/Downloads/576_RW_R_11_18_2019_103_dewarped_ch4201_packbits.avi'
#vvideo_path = r'C:/Users/Eleftherios/Downloads/576_RW_R_11_18_2019_103_dewarped_ch4201_packbits.avi'
video_path = r'C:/Users/elef/Downloads/576_RW_R_11_18_2019_103_dewarped_ch4201_packbits.avi'


def read_video(fname):
    cap = cv2.VideoCapture(fname)

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
    return frames


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


def process_as_timeseries(frames, theta_value, limit=None, enh=2.0, srt=False):

    sh = frames.shape
    X = frames.reshape((sh[0] * sh[1], sh[2])).astype('f8')
    if srt:
        X = np.sort(X, axis=-1)
    X = np.ascontiguousarray(X)


    print(X.shape)

    print('Starting Berserker 2 on timeseries')
    t = time()
    centroids, labels = berserker2(X, theta_value, all_out=False)
    print('Duration ', time() - t)

    labels_im = labels.reshape((sh[0], sh[1])).astype('f8')

    labels_unq = np.unique(labels_im)
    centroids_im = np.zeros(frames.shape)
    for i, l in enumerate(labels_unq):
        centroids_im[labels_im == l] = centroids[i]

    return X, centroids_im, labels_im



def process_all_frames(frames, theta_value, limit=None, enh=2.0, pg=False, ng=False):

    print(frames.shape)

    frames_orig = frames.copy()

    # values = np.unique(frames)
    # frames = np.interp(frames, (values[1], frames.max()), (0, 255.))
    # frames2 = gaussian_filter(frames, sigma=1)
    # frames2 = median_filter(frames, size=3)

    # frameon = patch_norm(frames_orig[..., 0], (15, 15))
    # framelh = hist_local_eq(frames_orig[..., 0], 20)
    # framelhg = gaussian_filter(framelh, sigma=1)

    t = 1

    # horizon(images=[(frames_orig, np.eye(4), 'frameso'), (frames, np.eye(4), 'frames'), (frames2, np.eye(4), 'frames2')])

    new_frames = np.zeros(frames.shape)
    data_labels_all = np.zeros(frames.shape)

    if limit is None:
        ran = range(frames.shape[2])
    else:
        ran = range(limit)
    for i in ran:
        print('Processing frame ', i)
        data = frames[..., i].astype('f8')
        # values = np.unique(data)
        # data = np.interp(data, (values[1], data.max()), (0, 255))

        # X = prepare_data_2d(data)
        # # I, J, W = X_to_many_2D(X, data.shape)
        # data_labels, centroids = berserker2_to_2D_labels(X, theta_value, data.shape)
        # centroids_im = centroid_images(data_labels, centroids)
        data = gaussian_filter(data, sigma=1)

        data_labels, centroids, centroids_im = berserker2_seg_2d(data, theta_value, enh=enh, pg=pg, ng=ng)

        new_frames[..., i] = centroids_im.copy()
        data_labels_all[..., i] = data_labels.copy()

        # grads = gaussian_gradient_magnitude(W, sigma=2)
        print()

    return new_frames, data_labels_all


frames = read_video(video_path)

timeseries = False
if timeseries:

    X, centroids_im, labels_im = process_as_timeseries(frames, 180, srt=False)

    # Calculate time derivative but adjust shape
    tmp = np.zeros(centroids_im.shape)
    centroids_im_dt = np.diff(centroids_im, axis=-1)
    tmp[..., 1:-1] = centroids_im_dt[..., 1:]

    from dipy.viz import horizon
    horizon(images=[(frames, np.eye(4), 'frames'), (centroids_im, np.eye(4), 'cim'), (tmp, np.eye(4), 'cim_dt')])

indyframes = True
if indyframes:

    t = time()
    new_frames, dl_all = process_all_frames(frames, 30., limit=None, enh=2., pg=True, ng=False)
    print('Duration is ', time() - t)

    from dipy.viz import horizon
    horizon(images=[(frames, np.eye(4), 'frames'),
                    (new_frames, np.eye(4), 'new_frames'),
                    (np.abs(frames - new_frames), np.eye(4), 'diff'),
                    (dl_all, np.eye(4), 'dl_all')],
                    bg_color=(1, 1, 1))

    avg_frame = np.mean(new_frames, axis=-1)


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
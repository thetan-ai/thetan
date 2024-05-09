import numpy as np
from time import time
from thetan.berserker import berserker2, X_to_many_2D, berserker2_to_2D_labels, centroid_images
from thetan.theta import img2d_to_ijv_features
from scipy.ndimage import gaussian_gradient_magnitude, gaussian_filter
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


def process_all_frames(frames, theta_value, limit=None):

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
        data = gaussian_filter(data, sigma=1)
        X = prepare_2d_data(data)
        # I, J, W = X_to_many_2D(X, data.shape)
        data_labels, centroids = berserker2_to_2D_labels(X, theta_value, data.shape)
        centroids_im = centroid_images(data_labels, centroids)
        new_frames[..., i] = centroids_im.copy()

        # grads = gaussian_gradient_magnitude(W, sigma=2)

    return new_frames

t = time()
new_frames = process_all_frames(frames, 40., 10)
print('Duration is ', time() - t)

from dipy.viz import horizon
horizon(images=[(frames, np.eye(4), 'frames'), (new_frames, np.eye(4), 'new_frames'), (np.abs(frames - new_frames), np.eye(4), 'diff')], bg_color=(1, 1, 1))

# from pylab import imshow, figure, show, title

# f = figure()
# title('W')
# imshow(W)
# f2 = figure()
# title('grads')
# imshow(grads)
# f3 = figure()
# title('centroids_im')
# imshow(centroids_im)
# show()

t = 1
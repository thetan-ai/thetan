import os
from time import time

import numpy as np

import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.filters import sobel
from skimage.segmentation import mark_boundaries

from thetan.berserker import berserker2, centroid_images
from thetan.theta import img2d_to_ijv_features

colors = ["#ffb300", "#803e75", "#ff6800", "#a6bdd7",
          "#c10020", "#cea262", "#817066", "#007d34",
          "#f6768e", "#00538a", "#ff7a5c", "#53377a",
          "#ff8e00", "#b32851", "#f4c800", "#7f180d",
          "#93aa00", "#593315", "#f13a13", "#232c16"]
times = []

folder = os.path.dirname(__file__)
img = cv.imread(os.path.join(folder, 'hand.jpg'), 0)

image = np.interp(img, (np.min(img), np.max(img)), (0, 1))

shape = image.shape

data = np.zeros((shape[0]*shape[1], 3))

cnt = 0
cluster_numbers = []
s_t = time()
data = img2d_to_ijv_features(image)
data[..., 0] = data[..., 0] / shape[0]
data[..., 1] = data[..., 1] / shape[1]

results = berserker2(data, 0.13, all_out=False)
labels = results[1]

label_img = np.zeros(shape).astype(np.int32)

cnt = 0
for i in range(shape[0]):
    for j in range(shape[1]):
        label_img[i, j] = labels[cnt]
        cnt += 1
times.append(time()-s_t)
cluster_numbers.append(len(np.unique(labels)))

# rgb_img = cv.imread('comparison_method/hand.jpg')
rgb_img = np.asarray(np.dstack((image, image, image)), dtype=np.uint8)

s_t = time()
segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
times.append(time()-s_t)

s_t = time()
segments_slic = slic(rgb_img, n_segments=250, compactness=10, sigma=1, start_label=1)
times.append(time()-s_t)
s_t = time()
segments_quick = quickshift(rgb_img, kernel_size=5, max_dist=13, ratio=0.5)
times.append(time()-s_t)
s_t = time()
gradient = sobel(img)
segments_watershed = watershed(gradient, markers=250, compactness=0.001)
times.append(time()-s_t)

print(f'Felzenszwalb number of segments: {len(np.unique(segments_fz))}')
cluster_numbers.append(len(np.unique(segments_fz)))
print(f'SLIC number of segments: {len(np.unique(segments_slic))}')
cluster_numbers.append(len(np.unique(segments_slic)))
print(f'Quickshift number of segments: {len(np.unique(segments_quick))}')
cluster_numbers.append(len(np.unique(segments_quick)))
print(f'Watershed number of segments: {len(np.unique(segments_watershed))}')
cluster_numbers.append(len(np.unique(segments_watershed)))

cluster_numbers = cluster_numbers[1:] + cluster_numbers[:1]

fig, ax = plt.subplots(2, 4, figsize=(10, 10), constrained_layout=True)

methods = ["Felzenszwalbs", 'SLIC', 'Quickshift', 'Compact watershed', 'Berserker']
times = times[1:] + times[:1]
ax[0, 0].imshow(mark_boundaries(img, segments_fz))
ax[0, 0].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
ax[0, 0].set_title("Felzenszwalbs's method")
ax[0, 1].imshow(mark_boundaries(img, segments_slic))
ax[0, 1].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
ax[0, 1].set_title('SLIC')
ax[0, 2].imshow(mark_boundaries(img, segments_quick))
ax[0, 2].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
ax[0, 2].set_title('Quickshift')
ax[0, 3].bar(methods, times, color=colors[:len(methods)])
ax[0, 3].legend(labels=methods, handles=[Patch(color=v, label=k) for k, v in zip(methods, colors[:len(methods)])], bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=8)
ax[0, 3].tick_params(labelbottom = False, bottom = False)

ax[1, 0].imshow(mark_boundaries(img, segments_watershed))
ax[1, 0].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
ax[1, 0].set_title('Compact watershed')
ax[1, 1].imshow(mark_boundaries(img, label_img))
ax[1, 1].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
ax[1, 1].set_title('TB (Ours)')
ax[1, 2].imshow(rgb_img)
ax[1, 2].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
ax[1, 2].set_title('Original image')
c_image = np.reshape(centroid_images(results[1], results[0]), shape)
ax[1, 3].imshow(c_image, cmap='gray')
ax[1, 3].tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
ax[1, 3].set_title('Centroid image')

ax[0, 3].set_xlabel('methods')
ax[0, 3].set_ylabel('time (sec)')
ax[0, 3].set_box_aspect(shape[0]/shape[1])

plt.show()
plt.close()
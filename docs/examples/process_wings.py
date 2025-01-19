import cv2
from thetan.theta import img2d_to_ijv_color


import numpy as np
import matplotlib.pyplot as plt

# fname = 'C:/Users/Eleftherios/Downloads/dorin.png'
fname = '/home/elef/Data/Wings/sample.jpg'
enh = 2.0
theta = 150

img = cv2.imread(fname)

rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Change here to LUV or LAB to make segmentation more realistic update thresholds accordingly

print(rgb_image.shape)

data = rgb_image

data = np.ascontiguousarray(data, dtype='f8')

ijluv = img2d_to_ijv_color(data)

ci, cj = np.asarray(data.shape[:2]) / 2.0

X = ijluv.copy()
X[:, 0] -= ci
X[:, 1] -= cj
X[:, 2:] *= enh

from thetan.berserker import berserker2_clean, centroid_images

centroids, labels = berserker2_clean(X, theta_value=theta)
print(centroids.shape)

unq = np.unique(labels)
centroids_image = np.zeros((data.shape[0] *  data.shape[1], data.shape[2]), dtype='f8')
for i, l in enumerate(unq):
    centroids_image[labels == l] = centroids[i, 2:] / enh

centroids_image = centroids_image.reshape(data.shape).astype(np.uint8)


from pylab import imshow, figure, show
figure()
imshow(rgb_image)
figure()
imshow(centroids_image)
show()
t = 1



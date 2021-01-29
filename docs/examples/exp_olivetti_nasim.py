import logging
from time import time
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_olivetti_faces
from sklearn.cluster import MiniBatchKMeans
from sklearn import decomposition

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
n_row, n_col = 2, 3
n_components = n_row * n_col
image_shape = (64, 64)
rng = RandomState(0)

# #############################################################################
# Load faces data
faces, target = fetch_olivetti_faces(return_X_y=True, shuffle=True,
                                     random_state=rng)

for f in range(faces.shape[0]):
    face0 = faces[f].reshape((64, 64))
    target0 = target[f]

    plt.figure()
    plt.title(target0)
    plt.imshow(face0)
    plt.show()

n_samples, n_features = faces.shape

# global centering
faces_centered = faces - faces.mean(axis=0)

# local centering
faces_centered -= faces_centered.mean(axis=1).reshape(n_samples, -1)

print("Dataset consists of %d faces" % n_samples)

test = 1

from theta_machines_tests import theta_g1, theta_g2, theta_g2x

centroids, indices = theta_g1(faces, theta=1000)
# kmeans
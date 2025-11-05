import cv2
import numpy as np
import matplotlib.pyplot as plt
from thetan.theta import img2d_to_ijv_features


# fname = '/home/elef/Data/Dorin/t1_up.png'
# fname = '/home/elef/Data/Dorin/t1_mid.png'
fname = '/home/elef/Data/Dorin/t1_down.png'

img = cv2.imread(fname, 0)
print(img.shape)

img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)
print(img.shape)

X = img2d_to_ijv_features(img.astype('f8'))

X[:, 2]  *= 4

x = X[:, 0]
y = X[:, 1]
v = X[:, 2]

sper = 40
ind = np.random.choice(np.arange(X.shape[0]), int(np.round((sper / 100.) * X.shape[0], 0)), replace=False)

x = x[ind]
y = y[ind]
v = v[ind]

print(X.shape)




from sklearn.cluster import MeanShift

from thetan.berserker import berserker2_clean, tbscan, tb_randomized

X = np.ascontiguousarray(X)
theta = 20.0
niceK = 200

tb_centroids, tb_labels = berserker2_clean(X, theta)

uniq, sizes = np.unique(tb_labels, return_counts=True)

print(np.sort(sizes)[::-1])

biggest_k = np.argsort(sizes)[-niceK:]

hd_centroids = np.ascontiguousarray(tb_centroids[biggest_k].copy())

print(tb_centroids.shape)

# tmp = 1

# Create a 3D scatter plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, v, c='b', marker='.', alpha=0.05)
# ax.scatter(tb_centroids[:, 0], tb_centroids[:, 1], tb_centroids[:, 2], c='r', marker='o', alpha=1)
ax.scatter(hd_centroids[:, 0], hd_centroids[:, 1], hd_centroids[:, 2], c='r', marker='o', alpha=1)

# Set axis labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('v')
ax.set_title('3D Scatter Plot of X')

plt.show()
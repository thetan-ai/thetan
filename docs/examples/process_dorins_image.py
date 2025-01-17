import cv2
import numpy as np
import matplotlib.pyplot as plt

# fname = 'C:/Users/Eleftherios/Downloads/dorin.png'
fname = '/home/elef/Data/Dorin/dorin.png'
img = cv2.imread(fname)

print(img.shape)

img = cv2.resize(img, (0,0), fx=0.3741, fy=0.37246)

# cv2.imshow('LUV Image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

print(img.shape)

luv_image = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)

print(luv_image.shape)

# # Display the LUV image
# cv2.imshow('LUV Image', luv_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

L = luv_image[..., 0].ravel()
U = luv_image[..., 1].ravel()
V = luv_image[..., 2].ravel() * 0

sper = 20
ind = np.random.choice(np.arange(L.shape[0]), int(np.round((sper / 100.) * L.shape[0], 0)), replace=False)

L = L[ind]
U = U[ind]
V = V[ind]

print(np.product(L.shape))


from sklearn.cluster import MeanShift

from thetan.berserker import berserker2_clean, tbscan, tb_randomized

X = np.zeros((L.shape[0], 3))

X[:, 0] = L
X[:, 1] = U
X[:, 2] = V

X = np.ascontiguousarray(X)
theta = 12.0
niceK = 50

tb_centroids, tb_labels = berserker2_clean(X, theta)

uniq, sizes = np.unique(tb_labels, return_counts=True)

print(np.sort(sizes)[::-1])

biggest_k = np.argsort(sizes)[-niceK:]

hd_centroids = np.ascontiguousarray(tb_centroids[biggest_k].copy())

print(tb_centroids.shape)

tmp = 1

# Create a 3D scatter plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(L, U, V, c='b', marker='.', alpha=0.1)
# ax.scatter(tb_centroids[:, 0], tb_centroids[:, 1], tb_centroids[:, 2], c='r', marker='o', alpha=1)
ax.scatter(hd_centroids[:, 0], hd_centroids[:, 1], hd_centroids[:, 2], c='r', marker='o', alpha=1)

# Set axis labels and title
ax.set_xlabel('L')
ax.set_ylabel('U')
ax.set_zlabel('V')
ax.set_title('3D Scatter Plot of LUV')

plt.show()
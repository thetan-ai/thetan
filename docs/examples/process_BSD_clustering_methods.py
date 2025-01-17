import os
import sys
import argparse

from time import time

import numpy as np
from scipy.io import loadmat

import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.filters import sobel
from skimage.segmentation import mark_boundaries, find_boundaries

from thetan.berserker import berserker2, centroid_images
from thetan.theta import img2d_to_ijv_features, img2d_to_ijv_color

from sklearn.cluster import KMeans, MeanShift
from MeanShiftPP import MeanShiftPP

from fury.colormap import distinguishable_colormap

colors = ["#ffb300", "#803e75", "#ff6800", "#a6bdd7",
          "#c10020", "#cea262", "#817066", "#007d34",
          "#f6768e", "#00538a", "#ff7a5c", "#53377a",
          "#ff8e00", "#b32851", "#f4c800", "#7f180d",
          "#93aa00", "#593315", "#f13a13", "#232c16"]
 

img_path = '/nobackup/jp109/thetan_experiments/BSD/images/all/100007.jpg'
img = cv.imread(img_path)
lab_img = cv.cvtColor(img, cv.COLOR_RGB2LAB)
norm_img = lab_img.astype(float)
norm_img[..., 0] = norm_img[..., 0] / 100
norm_img[..., 1:] = norm_img[..., 1:] / 255
ori_img = img
data = img2d_to_ijv_color(norm_img)
shape = img.shape[:2]

data[..., 0] = data[..., 0] / shape[0]
data[..., 1] = data[..., 1] / shape[1]
data[..., 2:] = data[..., 2:] * 0.6

_, mspp_labels = MeanShiftPP(bandwidth=0.045).fit_predict(data, True)
mspp_labels = np.reshape(mspp_labels, shape)
print('Meanshift++', np.max(mspp_labels)+1)
ms_labels = MeanShift(bandwidth=0.078, bin_seeding=True).fit_predict(data)
ms_labels = np.reshape(ms_labels, shape)
print('Meanshift', np.max(ms_labels)+1)
km_labels = KMeans(100).fit_predict(data)
km_labels = np.reshape(km_labels, shape)

colormap = distinguishable_colormap(nb_colors=np.max([mspp_labels, ms_labels, km_labels])+1)

color_label = np.zeros((shape[0], shape[1], 3))
for i in range(shape[0]):
    for j in range(shape[1]):
        color_label[i, j] = colormap[mspp_labels[i, j]]
mean_img = np.zeros((shape[0], shape[1], 3))
mean_vs = [np.mean(img[mspp_labels==l], axis=0) for l in range(np.max(np.unique(mspp_labels))+1)]
for i in range(shape[0]):
    for j in range(shape[1]):
        mean_img[i, j] = mean_vs[mspp_labels[i, j]]
mean_img = np.clip(np.round(mean_img), 0, 255).astype(np.uint8)
plt.figure()
plt.imshow(color_label)
plt.axis('off')
plt.savefig('bears/Meanshitpp.jpg', bbox_inches='tight', pad_inches = 0)
plt.close()

plt.figure()
plt.imshow(color_label)
plt.axis('off')
plt.savefig('bears/Meanshitpp.svg', bbox_inches='tight', pad_inches = 0)
plt.close()

plt.figure()
plt.imshow(mean_img)
plt.axis('off')
plt.savefig('bears/Meanshitpp_avg.svg', bbox_inches='tight', pad_inches = 0)
plt.close()
        
plt.figure()
plt.imshow(mean_img)
plt.axis('off')
plt.savefig('bears/Meanshitpp_avg.jpg', bbox_inches='tight', pad_inches = 0)
plt.close()



color_label = np.zeros((shape[0], shape[1], 3))
for i in range(shape[0]):
    for j in range(shape[1]):
        color_label[i, j] = colormap[ms_labels[i, j]]
mean_img = np.zeros((shape[0], shape[1], 3))
mean_vs = [np.mean(img[ms_labels==l], axis=0) for l in range(np.max(np.unique(ms_labels))+1)]
for i in range(shape[0]):
    for j in range(shape[1]):
        mean_img[i, j] = mean_vs[ms_labels[i, j]]
mean_img = np.clip(np.round(mean_img), 0, 255).astype(np.uint8)
plt.figure()
plt.imshow(color_label)
plt.axis('off')
plt.savefig('bears/Meanshit.jpg', bbox_inches='tight', pad_inches = 0)
plt.close()

plt.figure()
plt.imshow(color_label)
plt.axis('off')
plt.savefig('bears/Meanshit.svg', bbox_inches='tight', pad_inches = 0)
plt.close()

plt.figure()
plt.imshow(mean_img)
plt.axis('off')
plt.savefig('bears/Meanshit_avg.svg', bbox_inches='tight', pad_inches = 0)
plt.close()
        
plt.figure()
plt.imshow(mean_img)
plt.axis('off')
plt.savefig('bears/Meanshit_avg.jpg', bbox_inches='tight', pad_inches = 0)
plt.close()

color_label = np.zeros((shape[0], shape[1], 3))
for i in range(shape[0]):
    for j in range(shape[1]):
        color_label[i, j] = colormap[km_labels[i, j]]
mean_img = np.zeros((shape[0], shape[1], 3))
mean_vs = [np.mean(img[km_labels==l], axis=0) for l in range(np.max(np.unique(km_labels))+1)]
for i in range(shape[0]):
    for j in range(shape[1]):
        mean_img[i, j] = mean_vs[km_labels[i, j]]
mean_img = np.clip(np.round(mean_img), 0, 255).astype(np.uint8)
plt.figure()
plt.imshow(color_label)
plt.axis('off')
plt.savefig('bears/KMeanspp.jpg', bbox_inches='tight', pad_inches = 0)
plt.close()

plt.figure()
plt.imshow(color_label)
plt.axis('off')
plt.savefig('bears/KMeanspp.svg', bbox_inches='tight', pad_inches = 0)
plt.close()

plt.figure()
plt.imshow(mean_img)
plt.axis('off')
plt.savefig('bears/KMeanspp_avg.svg', bbox_inches='tight', pad_inches = 0)
plt.close()
        
plt.figure()
plt.imshow(mean_img)
plt.axis('off')
plt.savefig('bears/KMeanspp_avg.jpg', bbox_inches='tight', pad_inches = 0)
plt.close()

#====================================
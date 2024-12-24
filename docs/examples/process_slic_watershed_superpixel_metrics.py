import os
import sys
import argparse

from time import time
import json

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

colors = ["#ffb300", "#803e75", "#ff6800", "#a6bdd7",
          "#c10020", "#cea262", "#817066", "#007d34",
          "#f6768e", "#00538a", "#ff7a5c", "#53377a",
          "#ff8e00", "#b32851", "#f4c800", "#7f180d",
          "#93aa00", "#593315", "#f13a13", "#232c16"]

# metrics from here
# https://arxiv.org/pdf/1612.01601

def boundary_recall(gt_bound, label_bound):
    FN = len(np.argwhere(np.all([gt_bound != False, label_bound == False], axis=0)))
    TP = len(np.argwhere(np.all([gt_bound == True, label_bound == True], axis=0)))
    return TP / (TP + FN)

def underseg_error(gt_label, sp_label, overlap_graph=None):
    gt_u, gt_c = np.unique(gt_label, return_counts=True)
    sp_u, sp_c = np.unique(sp_label, return_counts=True)
    shape = gt_label.shape
    if overlap_graph is None:
        overlap_graph = np.zeros((len(gt_u), len(sp_u)))
        for i in range(shape[0]):
            for j in range(shape[1]):
                overlap_graph[gt_label[i, j], sp_label[i, j]] += 1
    
    return np.sum([(np.sum([sp_c[j] for j in np.argwhere(overlap_graph[i] != 0)]) - gt_c[i]) / gt_c[i] for i in range(len(gt_u))]) / (shape[0] * shape[1]), overlap_graph

def underseg_bergh_error(gt_label, sp_label, overlap_graph=None):
    gt_u, gt_c = np.unique(gt_label, return_counts=True)
    sp_u, sp_c = np.unique(sp_label, return_counts=True)
    shape = gt_label.shape
    if overlap_graph is None:
        overlap_graph = np.zeros((len(gt_u), len(sp_u)))
        for i in range(shape[0]):
            for j in range(shape[1]):
                overlap_graph[gt_label[i, j], sp_label[i, j]] += 1
    
    return np.sum([sp_c[j] - np.max(overlap_graph[:, j]) for j in range(len(sp_u))]) / (shape[0] * shape[1]), overlap_graph

def underseg_np_error(gt_label, sp_label, overlap_graph=None):
    gt_u, gt_c = np.unique(gt_label, return_counts=True)
    sp_u, sp_c = np.unique(sp_label, return_counts=True)
    shape = gt_label.shape
    if overlap_graph is None:
        overlap_graph = np.zeros((len(gt_u), len(sp_u)))
        for i in range(shape[0]):
            for j in range(shape[1]):
                overlap_graph[gt_label[i, j], sp_label[i, j]] += 1
    
    return np.sum([np.sum([np.min([overlap_graph[i, j], sp_c[j] - overlap_graph[i, j]]) for j in np.argwhere(overlap_graph[i] != 0)]) for i in range(len(gt_u))]) / (shape[0] * shape[1]), overlap_graph

# This is higher the better
def explained_variation(img, sp_label):
    sp_u, sp_c = np.unique(sp_label, return_counts=True)
    avg_values = [np.mean(img[sp_label==i]) for i in sp_u]
    mean_v = np.mean(img)
    return np.sum([sp_c[j] * np.square(avg_values[j] - mean_v) for j in range(len(sp_u))]) / np.sum(np.square(img-mean_v))

# For this all pixels of the superpixel has to be connected
# I have no idea how area and size of superpixel can be different, so I squared it
# Open to ideas
def compactness_score(sp_label):
    shape = sp_label.shape
    sp_u, sp_c = np.unique(sp_label, return_counts=True)

    return np.sum([sp_c[j]**2 * 4 * np.pi / np.square(np.sum(find_boundaries(sp_label, connectivity=3))) for j in range(len(sp_u))]) / (shape[0] * shape[1])

def get_metrics(img, sp_label, sp_bound, gt=False, gt_bound=None):
    return_dict = {}
    if gt:
        return_dict['REC'] = boundary_recall(gt_bound, sp_bound)
        return_dict['US'], overlap_graph = underseg_error(gt_bound, sp_bound)
        return_dict['USB'], _ = underseg_bergh_error(gt_bound, sp_bound, overlap_graph)
        return_dict['USN'], _ = underseg_np_error(gt_bound, sp_bound, overlap_graph)
    return_dict['EV'] = explained_variation(img, sp_label)
    return_dict['CO'] = compactness_score(sp_label)

    return return_dict

def load_gt(gt_path):
    gt_path = gt_path[:-4] + '.mat'
    if gt_path.endswith('.mat'):
        data = loadmat(gt_path)
        data_len = data['groundTruth'].shape[1]
        return [data['groundTruth'][0][i][0][0][0] for i in range(data_len)]


# colors = ["#ffb300", "#803e75", "#ff6800", "#a6bdd7",
#           "#c10020", "#cea262", "#817066", "#007d34",
#           "#f6768e", "#00538a", "#ff7a5c", "#53377a",
#           "#ff8e00", "#b32851", "#f4c800", "#7f180d",
#           "#93aa00", "#593315", "#f13a13", "#232c16"]
# times = []



parser = argparse.ArgumentParser(description="Evaluate 2D images")

parser.add_argument('dir_path')
parser.add_argument('output_path')
parser.add_argument('-c', '--color', action='store_true', help='whether if color or not')
parser.add_argument('-g', '--gt_path', type=str, help='ground truth path')

args = parser.parse_args()

if args.color:
    print("Assumes color image...")

if args.gt_path:
    print("Ground truth path:", args.gt_path)
 
dir_path = args.dir_path
output_path = args.output_path
gt_path = args.gt_path
rgb = args.color


#SLICO
methods = ['SLIC_m5_25', 'SLIC_m10_25', 'SLIC_m20_25', 'SLIC_m30_25',
           'SLIC_m5_100', 'SLIC_m10_100', 'SLIC_m20_100', 'SLIC_m30_100',
           'SLIC_m5_250', 'SLIC_m10_250', 'SLIC_m20_250', 'SLIC_m30_250',
           'SLIC_m5_500', 'SLIC_m10_500', 'SLIC_m20_500', 'SLIC_m30_500',
           'SLIC_m5_1000', 'SLIC_m10_1000', 'SLIC_m20_1000', 'SLIC_m30_1000',
           'SLIC_m5_2500', 'SLIC_m10_2500', 'SLIC_m20_2500', 'SLIC_m30_2500',
           'Compact_watershed_m5_25', 'Compact_watershed_m10_25', 'Compact_watershed_m20_25', 'Compact_watershed_m30_25',
           'Compact_watershed_m5_100', 'Compact_watershed_m10_100', 'Compact_watershed_m20_100', 'Compact_watershed_m30_100',
           'Compact_watershed_m5_250', 'Compact_watershed_m10_250', 'Compact_watershed_m20_250', 'Compact_watershed_m30_250',
           'Compact_watershed_m5_500', 'Compact_watershed_m10_500', 'Compact_watershed_m20_500', 'Compact_watershed_m30_500',
           'Compact_watershed_m5_1000', 'Compact_watershed_m10_1000', 'Compact_watershed_m20_1000', 'Compact_watershed_m30_1000',
           'Compact_watershed_m5_2500', 'Compact_watershed_m10_2500', 'Compact_watershed_m20_2500', 'Compact_watershed_m30_2500']

metric_avg = {}
metrics = {}

for m in methods:
    metric_avg[m] = {}
    metrics[m] = []

if os.path.exists(output_path) == False:
    os.makedirs(output_path)

if rgb:
    slic_channel_axis=-1
else:
    slic_channel_axis=None

with open(os.path.join(output_path, 'output.txt'), 'w+') as file:
    for img_path in os.listdir(dir_path):
        if img_path.startswith('Thumb'):
            continue
        if os.path.exists(os.path.join(output_path, img_path[:-4])+'.npz'):
            with open(os.path.join(output_path, img_path[:-4]+'_metrics_eval.json'), "r") as file:
                metrics = json.load(file)
        else:
            print(os.path.join(output_path, img_path[:-4])+'.npz')
            if rgb:
                try:
                    img = np.load(os.path.join(dir_path, img_path))['arr_0']
                except:
                    img = cv.imread(os.path.join(dir_path, img_path))
                gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
                lab_img = cv.cvtColor(img, cv.COLOR_RGB2LAB)
                s_t = time()
                # norm_img = lab_img / 255
                norm_img = img / 255
                ori_img = img
            else:
                try:
                    img = np.load(os.path.join(dir_path, img_path))['arr_0']
                except:
                    img = cv.imread(os.path.join(dir_path, img_path), 0)
                gray_img = img
                rgb_img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
                rgb_img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
            shape = img.shape[:2]

            if gt_path:
                # Some line to load gt
                gts = load_gt(os.path.join(gt_path, img_path))
                gt_bounds = []
                for gt in gts:
                    gt_bounds.append(find_boundaries(gt))

            segment_dict = {}
            for method in methods:
                if method.startswith('SLIC'):
                    n_segments = int(method.split('_')[-1])
                    compactness = float(method.split('_')[-2][1:])
                    s_t = time()
                    if rgb:
                        segments = slic(lab_img, n_segments=n_segments, compactness=compactness, sigma=1, start_label=1, channel_axis=slic_channel_axis)
                    else:
                        segments = slic(img, n_segments=n_segments, compactness=compactness, sigma=1, start_label=1, channel_axis=slic_channel_axis)

                elif method.startswith('Compact_watershed'):
                    n_segments = int(method.split('_')[-1])
                    compactness = float(method.split('_')[-2][1:])
                    s_t = time()
                    gradient = sobel(gray_img)
                    segments = watershed(gradient, markers=n_segments, compactness=compactness)

                exec_time = time()-s_t
                cluster_number = len(np.unique(segments))

                bounds = find_boundaries(segments)

                if gt_path:
                    temp_sum = None
                    for gt_idx in range(len(gts)):
                        temp = get_metrics(img, segments, bounds, gt=True, gt_bound=gt_bounds[gt_idx])
                        if temp_sum is None:
                            temp_sum = temp
                        else:
                            for key in temp_sum.keys():
                                temp_sum[key] = temp_sum[key] + temp[key]
                    for key in temp_sum.keys():
                        temp_sum[key] = temp_sum[key] / len(gts)
                    temp = temp_sum

                else:
                    temp = get_metrics(img, segments, bounds)
                temp['time'] = exec_time
                temp['cluster_n'] = cluster_number
                metrics[method].append(temp)
                segment_dict[method] = segments

            np.savez(os.path.join(output_path, img_path[:-4]), img=ori_img, label=gt, **segment_dict)
            
            
            with open(os.path.join(output_path, img_path[:-4]+'_metrics_eval.json'), "w") as fp:
                json.dump(metrics, fp)
    
    for m in methods:
        print(m + ': average and std', file=file)
        for key in metrics[m][0].keys():
            print(key, np.mean([metrics[m][i][key] for i in range(len(metrics[m]))]), file=file)
            print(key, np.std([metrics[m][i][key] for i in range(len(metrics[m]))]), file=file)
    
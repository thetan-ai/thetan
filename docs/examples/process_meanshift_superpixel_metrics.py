import os
import sys
import argparse

from time import time

import numpy as np

import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.filters import sobel
from skimage.segmentation import mark_boundaries, find_boundaries

from sklearn.cluster import MeanShift

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
def compactness(sp_label):
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
    return_dict['CO'] = compactness(sp_label)

    return return_dict

def load_gt(gt_path):
    return np.load(gt_path)['arr_0']


parser = argparse.ArgumentParser(description="Evaluate 2D images")

parser.add_argument('dir_path')
parser.add_argument('output_path')
parser.add_argument('-b', '--meanshiftpp', type=float, help='bandwidth value')
parser.add_argument('-g', '--gt_path', type=str, help='ground truth path')
parser.add_argument('-c', '--color', action='store_true', help='whether if color or not')
parser.add_argument('-p', '--meanshiftpp', action='store_false', help='whether we use meanshift++')

args = parser.parse_args()

if args.color:
    print("Assumes color image...")

if args.gt_path:
    print("Ground truth path:", args.gt_path)

dir_path = args.dir_path
output_path = args.output_path
gt_path = args.gt_path
rgb = args.color
pp_flag = args.meanshiftpp
bw = args.bandwidth


if pp_flag == False:
    times = {}
    cluster_numbers = {}
    methods = ['MeanShift']
    metric_avg = {}

    for m in methods:
        times[m] = []
        cluster_numbers[m] = []
        metric_avg[m] = {}

    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    print('Running meanshift experiments...')
    print('Saving at output.txt...')
    with open(os.path.join(output_path, 'output.txt'), 'w+') as file:
        for img_path in os.listdir(dir_path):
            # if os.path.exists(os.path.join(output_path, img_path)):
            #     continue
            if rgb:
                img = np.load(os.path.join(dir_path, img_path))['arr_0']
                gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
                lab_img = cv.cvtColor(img, cv.COLOR_RGB2LAB)
                s_t = time()
                # norm_img = lab_img / 255
                norm_img = img / 255
                ori_img = img
                img = lab_img
                data = img2d_to_ijv_color(norm_img)
            else:
                img = np.load(os.path.join(dir_path, img_path))['arr_0']
                gray_img = img
                rgb_img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
                rgb_img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
                s_t = time()
                norm_img = np.interp(img, (np.percentile(img, 1), np.percentile(img, 99)), (0, 1))
                data = img2d_to_ijv_features(norm_img)
                data[..., -1] = data[..., -1] * 2
            shape = img.shape[:2]

            data[..., 0] = data[..., 0] / shape[0]
            data[..., 1] = data[..., 1] / shape[1]
            data[..., 2:] = data[..., 2:] * 0.6


            labels = MeanShift(bandwidth=bw, bin_seeding=True).fit(data).labels_

            label_img = np.reshape(labels, shape)

            cnt = 0
            for i in range(shape[0]):
                for j in range(shape[1]):
                    label_img[i, j] = labels[cnt]
                    cnt += 1

            times['MeanShift'].append(time()-s_t)
            cluster_numbers['MeanShift'].append(len(np.unique(labels)))


            segments = [label_img]


            bounds = [find_boundaries(segments[i]) for i in range(len(segments))]

            metrics = []
            for i in range(len(segments)):
                if gt_path:
                    # Some line to load gt
                    gt = load_gt(os.path.join(gt_path, img_path))
                    gt_bound = find_boundaries(gt)
                    temp = get_metrics(img, segments[i], bounds[i], gt=True, gt_bound=gt_bound)
                else:
                    temp = get_metrics(img, segments[i], bounds[i])
                temp['time'] = times[methods[i]][-1]
                temp['cluster_n'] = cluster_numbers[methods[i]][-1]
                metrics.append(temp)


            for i, m in enumerate(methods):
                for key in metrics[i].keys():
                    try:
                        metric_avg[m][key].append(metrics[i][key])
                    except:
                        metric_avg[m][key] = [metrics[i][key]]

            cnt += 1
            np.savez(os.path.join(output_path, img_path), img=ori_img, label=gt, MeanShift=label_img)
            import json
            with open(os.path.join(output_path, img_path[:-8]+'_metrics_eval.json'), "w") as fp:
                json.dump(metrics, fp)


        for m in methods:
            print(m + ': average and std', file=file)
            for key in metrics[i].keys():
                print(key, np.mean(metric_avg[m][key]), file=file)
                print(key, np.std(metric_avg[m][key]), file=file)


else:
    times = {}
    cluster_numbers = {}
    methods = ['MeanShiftpp']
    metric_avg = {}

    for m in methods:
        times[m] = []
        cluster_numbers[m] = []
        metric_avg[m] = {}

    if os.path.exists(output_path) == False:
        os.makedirs(output_path)

    print('Running meanshift++ experiments...')
    print('Saving at output.txt...')
    with open(os.path.join(output_path, 'output.txt'), 'w+') as file:
        for img_path in os.listdir(dir_path):
            # if os.path.exists(os.path.join(output_path, img_path)):
            #     continue
            if rgb:
                img = np.load(os.path.join(dir_path, img_path))['arr_0']
                gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
                lab_img = cv.cvtColor(img, cv.COLOR_RGB2LAB)
                s_t = time()
                # norm_img = lab_img / 255
                norm_img = img / 255
                ori_img = img
                img = lab_img
                data = img2d_to_ijv_color(norm_img)
            else:
                img = np.load(os.path.join(dir_path, img_path))['arr_0']
                gray_img = img
                rgb_img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
                rgb_img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
                s_t = time()
                norm_img = np.interp(img, (np.percentile(img, 1), np.percentile(img, 99)), (0, 1))
                data = img2d_to_ijv_features(norm_img)
                data[..., -1] = data[..., -1] * 2
            shape = img.shape[:2]

            data[..., 0] = data[..., 0] / shape[0]
            data[..., 1] = data[..., 1] / shape[1]
            data[..., 2:] = data[..., 2:] * 0.6


            labels = MeanShiftPP.MeanShiftPP(bandwidth=bw).fit_predict(data)
            print(np.max(labels))

            label_img = np.reshape(labels, shape)

            cnt = 0
            for i in range(shape[0]):
                for j in range(shape[1]):
                    label_img[i, j] = labels[cnt]
                    cnt += 1

            times['MeanShiftpp'].append(time()-s_t)
            cluster_numbers['MeanShiftpp'].append(len(np.unique(labels)))


            segments = [label_img]


            bounds = [find_boundaries(segments[i]) for i in range(len(segments))]

            metrics = []
            for i in range(len(segments)):
                if gt_path:
                    # Some line to load gt
                    gt = load_gt(os.path.join(gt_path, img_path))
                    gt_bound = find_boundaries(gt)
                    temp = get_metrics(img, segments[i], bounds[i], gt=True, gt_bound=gt_bound)
                else:
                    temp = get_metrics(img, segments[i], bounds[i])
                temp['time'] = times[methods[i]][-1]
                temp['cluster_n'] = cluster_numbers[methods[i]][-1]
                metrics.append(temp)


            for i, m in enumerate(methods):
                for key in metrics[i].keys():
                    try:
                        metric_avg[m][key].append(metrics[i][key])
                    except:
                        metric_avg[m][key] = [metrics[i][key]]

            cnt += 1
            np.savez(os.path.join(output_path, img_path), img=ori_img, label=gt, MeanShiftpp=label_img)
            import json
            with open(os.path.join(output_path, img_path[:-8]+'_metrics_eval.json'), "w") as fp:
                json.dump(metrics, fp)


        for m in methods:
            print(m + ': average and std', file=file)
            for key in metrics[i].keys():
                print(key, np.mean(metric_avg[m][key]), file=file)
                print(key, np.std(metric_avg[m][key]), file=file)

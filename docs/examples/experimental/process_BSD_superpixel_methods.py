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

from fury.colormap import distinguishable_colormap

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
    return_dict['REC'] = boundary_recall(gt_bound, sp_bound)
    return_dict['US'], overlap_graph = underseg_error(gt_bound, sp_bound)
    return_dict['USB'], _ = underseg_bergh_error(gt_bound, sp_bound, overlap_graph)
    return_dict['USN'], _ = underseg_np_error(gt_bound, sp_bound, overlap_graph)
    return_dict['EV'] = explained_variation(img, sp_label)
    return_dict['CO'] = compactness(sp_label)

    return return_dict

def load_gt(gt_path):
    gt_path = gt_path[:-4] + '.mat'
    if gt_path.endswith('.mat'):
        data = loadmat(gt_path)
        data_len = data['groundTruth'].shape[1]
        return [data['groundTruth'][0][i][0][0][0] for i in range(data_len)]
    else:
        return [np.load(gt_path)['arr_0']]
    


slic_channel_axis=-1
dir_path = '/nobackup/jp109/thetan_experiments/BSD/images/all'
gt_path = '/nobackup/jp109/thetan_experiments/BSD/ground_truth/all'
output_path = '/l/research/ps-oct.NOBACKUP/test_BSD_images'

if os.path.exists(output_path) == False:
    os.makedirs(output_path)

with open('test_BSD.txt', 'w+') as file:
    for i_idx, img_path in enumerate(os.listdir(dir_path)):
        if img_path.endswith('.jpg') == False:
            continue
        gts = load_gt(os.path.join(gt_path, img_path))
        gt_bounds = []
        for gt in gts:
            gt_bounds.append(find_boundaries(gt))

        img = cv.imread(os.path.join(dir_path, img_path))
    
        gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        lab_img = cv.cvtColor(img, cv.COLOR_RGB2LAB)
        s_t = time()
        norm_img = lab_img.astype(float)
        norm_img[..., 0] = norm_img[..., 0] / 100
        norm_img[..., 1:] = norm_img[..., 1:] / 255
        ori_img = img
        img = lab_img
        data = img2d_to_ijv_color(norm_img)

        shape = img.shape[:2]

        data[..., 0] = data[..., 0] / shape[0]
        data[..., 1] = data[..., 1] / shape[1]
        data[..., 2:] = data[..., 2:] * 0.6

        plt.figure()
        plt.imshow(ori_img)
        plt.axis('off')
        plt.savefig(os.path.join(output_path, str(i_idx)+'_img.svg'), bbox_inches='tight', pad_inches = 0)
        plt.close()

        gt = gts[0]
        colormap = distinguishable_colormap(nb_colors=np.max(np.unique(gt))+1)
        
        color_label = np.zeros((shape[0], shape[1], 3))
        for i in range(shape[0]):
            for j in range(shape[1]):
                color_label[i, j] = colormap[gt[i, j]]
        
        plt.figure()
        plt.imshow(color_label)
        plt.axis('off')
        plt.savefig(os.path.join(output_path, str(i_idx)+'_gt.svg'), bbox_inches='tight', pad_inches = 0)
        plt.close()

        # 0.17
        TB_data_runtime = time()-s_t
        for TB_th in [0.17]:
            s_t = time()
            results = berserker2(data, TB_th, all_out=False)
            labels = results[1]

            label_img = np.reshape(labels, shape)

            cnt = 0
            for i in range(shape[0]):
                for j in range(shape[1]):
                    label_img[i, j] = labels[cnt]
                    cnt += 1
            TB_clust_runtime = time()-s_t
            print('image', str(i_idx), 'TB ' + str(TB_th) + ' metrics', file=file)
            temp_dict = {}
            for gt_idx in range(len(gts)):
                metric_dict = get_metrics(img, label_img, find_boundaries(label_img), gts[gt_idx], gt_bounds[gt_idx])
                if gt_idx == 0:
                    for key in metric_dict.keys():
                        temp_dict[key] = []
                for key in metric_dict.keys():
                    temp_dict[key].append(metric_dict[key])
            
            metric_dict = {}
            for key in temp_dict.keys():
                metric_dict[key] = np.mean(temp_dict[key])

            print(metric_dict, file=file)
            print('Time: ', str(TB_data_runtime+TB_clust_runtime), file=file)

            colormap = distinguishable_colormap(nb_colors=np.max(np.unique(label_img))+1)
        
            color_label = np.zeros((shape[0], shape[1], 3))
            for i in range(shape[0]):
                for j in range(shape[1]):
                    color_label[i, j] = colormap[label_img[i, j]]
            
            plt.figure()
            plt.imshow(color_label)
            plt.axis('off')
            plt.savefig(os.path.join(output_path, str(i_idx) + '_TB_' + str(TB_th) + '_color.svg'), bbox_inches='tight', pad_inches = 0)
            plt.close()

            mean_img = np.zeros((shape[0], shape[1], 3))
            mean_vs = [np.mean(ori_img[label_img==l], axis=0) for l in range(np.max(np.unique(label_img))+1)]
            for i in range(shape[0]):
                for j in range(shape[1]):
                    mean_img[i, j] = mean_vs[label_img[i, j]]
            mean_img = np.clip(np.round(mean_img), 0, 255).astype(np.uint8)
            plt.figure()
            plt.imshow(mean_img)
            plt.axis('off')
            plt.savefig(os.path.join(output_path, str(i_idx) + '_TB_' + str(TB_th) + '_avg.svg'), bbox_inches='tight', pad_inches = 0)
            plt.close()

        # 300 30
        for scale in [300]:
            for min_size in [30]:
                s_t = time()
                segments_fz = felzenszwalb(img, scale=scale, sigma=1.0, min_size=min_size)
                fz_time = time()-s_t
                print('image', str(i_idx), 'Felz ' + str(scale) + ' ' + str(min_size) + ' metrics', file=file)
                temp_dict = {}
                for gt_idx in range(len(gts)):
                    metric_dict = get_metrics(img, segments_fz, find_boundaries(segments_fz), gts[gt_idx], gt_bounds[gt_idx])
                    if gt_idx == 0:
                        for key in metric_dict.keys():
                            temp_dict[key] = []
                    for key in metric_dict.keys():
                        temp_dict[key].append(metric_dict[key])
                
                metric_dict = {}
                for key in temp_dict.keys():
                    metric_dict[key] = np.mean(temp_dict[key])

                print(metric_dict, file=file)
                print('Time: ', str(fz_time), file=file)

                colormap = distinguishable_colormap(nb_colors=np.max(np.unique(segments_fz))+1)
        
                color_label = np.zeros((shape[0], shape[1], 3))
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        color_label[i, j] = colormap[segments_fz[i, j]]
                
                plt.figure()
                plt.imshow(color_label)
                plt.axis('off')
                plt.savefig(os.path.join(output_path, str(i_idx) + '_Felz_' + str(scale) + '_' + str(min_size) + '_color.svg'), bbox_inches='tight', pad_inches = 0)
                plt.close()

                mean_img = np.zeros((shape[0], shape[1], 3))
                mean_vs = [np.mean(ori_img[segments_fz==l], axis=0) for l in range(np.max(np.unique(segments_fz))+1)]
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        mean_img[i, j] = mean_vs[segments_fz[i, j]]
                mean_img = np.clip(np.round(mean_img), 0, 255).astype(np.uint8)
                plt.figure()
                plt.imshow(mean_img)
                plt.axis('off')
                plt.savefig(os.path.join(output_path, str(i_idx) + '_Felz_' + str(scale) + '_' + str(min_size) + '_avg.svg'), bbox_inches='tight', pad_inches = 0)
                plt.close()

        # 200 0.1
        for n_segments in [200]:
            for compactness_v in [0.1]:
                s_t = time()
                segments_slic = slic(lab_img, n_segments=n_segments, compactness=compactness_v, sigma=1, start_label=1, channel_axis=slic_channel_axis, slic_zero=True)
                slic_time = time()-s_t
                print('image', str(i_idx), 'SLICO ' + str(n_segments) + ' ' + str(compactness_v) + ' metrics', file=file)
                temp_dict = {}
                for gt_idx in range(len(gts)):
                    metric_dict = get_metrics(img, segments_slic, find_boundaries(segments_slic), gts[gt_idx], gt_bounds[gt_idx])
                    if gt_idx == 0:
                        for key in metric_dict.keys():
                            temp_dict[key] = []
                    for key in metric_dict.keys():
                        temp_dict[key].append(metric_dict[key])
                
                metric_dict = {}
                for key in temp_dict.keys():
                    metric_dict[key] = np.mean(temp_dict[key])

                print(metric_dict, file=file)
                print('Time: ', str(slic_time), file=file)

                colormap = distinguishable_colormap(nb_colors=np.max(np.unique(segments_slic))+1)
        
                color_label = np.zeros((shape[0], shape[1], 3))
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        color_label[i, j] = colormap[segments_slic[i, j]]
                
                plt.figure()
                plt.imshow(color_label)
                plt.axis('off')
                plt.savefig(os.path.join(output_path, str(i_idx) + '_SLIC_' + str(n_segments) + '_' + str(compactness_v) + '_color.svg'), bbox_inches='tight', pad_inches = 0)
                plt.close()

                mean_img = np.zeros((shape[0], shape[1], 3))
                mean_vs = [np.mean(ori_img[segments_slic==l], axis=0) for l in range(np.max(np.unique(segments_slic))+1)]
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        mean_img[i, j] = mean_vs[segments_slic[i, j]]
                mean_img = np.clip(np.round(mean_img), 0, 255).astype(np.uint8)
                plt.figure()
                plt.imshow(mean_img)
                plt.axis('off')
                plt.savefig(os.path.join(output_path, str(i_idx) + '_SLIC_' + str(n_segments) + '_' + str(compactness_v) + '_avg.svg'), bbox_inches='tight', pad_inches = 0)
                plt.close()

        # 7 30
        for kernel_size in [7]:
            for max_dist in [30]:
                s_t = time()
                segments_quick = quickshift(img, kernel_size=kernel_size, max_dist=max_dist, ratio=0.5)
                quick_time = time()-s_t
                print('image', str(i_idx), 'Quickshift ' + str(kernel_size) + ' ' + str(max_dist) + ' metrics', file=file)
                temp_dict = {}
                for gt_idx in range(len(gts)):
                    metric_dict = get_metrics(img, segments_quick, find_boundaries(segments_quick), gts[gt_idx], gt_bounds[gt_idx])
                    if gt_idx == 0:
                        for key in metric_dict.keys():
                            temp_dict[key] = []
                    for key in metric_dict.keys():
                        temp_dict[key].append(metric_dict[key])
                
                metric_dict = {}
                for key in temp_dict.keys():
                    metric_dict[key] = np.mean(temp_dict[key])

                print(metric_dict, file=file)
                print('Time: ', str(quick_time), file=file)

                colormap = distinguishable_colormap(nb_colors=np.max(np.unique(segments_quick))+1)
        
                color_label = np.zeros((shape[0], shape[1], 3))
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        color_label[i, j] = colormap[segments_quick[i, j]]
                
                plt.figure()
                plt.imshow(color_label)
                plt.axis('off')
                plt.savefig(os.path.join(output_path, str(i_idx) + '_Quickshift_' + str(kernel_size) + '_' + str(max_dist) + '_color.svg'), bbox_inches='tight', pad_inches = 0)
                plt.close()

                mean_img = np.zeros((shape[0], shape[1], 3))
                mean_vs = [np.mean(ori_img[segments_quick==l], axis=0) for l in range(np.max(np.unique(segments_quick))+1)]
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        mean_img[i, j] = mean_vs[segments_quick[i, j]]
                mean_img = np.clip(np.round(mean_img), 0, 255).astype(np.uint8)
                plt.figure()
                plt.imshow(mean_img)
                plt.axis('off')
                plt.savefig(os.path.join(output_path, str(i_idx) + '_Quickshift_' + str(kernel_size) + '_' + str(max_dist) + '_avg.svg'), bbox_inches='tight', pad_inches = 0)
                plt.close()


        # 125 0.00001
        for markers in [125]:
            for compactness_v in [0.00001]:
                s_t = time()
                gradient = sobel(gray_img)
                segments_watershed = watershed(gradient, markers=markers, compactness=compactness_v)
                watershed_time = time()-s_t
                print('image', str(i_idx), 'Watershed ' + str(markers) + ' ' + str(compactness_v) + ' metrics', file=file)
                temp_dict = {}
                for gt_idx in range(len(gts)):
                    metric_dict = get_metrics(img, segments_watershed, find_boundaries(segments_watershed), gts[gt_idx], gt_bounds[gt_idx])
                    if gt_idx == 0:
                        for key in metric_dict.keys():
                            temp_dict[key] = []
                    for key in metric_dict.keys():
                        temp_dict[key].append(metric_dict[key])
                
                metric_dict = {}
                for key in temp_dict.keys():
                    metric_dict[key] = np.mean(temp_dict[key])

                print(metric_dict, file=file)
                print('Time: ', str(watershed_time), file=file)

                colormap = distinguishable_colormap(nb_colors=np.max(np.unique(segments_watershed))+1)
        
                color_label = np.zeros((shape[0], shape[1], 3))
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        color_label[i, j] = colormap[segments_watershed[i, j]]
                
                plt.figure()
                plt.imshow(color_label)
                plt.axis('off')
                plt.savefig(os.path.join(output_path, str(i_idx) + '_Watershed_' + str(markers) + '_' + str(compactness_v) + '_color.svg'), bbox_inches='tight', pad_inches = 0)
                plt.close()

                mean_img = np.zeros((shape[0], shape[1], 3))
                mean_vs = [np.mean(ori_img[segments_watershed==l], axis=0) for l in range(np.max(np.unique(segments_watershed))+1)]
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        mean_img[i, j] = mean_vs[segments_watershed[i, j]]
                mean_img = np.clip(np.round(mean_img), 0, 255).astype(np.uint8)
                plt.figure()
                plt.imshow(mean_img)
                plt.axis('off')
                plt.savefig(os.path.join(output_path, str(i_idx) + '_Watershed_' + str(markers) + '_' + str(compactness_v) + '_avg.svg'), bbox_inches='tight', pad_inches = 0)
                plt.close()
        
        if i_idx == 2:
            break
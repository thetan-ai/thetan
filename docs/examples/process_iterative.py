from time import time
import numpy as np
import pylab as pl
import pandas as pd
from thetan.berserker import berserker2, berserker2_iterative
from sklearn.metrics import normalized_mutual_info_score, silhouette_score, fowlkes_mallows_score
from thetan.theta import distance_matrix as tdm
from thetan.theta import theta_g1 as tgs


def evaluate(ecentroids=None, centroids=None, thr=0.5, X=None, label=None, true_label=None):

    D = np.sqrt(tdm(ecentroids, centroids))
    closest = np.min(D, axis=1)
    count = np.sum(closest < thr)
    ac_score = count / len(ecentroids)

    nmi_score = normalized_mutual_info_score(true_label, label)

    sil_score = silhouette_score(X, label)

    fms_score = fowlkes_mallows_score(true_label, label)

    return ac_score, nmi_score, sil_score, fms_score, ecentroids.shape[0]


def gaussian_clusters(n_local_samples, blob_dis = 10, cluster_x = 10, cluster_y = 10):

    xx = np.arange(-1*blob_dis*cluster_x//2, blob_dis*cluster_x//2, blob_dis)
    yy = np.arange(-1*blob_dis*cluster_y//2, blob_dis*cluster_y//2, blob_dis)

    xx, yy = np.meshgrid(xx, yy)
    n_centres = np.hstack((np.ravel(xx)[:, np.newaxis],
                        np.ravel(yy)[:, np.newaxis]))

    gt_centroids = []
    X = np.zeros((n_local_samples * n_centres.shape[0], 2))
    y = np.zeros((n_local_samples * n_centres.shape[0]), dtype=np.int16)
    for i in range(n_centres.shape[0]):
        cpos = n_centres[i]

        local_X = cpos + np.random.normal(0, 1, (n_local_samples, 2))
        X[i * n_local_samples: (i + 1) * n_local_samples] = local_X
        y[i * n_local_samples: (i + 1) * n_local_samples] = i
        gt_centroids.append(local_X.mean(axis=0))

    return X, y, gt_centroids


def uniform_clusters(scale = 4, n_local_samples = 30, blob_dis = 33, cluster_per_dim = 10):

    xx = np.arange(-1*blob_dis, blob_dis, blob_dis / cluster_per_dim*2)
    yy = np.arange(-1*blob_dis, blob_dis, blob_dis / cluster_per_dim*2)

    xx, yy = np.meshgrid(xx, yy)
    n_centres = np.hstack((np.ravel(xx)[:, np.newaxis],
                        np.ravel(yy)[:, np.newaxis]))

    gt_centroids = []
    X = np.zeros((n_local_samples * n_centres.shape[0], 2))
    y = np.zeros((n_local_samples * n_centres.shape[0]), dtype=np.int16)
    for i in range(n_centres.shape[0]):
        cpos = n_centres[i]
        local_X = cpos + scale*(np.random.rand(n_local_samples, 2) - 0.5)

        X[i * n_local_samples: (i + 1) * n_local_samples] = local_X
        y[i * n_local_samples: (i + 1) * n_local_samples] = i
        gt_centroids.append(local_X.mean(axis=0))

    return X, y, gt_centroids


def disk_clusters(sz = (20, 6), scale=5, samples=100):

    coords = []
    centroids = []
    Y = [] # labels

    cnt = 0
    for x, y in np.ndindex(sz):

        for d in range(samples):
            r = np.sqrt(np.random.random())
            theta = np.random.random() * 2 * np.pi
            x1 = scale * x + r * np.cos(theta)
            y1 = scale * y + r * np.sin(theta)
            coords.append([x1, y1])
            Y.append(cnt)

        cnt += 1
        centroids.append([scale * x, scale * y])

    points = np.array(coords)
    centroids = np.array(centroids, dtype='f4')

    return points, Y, centroids


def plot_centroids_clusters(data, centroids, labels, title, exp_type, initial_centroids=None,
                            uncertain_data=None, final_counts=None, final_nearc_value=None,
                            merged_centroids=None, show=True, dpath=None):

    new_colors = np.zeros((len(data), 3))
    no_clusters = len(centroids)
    cluster_colors = np.random.rand(no_clusters, 3)

    for ind in range(data.shape[0]):
        new_colors[ind] = cluster_colors[labels[ind]]

    fig = pl.figure(figsize=(30, 15))
    pl.title(title)
    pl.axis('equal')
    pl.scatter(data[:, 0], data[:, 1], c=new_colors, alpha=0.2)
    pl.plot(centroids[:, 0], centroids[:, 1], 'rx')

    for c in range(centroids.shape[0]):
        pl.text(centroids[c, 0], centroids[c, 1], str(c), fontsize=8)

    if initial_centroids is not None:
        pl.plot(initial_centroids[:, 0], initial_centroids[:, 1], 'bx')

    if uncertain_data is not None:
        pl.plot(uncertain_data[:, 0], uncertain_data[:, 1], 'g.')

    cnt = 0
    if final_counts is not None:
        for i in range(final_counts.shape[0]):
            prev = cnt
            cnt += final_counts[i]
            if final_counts[i] > 1:
                pl.text(data[i, 0], data[i, 1], str(final_nearc_value[prev: cnt]) + ' ' + str(labels[i]), fontsize=8)

    if merged_centroids is not None:
        pl.plot(merged_centroids[:, 0], merged_centroids[:, 1], 'mo')

    if show:
        pl.show()

    pl.savefig(dpath + '2_' + exp_type + '_' + title + '.png', dpi=300)
    pl.close(fig)


def simulate_data(dist_type='gaussian',blob_dis=10):

    print('Simulating data')

    if dist_type == 'gaussian':
        X, y, gt_centroids = gaussian_clusters(n_local_samples=500, blob_dis = blob_dis, cluster_x = 30, cluster_y = 10) # cluster_x should be 30

    if dist_type == 'uniform':
        X, y, gt_centroids = uniform_clusters(scale = 4, n_local_samples = 30, blob_dis = 33, cluster_per_dim = 10)

    if dist_type == 'disk':
        X, y, gt_centroids = disk_clusters(sz = (20, 6), scale=5, samples=100)

    gt_centroids = np.array(gt_centroids)

    points = X
    labels = y
    # sz = len(n_centres)

    centroids = gt_centroids

    print('Before shuffling')
    print(points[:10])

    points_idx = np.arange(np.shape(points)[0])
    np.random.shuffle(points_idx)
    new_points = np.zeros(np.shape(points))
    new_labels = np.zeros(np.shape(labels))
    for i, idx in zip(range(np.shape(points)[0]), points_idx):
        new_points[i] = points[idx]
        new_labels[i] = labels[idx]

    points = new_points
    labels = new_labels

    return points, labels, gt_centroids


# rows = ['TB2', 'TB4', 'TB6', 'TB8', 'TB10', 'TB12']

rows = ['TB' + str(i) for i in range(2, 15)]

cols = ['AC', 'NMI', 'SIL', 'FMS', 'Clusters']


# dpath = '/home/elef/Data/tb_output_iteration/'
# dpath = 'C:/Users/elef/Data/tb_output_iteration/'
dpath = 'C:/Users/Eleftherios/Data/tb_output_iteration/'

run_experiments = True
plot_results = True

reps = 4
save_results = True
evaluation = True

theta_value = 3.6
iter_max = 2


if run_experiments:

    t_start = time()

    for expno in range(reps):

        print('######################################### > Experiment ', str(expno), ' Go Go Team Thetan!')

        evaluations = {}
        evaluation_thr = 0.1
        verbose = False

        X, L, gt_centroids = simulate_data('gaussian', blob_dis=5)

        print('Number of features ', X.shape[0])
        print('Number of GT centroids ', gt_centroids.shape[0])

        n_clusters = gt_centroids.shape[0]

        for i in range(2, 15):

            print('Actual iter_max ', iter_max * expno + 2)

            res = berserker2_iterative(X, theta_value, all_out=True, iter_max=i)

            even_centroids_ex2, updated_labels_ex2, initial_centroids, initial_labels, even_centroids_ex, updated_labels2, p_centroids, p_labels, ind_inv, ind2_inv = res

        
            centroids = even_centroids_ex2
            labels = updated_labels_ex2

            if verbose:
                plot_centroids_clusters(X, centroids, labels, 'TB', exp_type=str(expno), initial_centroids=gt_centroids, show=False, dpath=dpath)

            if evaluation:
                print('Starting evaluation')
                t = time()
                scores = evaluate(ecentroids=centroids, centroids=gt_centroids, thr=evaluation_thr, X=X, label=labels, true_label=L)
                print(scores)
                print('Duration is ', time()-t)
                evaluations['TB' + str(i)] = scores


        if evaluation:
            print('ALL EVALUATIONS')
            print(evaluations)

            evals = np.zeros((len(rows), len(cols)))

            for i, row in enumerate(rows):
                for j, col in enumerate(cols):
                    evals[i, j] = evaluations[row][j]

            df = pd.DataFrame(evals, columns=cols)
            df.index = rows

            print('Printing dataframe of evaluations ... ' + str(expno))
            print(df)

            if save_results:
                df.to_excel(dpath + "expno_" + str(expno) + ".xlsx")
                print('Saving evaluations')
    
    t_end = time()
    print('Total Duration ', t_end - t_start)

files = []
if plot_results:
    for expno in range(reps):
        fname_ = dpath + "expno_" + str(expno) + ".xlsx"
        files.append(fname_)
        print(pd.read_excel(fname_))
        print()

df = pd.concat([pd.read_excel(f) for f in files])
result = df.groupby('Unnamed: 0').agg(['mean', 'std'])

import matplotlib.pyplot as plt
import numpy as np

t = 1

# x = np.array([1, 2, 3, 4, 5])
# y = result['AC']['mean'].values
# e = result['AC']['std'].values

# plt.figure()
# plt.title('Iteration study - AC - Error plot')
# plt.errorbar(x, y, e, linestyle='None', marker='o', capsize=3)
# plt.xticks(x, result.index.tolist())
# plt.grid()

# ac = df['AC'].to_numpy()
# data = ac.reshape(10, 5)
# plt.figure()
# plt.title('Iteration study - AC - Violin plot')
# plt.violinplot(data, showmeans=True)
# plt.xticks(x, result.index.tolist())
# plt.grid()

# plt.figure()
# plt.title('Iteration study - AC - Box plot')
# plt.boxplot(data)
# plt.xticks(x, result.index.tolist())
# plt.grid()

# plt.show()

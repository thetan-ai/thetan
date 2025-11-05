from time import time
import numpy as np
from thetan.berserker import berserker2, berserker2_clean, random_walk_auto
import pylab as pl
import tracemalloc
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from thetan.theta import assign, centroids_from_labels
from thetan.theta import distance_matrix as tdm

# from genieclust.compare_partitions import normalized_clustering_accuracy
from sklearn.metrics import normalized_mutual_info_score, silhouette_score, fowlkes_mallows_score


def evaluate(ecentroids=None, centroids=None, thr=0.5, X=None, label=None, true_label=None, duration=-1, memory=None):

    D = np.sqrt(tdm(ecentroids, centroids))
    closest = np.min(D, axis=1)
    count = np.sum(closest < thr)
    ac_score = count / len(ecentroids)

    nmi_score = normalized_mutual_info_score(true_label, label)

    sil_score = silhouette_score(X, label)

    fms_score = fowlkes_mallows_score(true_label, label)

    return ac_score, nmi_score, sil_score, fms_score, ecentroids.shape[0], duration, memory


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


def escape_clusters(n_local_samples, blob_dis = 10, cluster_x = 10, cluster_y = 10, i_x = 0, escape_x = 50):

    xx = np.arange(-1*blob_dis*cluster_x//2, blob_dis*cluster_x//2, blob_dis)
    yy = np.arange(-1*blob_dis*cluster_y//2, blob_dis*cluster_y//2, blob_dis)

    xx, yy = np.meshgrid(xx, yy)
    n_centres = np.hstack((np.ravel(xx)[:, np.newaxis],
                        np.ravel(yy)[:, np.newaxis]))

    gt_centroids = []
    X = np.zeros((n_local_samples * n_centres.shape[0], 2))
    y = np.zeros((n_local_samples * n_centres.shape[0]), dtype=np.int16)
    for i in range(n_centres.shape[0]):

        if i == i_x:
            cpos = n_centres[i] + np.array([[escape_x, 0]])
        else:
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

    # for c in range(centroids.shape[0]):
    #    pl.text(centroids[c, 0], centroids[c, 1], str(c), fontsize=8)

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
    if dpath is not None:
        pl.savefig(dpath + '2_' + exp_type + '_' + title + '.png', dpi=300)
        pl.close(fig)


def simulate_data(dist_type='gaussian',blob_dis=10, cluster_x=10, cluster_y=10):

    print('Simulating data')

    if dist_type == 'gaussian':
        X, y, gt_centroids = gaussian_clusters(n_local_samples=500, blob_dis = blob_dis, cluster_x = cluster_x, cluster_y = cluster_y) # cluster_x should be 30

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


def ridgeline(data, bins, overlap=0, fill=True, labels=None, n_points=150):
    """
    Creates a standard ridgeline plot.

    data, list of lists.
    overlap, overlap between distributions. 1 max overlap, 0 no overlap.
    fill, matplotlib color to fill the distributions.
    n_points, number of points to evaluate each distribution function.
    labels, values to place on the y axis to describe the distributions.
    """
    if overlap > 1 or overlap < 0:
        raise ValueError('overlap must be in [0 1]')
    xx = np.linspace(np.min(np.concatenate(data)),
                     np.max(np.concatenate(data)), n_points)
    curves = []
    ys = []

    plt.figure(figsize=(20, 12))

    for i, d in enumerate(data):
        d2 = gaussian_filter1d(d, sigma=2.)
        # x = np.linspace(d2.min(), d2.max(), n_points)
        # pdf = gaussian_kde(d)
        # y = i*(1.0-overlap)
        # ys.append(y)
        # curve = pdf(xx)
        # if fill:
        #     plt.fill_between(xx, np.ones(n_points)*y,
        #                      curve+y, zorder=len(data)-i+1)
        # plt.plot(xx, curve+y, c='k', zorder=len(data)-i+1)
        # plt.plot(xx, d2, label=str(i)) # + i * 200000)
        plt.plot(bins[i][:-1], d2, label=labels[i], linewidth=3)
        plt.fill_between(bins[i][:-1], np.zeros(n_points), d2, alpha=0.5)
    # if labels:
    #     plt.yticks(ys, labels)

    legend = plt.legend(fontsize=22)
    legend.set_title(title='Inter-class distance', prop={'size': 22})
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel('Distance ranges', fontsize=24, labelpad=22)
    plt.ylabel('Density', fontsize=24, labelpad=22)
    plt.show()


def random_walks():
    dpath = '/home/elef/Data/tb_output_predicts/'

    # X, y, gt_centroids = escape_clusters(500, blob_dis = 10, cluster_x = 10, cluster_y = 10, i_x = 0, escape_x = -50)
    # gt_centroids = np.asarray(gt_centroids)

    # plot_centroids_clusters(X, gt_centroids, y, 'Escape', 'large')

    t = 1

    data = []
    bins = []
    # fig = plt.figure()
    # [5, 25, 45]: # [5, 15, 25, 35, 45]
    # [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    inter_dists = [5, 15, 25, 35, 45]
    for blob_dis in inter_dists:

        print('Blob dis', blob_dis)

        X, L, gt_centroids = simulate_data('gaussian', blob_dis=blob_dis, cluster_x=30, cluster_y=10)
        gt_centroids = np.asarray(gt_centroids)

        # X2, y2, gt_centroids2 = escape_clusters(500, blob_dis = 10, cluster_x = 10, cluster_y = 10, i_x = 0, escape_x = -blob_dis)
        # gt_centroids2 = np.asarray(gt_centroids2)

        # X = np.concatenate((X, X2))
        # gt_centroids = np.concatenate((gt_centroids, gt_centroids2))

        # plot_centroids_clusters(X, gt_centroids, y, 'Escape', 'large')

        print('Number of features ', X.shape[0])
        print('Number of GT centroids ', gt_centroids.shape[0])
        print('Size of X in MBytes', X.nbytes // 1024 ** 2)

        verbose = True

        print('\n >> Aiming to predict theta ...')

        dists = random_walk_auto(X, reps=100)
        dists = np.asarray(dists)

        print('Size of Dists in MBytes', dists.nbytes // 1024 ** 2)

        # fig = plt.figure()

        # plt.title('Inter-class distance at ' + str(blob_dis))
        # axes = plt.gca()
        # axes.set_xlim(0, 1000)
        # axes.set_ylim(0, 1000000)

        # plt.hist(dists.ravel(), bins=100, alpha=0.2)
        # # pl.show()

        # fig.savefig(dpath + 'hist_' + str(blob_dis) + '.png', dpi=300)
        # data.append(dists.ravel())
        # for d in dists:
        #     data.append(d)

        hist_, binv = np.histogram(dists.ravel(), bins=100)
        data.append(hist_)
        bins.append(binv)

    labels = [str(i).zfill(2) for i in inter_dists]
    ridgeline(data, bins, overlap=0, fill=True, labels=labels, n_points=100)


if __name__ == '__main__':

    random_walks()

    1/0

    verbose = True
    dpath = '/home/elef/Data/tb_output_subsampling/'
    show = True
    run_experiment = False

    # fig.savefig(dpath + 'hist_all.png', dpi=300)

    # countries = [5, 15, 25, 35, 45]
    # colors = ['#0000ff', '#3300cc', '#660099', '#990066', '#cc0033', '#ff0000']
    # data = np.ascontiguousarray(np.asarray(data).T)

    # plot_ridge(countries, colors, data)

    # [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    sampling_percents = [100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

    if run_experiment:



        all_scores = []
        show = False
        evaluation_thr = 0.1

        for sper in sampling_percents:

            print('Sampling percentage is ... ', sper)

            X, L, gt_centroids = simulate_data('gaussian', blob_dis=5, cluster_x=10, cluster_y=10)
            gt_centroids = np.asarray(gt_centroids)

            print('Selecting ...', int(np.round((sper / 100.) * X.shape[0], 0)))
            ind = np.random.choice(np.arange(X.shape[0]), int(np.round((sper / 100.) * X.shape[0], 0)), replace=False)
            points = X[ind].copy()

            theta_value = 3.6

            print('\n >> Starting Berserker 2 (clean version)...')

            t = time()
            tracemalloc.start()
            centroids, labels = berserker2_clean(points, theta_value)
            amem, peakmem = tracemalloc.get_traced_memory()
            peakmem1 = peakmem // (1024**2)
            tracemalloc.stop()
            duration1 = time() - t
            print('Sub Duration is ', duration1)
            print('Sub Peak memory ', peakmem1)

            t = time()
            tracemalloc.start()
            labels2 = assign(X, centroids)
            centroids2 = centroids_from_labels(X, labels2)
            amem2, peakmem2 = tracemalloc.get_traced_memory()
            peakmem2 = peakmem2 // (1024**2)
            tracemalloc.stop()
            duration2 = time() - t
            print('Assign Duration is ', duration2)
            print('Assign Peak memory ', peakmem2)

            # plot_centroids_clusters(points, centroids, labels, 'TB_' + str(sper).zfill(2), exp_type='_runtime_', initial_centroids=gt_centroids, show=show, dpath=dpath)
            plot_centroids_clusters(X, centroids2, labels2, 'sub_TB_' + str(sper).zfill(2), exp_type='_runtime_', initial_centroids=gt_centroids, show=show, dpath=dpath)

            # TODO: find a nice way to demonstrate this subsampling result
            # Try different evaluation metrics
            # Try also TBSCAN
            # ###

            t = time()
            scores = evaluate(ecentroids=centroids2, centroids=gt_centroids, thr=evaluation_thr, X=X, label=labels2, true_label=L, duration=duration1 + duration2, memory=max(peakmem1, peakmem2))
            print(scores)

            ac_score, nmi_score, sil_score, fms_score, K_, duration, memory = scores

            nca = normalized_clustering_accuracy(labels2, L)
            print(nca)

            all_scores.append([nca, ac_score, nmi_score, sil_score, fms_score, K_, duration, memory, duration1, duration2])
            print('Evaluation duration is ', time() - t)


        data = np.asarray(all_scores)
        np.set_printoptions(3, suppress=True)
        print(data)

# print(all_scores)
t = 1


# dpath2 = 'C:/Users/Eleftherios/Data/tb_sub/'
dpath2 = '/home/elef/Data/tb_output_subsampling/'


tmp = np.load(dpath2 + 'dat_100x100.npy')

import pandas as pd

df = pd.DataFrame(tmp, index=sampling_percents, columns=['nca', 'ac', 'nmi', 'sil', 'fms', 'clusters', 'duration', 'memory', 'duration1', 'duration2'])

print(df)

fig = plt.figure()
plt.plot(df['ac'].to_numpy(), label='AC')
# plt.plot(df['nca'].to_numpy())
# plt.plot(df['nmi'].to_numpy(), label='NMI')
plt.plot(df['fms'].to_numpy(), label='FMS')

runtime = df['duration'].to_numpy()
runtime = runtime/runtime.max()

plt.plot(runtime, label='Runtime')
plt.xticks(np.arange(len(sampling_percents)), sampling_percents)
plt.vlines(15, 0, 1.0, colors='k', linestyles='dashed')
plt.grid()
plt.legend()
plt.show()

t = 2


from random import uniform
import numpy as np

from thetan.theta import theta_g1, assign, theta_g1w
from thetan.theta import distance_matrix as tdm
from thetan.theta import theta_delta
import pylab as pl



def plot_centroids_clusters(data, centroids, labels, title=None, exp_type=None, gt_centroids=None,
                            uncertain_data=None, final_counts=None, final_nearc_value=None,
                            merged_centroids=None, show=True):

    new_colors = np.zeros((len(data), 3))
    no_clusters = len(centroids)
    cluster_colors = np.random.rand(no_clusters, 3)

    for ind in range(len(points)):
        new_colors[ind] = cluster_colors[labels[ind]]

    fig = pl.figure()
    pl.title(title)
    pl.axis('equal')
    pl.scatter(data[:, 0], data[:, 1], c=new_colors, alpha=0.2)

    if centroids is not None:

        pl.plot(centroids[:, 0], centroids[:, 1], 'rx')

        for c in range(centroids.shape[0]):
            pl.text(centroids[c, 0], centroids[c, 1], str(c), fontsize=8)

    if gt_centroids is not None:
        pl.plot(gt_centroids[:, 0], gt_centroids[:, 1], 'bx')

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


n_local_samples = 400

blob_dis = 5
cluster_x = 10
cluster_y = 10

normal_distribution = True
uniform_distribution = False

normal_sigma = 1.0
uniform_low = -2.
uniform_high = 2.

theta_value = 3.6


xx = np.arange(-1 * blob_dis*cluster_x // 2, blob_dis * cluster_x // 2, blob_dis)
yy = np.arange(-1 * blob_dis*cluster_y // 2, blob_dis * cluster_y // 2, blob_dis)

xx, yy = np.meshgrid(xx, yy)
n_centres = np.hstack((np.ravel(xx)[:, np.newaxis],
                       np.ravel(yy)[:, np.newaxis]))

gt_centroids = []
X = np.zeros((n_local_samples * n_centres.shape[0], 2))
y = np.zeros((n_local_samples * n_centres.shape[0]), dtype=np.int16)

for i in range(n_centres.shape[0]):
    cpos = n_centres[i]

    if normal_distribution:
        local_X = cpos + np.random.normal(0, normal_sigma, (n_local_samples, 2))
    if uniform_distribution:
        local_X = cpos + np.random.uniform(uniform_low,
                                           uniform_high,
                                           (n_local_samples, 2))

    X[i * n_local_samples: (i + 1) * n_local_samples] = local_X
    y[i * n_local_samples: (i + 1) * n_local_samples] = i
    gt_centroids.append(local_X.mean(axis=0))

gt_centroids = np.array(gt_centroids)

points = X
labels = y
sz = len(n_centres)

centroids = gt_centroids


plot_centroids_clusters(points, centroids=centroids, labels=labels,
                        title='>> Original data and ground truth centroids, # ' + str(centroids.shape[0]),
                        gt_centroids=centroids, show=False)

tsg_centroids, tsg_labels, tsg_summs, tsg_clusters_sizes = theta_g1(points, theta=theta_value, metric='euclidean', all_out=True)

plot_centroids_clusters(points, centroids=tsg_centroids, labels=tsg_labels,
                        title='>> TSG centroids, # ' + str(tsg_centroids.shape[0]),
                        exp_type='hard', gt_centroids=centroids, show=False)


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

print('After shuffling')
print(points[:10])


print('Number of points ', points.shape)
tsg_centroids, tsg_labels, tsg_summs, tsg_clusters_sizes = theta_g1(points, theta=theta_value, metric='euclidean', all_out=True)


plot_centroids_clusters(points, centroids=tsg_centroids, labels=tsg_labels,
                        title='>> TSG centroids (shuffled), # ' + str(tsg_centroids.shape[0]),
                        exp_type='hard', gt_centroids=centroids, show=False)


def reduce_centroids(centroids, labels, centroids_prev):
    # when two centroids are too close keep the one with a smaller index
    clusters = np.empty((centroids.shape[0], 0)).tolist()
    for i in range(labels.shape[0]):
        clusters[labels[i]].append(i)

    candidates = []
    for c in clusters:
        if len(c) > 1:
            print('yo', c)
            maxc = np.max(c)
            candidates.append(maxc)

    # centroids2 = np.empty((labels.max() + 1, centroids.shape[1]))
    centroids_reduced = np.delete(centroids_prev, candidates, axis=0)

    return centroids_reduced

    # We have to exclude the candidates and reassign their contents.
    # candidatesX = []
    # for i in range(final_indices.shape[0]):
    #     if final_indices[i] in candidates:
    #         candidatesX.append(i)
    # X_candidates = points[candidatesX]

    # final_centroids_reduced = final_centroids.copy()
    # final_centroids_reduced = np.delete(final_centroids_reduced, candidates, axis=0)


def theta_grouping(points, theta=theta_value, metric='euclidean', all_out=True, verbose=False):

    tsg_centroids, tsg_labels, _, _ = theta_g1(points, theta=theta_value, metric=metric, all_out=all_out)
    print('Phase A ', tsg_centroids.shape[0])

    if verbose:
        plot_centroids_clusters(points, centroids=tsg_centroids, labels=tsg_labels,
                                title='>> Phase A TSG centroids (shuffled), # ' + str(tsg_centroids.shape[0]),
                                exp_type='hard', gt_centroids=centroids, show=True)


    tsg_centroids_c, tsg_labels_c, _ , _ = theta_g1(tsg_centroids, theta=theta_value, metric=metric, all_out=all_out)
    print('Phase B ', tsg_centroids_c.shape[0])

    tsg_centroids_r = reduce_centroids(tsg_centroids_c, tsg_labels_c, tsg_centroids)
    print('Phase C ', tsg_centroids_r.shape[0])

    tsg_centroids_m = tsg_centroids_r


    for iter in range(4):

        new_points = np.concatenate((tsg_centroids_m, points))

        sz = tsg_centroids_m.shape[0]

        tsg_centroids2, tsg_labels2, tsg_summs2, tsg_clusters_sizes2 = theta_g1(new_points, theta=theta_value, metric=metric, all_out=all_out)

        print('Phase D ', tsg_centroids2.shape[0], ' Iter ', iter)

        tsg_centroids_m = tsg_centroids2

    return tsg_centroids2, tsg_labels2[sz:], tsg_summs2, tsg_clusters_sizes2


tg_centroids, tg_labels, tg_summs, tg_clusters_sizes = theta_grouping(points, theta=theta_value, metric='euclidean', all_out=True)

plot_centroids_clusters(points, centroids=tg_centroids, labels=tg_labels,
                        title='>> TG centroids (shuffled), # ' + str(tg_centroids.shape[0]),
                        exp_type='hard', gt_centroids=centroids, show=True)

# tsgw_centroids, tsgw_labels, tsgw_summs, tsgw_clusters_sizes = theta_g1w(points, theta=theta_value, metric='euclidean', w=1., all_out=True)


# plot_centroids_clusters(points, centroids=tsgw_centroids, labels=tsgw_labels,
#                         title='>> TSGW centroids (shuffled), # ' + str(tsgw_centroids.shape[0]),
#                         exp_type='hard', gt_centroids=centroids, show=False)


# td_centroids, td_labels, tdsumms, td_clusters_sizes, td_deltas, td_deltasxo = theta_delta(points, theta=theta_value, metric='euclidean', w=1., dw=.01, all_out=True)

# plot_centroids_clusters(points, centroids=td_deltas, labels=td_labels,
#                         title='>> Theta Delta centroids (shuffled), # ' + str(td_centroids.shape[0]),
#                         exp_type='hard', gt_centroids=centroids, show=True)



# final_centroids2, final_indices2, final_summs2, final_clusters_sizes2, final_counts2, final_nearc_value2, final_nearc_dist2 = thetan(tsg_centroids, theta=theta_value, all_out=True)

# clusters = np.empty((final_centroids2.shape[0], 0)).tolist()
# for i in range(final_indices2.shape[0]):
#     clusters[final_indices2[i]].append(i)

# fig = pl.figure()
# pl.axis('equal')
# pl.title('>> Thetan on centroids (2nd round) (shuffled), # ' + str(final_centroids2.shape[0]))
# pl.scatter(points[:, 0], points[:, 1], alpha=0.2)
# pl.plot(centroids[:, 0], centroids[:, 1], 'bx')
# pl.plot(final_centroids2[:, 0], final_centroids2[:, 1], 'rx')
# pl.show()

# fig = pl.figure()
# pl.axis('equal')
# pl.title('>> Theta Delta (shuffled), # ' + str(td_centroids.shape[0]))
# pl.scatter(points[:, 0], points[:, 1], alpha=0.04)
# pl.plot(centroids[:, 0], centroids[:, 1], 'bx')
# pl.plot(td_centroids[:, 0], td_centroids[:, 1], 'rx')
# pl.plot(td_deltasxo[:, 0], td_deltasxo[:, 1], 'gx')

# pl.show()




# # NOTE: I have reshuffled here once.
# np.random.shuffle(points)

# # TODO: Concatenate (C + X) and Shuffle X (only X - not C) and TSG on the centroids.


# new_points = np.concatenate((td_centroids, points))

# tsg_centroids2, tsg_labels2, tsg_summs2, tsg_clusters_sizes2 = theta_g1(new_points, theta=theta_value, metric='euclidean', all_out=True)


# plot_centroids_clusters(new_points, centroids=tsg_centroids2, labels=tsg_labels2,
#                         title='>> TSG centroids (concat), # ' + str(tsg_centroids2.shape[0]),
#                         exp_type='hard', gt_centroids=centroids, show=False)





# more_iter = True

# if more_iter:

#     new_points2 = np.concatenate((tsg_centroids2, new_points))

#     tsg_centroids3, tsg_labels3, tsg_summs3, tsg_clusters_sizes3 = theta_g1(new_points2, theta=theta_value, metric='euclidean', all_out=True)


#     plot_centroids_clusters(new_points2, centroids=tsg_centroids3, labels=tsg_labels3,
#                             title='>> TSG centroids (concat), # ' + str(tsg_centroids3.shape[0]),
#                             exp_type='hard', gt_centroids=centroids, show=False)

#     new_points3 = np.concatenate((tsg_centroids3, new_points2))

#     tsg_centroids4, tsg_labels4, tsg_summs4, tsg_clusters_sizes4 = theta_g1(new_points3, theta=theta_value, metric='euclidean', all_out=True)


#     plot_centroids_clusters(new_points3, centroids=tsg_centroids4, labels=tsg_labels4,
#                             title='>> TSG centroids (concat), # ' + str(tsg_centroids4.shape[0]),
#                             exp_type='hard', gt_centroids=centroids, show=True)


tmp = 1
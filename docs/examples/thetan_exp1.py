import numpy as np
from thetan.theta import thetan, thetan_density
from thetan.theta import theta_g1, assign
from thetan.theta import distance_matrix as tdm
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


n_local_samples = 30
cluster_per_dim = 10

blob_dis = 5
cluster_x = 10
cluster_y = 10

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

    local_X = cpos + np.random.normal(0, 1, (n_local_samples, 2))
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

tsg_centroids, tsg_labels, tsg_summs, tsg_clusters_sizes = theta_g1(points, theta=3.6, metric='euclidean', all_out=True)


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
tsg_centroids, tsg_labels, tsg_summs, tsg_clusters_sizes = theta_g1(points, theta=3.6, metric='euclidean', all_out=True)


plot_centroids_clusters(points, centroids=tsg_centroids, labels=tsg_labels,
                        title='>> TSG centroids (shuffled), # ' + str(tsg_centroids.shape[0]),
                        exp_type='hard', gt_centroids=centroids, show=True)

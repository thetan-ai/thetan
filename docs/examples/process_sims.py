from time import time
import numpy as np
import pylab as pl
import pandas as pd
from thetan.berserker import berserker2
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score, fowlkes_mallows_score
from sklearn.cluster import AgglomerativeClustering, DBSCAN, HDBSCAN
from sklearn.cluster import BisectingKMeans, MeanShift
from thetan.theta import distance_matrix as tdm
from thetan.theta import theta_g1 as tgs



def evaluate(ecentroids=None, centroids=None, thr=0.5, X=None, label=None, true_label=None, duration=-1):

    D = np.sqrt(tdm(ecentroids, centroids))
    closest = np.min(D, axis=1)
    count = np.sum(closest < thr)
    ssd_score = count / len(ecentroids)
    
    nmi_score = normalized_mutual_info_score(true_label, label)

    sil_score = silhouette_score(X, label)

    fms_score = fowlkes_mallows_score(true_label, label)

    return ssd_score, nmi_score, sil_score, fms_score, duration



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
                            merged_centroids=None, show=True):

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
    
    pl.savefig('2_' + exp_type + '_' + title + '.png', dpi=300)
    pl.close(fig)


def simulate_data(dist_type='gaussian'):
  
    print('Simulating data')

    if dist_type == 'gaussian':
        X, y, gt_centroids = gaussian_clusters(n_local_samples=500, blob_dis = 10, cluster_x = 30, cluster_y = 10)

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


evaluations = {}
evaluation_thr = 0.1
verbose = True

X, L, gt_centroids = simulate_data('gaussian')

theta_value = 3.6

print('Number of features ', X.shape[0])
print('Number of GT centroids ', gt_centroids.shape[0])

print('Starting Berserker')
t = time()
centroids, labels = berserker2(X, theta_value, all_out=False)
duration = time() - t
print('Duration is ', duration)

print('Starting evaluation')
t = time()
scores = evaluate(ecentroids=centroids, centroids=gt_centroids, thr=evaluation_thr, X=X, label=labels, true_label=L, duration=duration)
print(scores)
print('Duration is ', time()-t)
evaluations['TB'] = scores


if verbose:
    plot_centroids_clusters(X, centroids, labels, 'TB', exp_type='2D', initial_centroids=gt_centroids, show=False)


n_clusters = gt_centroids.shape[0]

print('Starting K-means++ ...')
t = time()
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=1,
                max_iter=300, tol=0.0001).fit(X)
duration = time() - t
print('Duration', duration)

kcentroids = kmeans.cluster_centers_
klabels = kmeans.labels_

print('Starting evaluation')
t = time()
scores = evaluate(ecentroids=kcentroids, centroids=gt_centroids, thr=evaluation_thr, X=X, label=klabels, true_label=L, duration=duration)
print(scores)
print('Duration is ', time()-t)
evaluations['kmeans++'] = scores
if verbose:
    plot_centroids_clusters(X, kcentroids, klabels, 'kmeans++', '-', initial_centroids=gt_centroids, show=False)


agglo = False # Due to large memory requirements - it kills my computer

if agglo:

    memory_dir = r"C:\\Users\\Eleftherios\\Devel\\data\\my_cached_memory_folder" 
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="average", memory="cachedir")
    ac = model.fit(X)
    agg_centroids = np.array([np.mean(X[model.labels_==l], axis=0) for l in range(model.n_clusters)])
    agg_labels = model.labels_

    if verbose:
        plot_centroids_clusters(X, agg_centroids, agg_labels, 'aggro', initial_centroids=centroids)



print('Starting with MeanShift ...')
t = time()
bandwidth = theta_value
model = MeanShift(bandwidth=bandwidth, cluster_all=True)
ms = model.fit(X)
duration = time() - t
print('Duration', duration)

ms_centroids = model.cluster_centers_ # np.array([np.mean(X[model.labels_==l], axis=0) for l in range(model.n_clusters)])
ms_labels = model.labels_


print('Starting evaluation')
t = time()
scores = evaluate(ecentroids=ms_centroids, centroids=gt_centroids, thr=evaluation_thr, X=X, label=ms_labels, true_label=L, duration=duration)
print(scores)
print('Duration is ', time()-t)
evaluations['meanshift'] = scores

if verbose:
    plot_centroids_clusters(X, ms_centroids, ms_labels, 'meanshift', '-', initial_centroids=gt_centroids, show=False)


print('Starting EM_GMM .....')
t = time()
cov_type = ['spherical', 'diag', 'tied', 'full']
model = GaussianMixture(n_components=n_clusters,
                        covariance_type=cov_type[3],
                        max_iter=200, n_init=5, random_state=0)
gmm = model.fit(X)
duration = time() - t
print('Duration ', duration)

gmm_centroids = gmm.means_
gmm_labels = gmm.predict(X)

print('Starting evaluation')
t = time()
scores = evaluate(ecentroids=gmm_centroids, centroids=gt_centroids, thr=evaluation_thr, X=X, label=gmm_labels, true_label=L, duration=duration)
print(scores)
print('Duration is ', time() - t)
evaluations['EM_GMM'] = scores

if verbose:
    plot_centroids_clusters(X, gmm_centroids, gmm_labels, 'gmm', '_', initial_centroids=gt_centroids, show=False)


print('Starting TSG')
t = time()
tgs_centroids, tgs_labels = tgs(X, theta_value, all_out=False)
duration = time() - t
print('Duration is ', duration)

print('Starting evaluation')
t = time()
scores = evaluate(ecentroids=tgs_centroids, centroids=gt_centroids, thr=evaluation_thr, X=X, label=tgs_labels, true_label=L, duration=duration)
print(scores)
print('Duration is ', time() - t)
evaluations['TSG'] = scores

if verbose:
    plot_centroids_clusters(X, tgs_centroids, tgs_labels, 'TGS', '_', initial_centroids=gt_centroids, show=False)


print('Starting Bisection K-means++ ...')
t = time()
bkmeans = BisectingKMeans(n_clusters=n_clusters, n_init=1).fit(X)
duration = time() - t
print('Duration', duration)

bkcentroids = bkmeans.cluster_centers_
bklabels = bkmeans.labels_

print('Starting evaluation')
t = time()
scores = evaluate(ecentroids=bkcentroids, centroids=gt_centroids, thr=evaluation_thr, X=X, label=bklabels, true_label=L, duration=duration)
print(scores)
print('Duration is ', time()-t)
evaluations['bisecting'] = scores
if verbose:
    plot_centroids_clusters(X, bkcentroids, bklabels, 'bisecting', '-', initial_centroids=gt_centroids, show=False)


eps = theta_value / 2.
min_samples = 20

print('Starting DBSCAN ...')
t = time()
model = DBSCAN(eps=eps, min_samples=min_samples)
model.fit(X)
duration = time() - t
print('Duration', duration)

unq = np.unique(model.labels_)
db_nclusters = np.sum(unq>=0)

dbcentroids = np.array([np.mean(X[model.labels_==l], axis=0) for l in range(db_nclusters)])
dblabels = model.labels_


print('Starting evaluation')
t = time()
scores = evaluate(ecentroids=dbcentroids, centroids=gt_centroids, thr=evaluation_thr, X=X, label=dblabels, true_label=L, duration=duration)
print(scores)
print('Duration is ', time()-t)
evaluations['dbscan'] = scores
if verbose:
    plot_centroids_clusters(X, dbcentroids, dblabels, 'dbscan', '-', initial_centroids=gt_centroids, show=False)


print('Starting HDBSCAN ...')
t = time()
model = HDBSCAN(min_samples=min_samples)
model.fit(X)
duration = time() - t
print('Duration', duration)

unq = np.unique(model.labels_)
hdb_nclusters = np.sum(unq>=0)

hdbcentroids = np.array([np.mean(X[model.labels_==l], axis=0) for l in range(hdb_nclusters)])
hdblabels = model.labels_

print('Starting evaluation')
t = time()
scores = evaluate(ecentroids=hdbcentroids, centroids=gt_centroids, thr=evaluation_thr, X=X, label=hdblabels, true_label=L, duration=duration)
print(scores)
print('Duration is ', time()-t)
evaluations['dbscan'] = scores
if verbose:
    plot_centroids_clusters(X, hdbcentroids, hdblabels, 'hdbscan', '-', initial_centroids=gt_centroids, show=False)



print('ALL EVALUATIONS')
print(evaluations)

evals = np.zeros((8, 5))

rows = ['TB', 'kmeans++', 'meanshift', 'EM_GMM', 'TSG', 'bisecting', 'dbscan', 'hdbscan']
cols = ['SSD', 'NMI', 'SIL', 'FMS', 'Duration']

for i, row in enumerate(rows):
    for j, col in enumerate(cols):
        evals[i, j] = evaluations[row][j]

df = pd.DataFrame(evals, colums=cols, rows=rows)

df.to_excel("evaluations" + '_' + str(theta_value) + ".xlsx")  

print('Saving evaluations')




t = 1

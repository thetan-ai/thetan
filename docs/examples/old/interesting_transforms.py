from time import time
import numpy as np
import string
import os
from numpy import testing as npt
from thetan.theta import theta_g1, assign, gravity, calibrate, theta_g1t
import matplotlib.pyplot as pl
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
# from sklearn.metrics.pairwise import _euclidean_distances
from sklearn.utils.extmath import row_norms, stable_cumsum, safe_sparse_dot
from sklearn.utils import check_array
from sklearn.utils import check_random_state


dir_path = os.path.dirname(os.path.realpath(__file__))
dir_path = os.path.join(dir_path, 'S-sets')


def _euclidean_distances(X, Y, X_norm_squared=None, Y_norm_squared=None,
                         squared=False):
    """Computational part of euclidean_distances
    Assumes inputs are already checked.
    If norms are passed as float32, they are unused. If arrays are passed as
    float32, norms needs to be recomputed on upcast chunks.
    TODO: use a float64 accumulator in row_norms to avoid the latter.
    """
    if X_norm_squared is not None:
        if X_norm_squared.dtype == np.float32:
            XX = None
        else:
            XX = X_norm_squared.reshape(-1, 1)
    elif X.dtype == np.float32:
        XX = None
    else:
        XX = row_norms(X, squared=True)[:, np.newaxis]

    if Y is X:
        YY = None if XX is None else XX.T
    else:
        if Y_norm_squared is not None:
            if Y_norm_squared.dtype == np.float32:
                YY = None
            else:
                YY = Y_norm_squared.reshape(1, -1)
        elif Y.dtype == np.float32:
            YY = None
        else:
            YY = row_norms(Y, squared=True)[np.newaxis, :]

    # if X.dtype == np.float32:
    #     # To minimize precision issues with float32, we compute the distance
    #     # matrix on chunks of X and Y upcast to float64
    #     distances = _euclidean_distances_upcast(X, XX, Y, YY)
    # else:
    # if dtype is already float64, no need to chunk and upcast
    distances = - 2 * safe_sparse_dot(X, Y.T, dense_output=True)
    distances += XX
    distances += YY
    np.maximum(distances, 0, out=distances)

    # Ensure that distances between vectors and themselves are set to 0.0.
    # This may not be the case due to floating point rounding errors.
    if X is Y:
        np.fill_diagonal(distances, 0)

    return distances if squared else np.sqrt(distances, out=distances)



def kmeans_plusplus(X, n_clusters, *, x_squared_norms=None,
                    random_state=None, n_local_trials=None):
    """Init n_clusters seeds according to k-means++
    .. versionadded:: 0.24
    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds from.
    n_clusters : int
        The number of centroids to initialize
    x_squared_norms : array-like of shape (n_samples,), default=None
        Squared Euclidean norm of each data point.
    random_state : int or RandomState instance, default=None
        Determines random number generation for centroid initialization. Pass
        an int for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.
    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)).
    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The inital centers for k-means.
    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.
    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007
    Examples
    --------
    >>> from sklearn.cluster import kmeans_plusplus
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> centers, indices = kmeans_plusplus(X, n_clusters=2, random_state=0)
    >>> centers
    array([[10,  4],
           [ 1,  0]])
    >>> indices
    array([4, 2])
    """

    # Check data
    check_array(X, accept_sparse='csr',
                dtype=[np.float64, np.float32])

    if X.shape[0] < n_clusters:
        raise ValueError(f"n_samples={X.shape[0]} should be >= "
                         f"n_clusters={n_clusters}.")

    # Check parameters
    if x_squared_norms is None:
        x_squared_norms = row_norms(X, squared=True)
    else:
        x_squared_norms = check_array(x_squared_norms,
                                      dtype=X.dtype,
                                      ensure_2d=False)

    if x_squared_norms.shape[0] != X.shape[0]:
        raise ValueError(
            f"The length of x_squared_norms {x_squared_norms.shape[0]} should "
            f"be equal to the length of n_samples {X.shape[0]}.")

    if n_local_trials is not None and n_local_trials < 1:
        raise ValueError(
            f"n_local_trials is set to {n_local_trials} but should be an "
            f"integer value greater than zero.")

    random_state = check_random_state(random_state)

    # Call private k-means++
    centers, indices = _kmeans_plusplus(X, n_clusters, x_squared_norms,
                                        random_state, n_local_trials)

    return centers, indices


def _kmeans_plusplus(X, n_clusters, x_squared_norms,
                     random_state, n_local_trials=None):
    """Computational component for initialization of n_clusters by
    k-means++. Prior validation of data is assumed.
    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds for.
    n_clusters : int
        The number of seeds to choose.
    x_squared_norms : ndarray of shape (n_samples,)
        Squared Euclidean norm of each data point.
    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.
    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.
    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        The inital centers for k-means.
    indices : ndarray of shape (n_clusters,)
        The index location of the chosen centers in the data array X. For a
        given index and center, X[index] = center.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and track index of point
    center_id = random_state.randint(n_samples)
    indices = np.full(n_clusters, -1, dtype=int)
    # if sp.issparse(X):
    #     centers[0] = X[center_id].toarray()
    # else:
    centers[0] = X[center_id]
    indices[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = _euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = _euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates,
                   out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        # if sp.issparse(X):
        #     centers[c] = X[best_candidate].toarray()
        # else:
        centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return centers, indices



def labels_to_centroids(data, labels):

    unique_labels, unique_counts = np.unique(labels,
                                             return_counts=True)

    centroids = np.zeros((len(unique_labels), data.shape[1]))

    for (i, (u, c)) in enumerate(zip(unique_labels, unique_counts)):
        centroids[i] = np.sum(data[labels == u], axis=0) / c

    return centroids


def grid_of_gaussians(n_local_samples, blob_dis=5):
    cluster_x = 10
    cluster_y = 10
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

    gt_centroids = np.array(gt_centroids)

    return X, gt_centroids


def random_md_gaussians(n_local_samples=250, rcenters=100, scenters=10, scov=1, ndim=10):

    pos = scenters * (np.random.rand(rcenters, ndim) - 0.5)
    gt_centroids = []
    X = np.zeros((n_local_samples * pos.shape[0], ndim))
    y = np.zeros((n_local_samples * pos.shape[0]), dtype=np.int64)

    for i in range(pos.shape[0]):
        cpos = pos[i]
        mnormal = np.random.multivariate_normal(np.zeros(ndim), scov*np.eye(ndim),
                                                size=n_local_samples)
        local_X = cpos + mnormal
        X[i * n_local_samples: (i + 1) * n_local_samples] = local_X
        y[i * n_local_samples: (i + 1) * n_local_samples] = i
        gt_centroids.append(local_X.mean(axis=0))

    gt_centroids = np.array(gt_centroids)

    return X, gt_centroids


def show_clusters(X, method, gt_centroids, labels, centroids):
    pl.figure()
    pl.title('Method is ' + method)
    # unique_colors = np.random.rand(np.unique(labels).shape[0], 3)
    # Tricky point here if there if some labels are missing
    unique_colors = np.random.rand(labels.max() + 1, 3)

    colors = np.zeros((labels.shape[0], 3), dtype='f8')
    for i in range(X.shape[0]):
        colors[i] = unique_colors[labels[i]]
    pl.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.5)
    pl.plot(gt_centroids[:, 0], gt_centroids[:, 1], 'ko')
    if centroids is not None:
        pl.plot(centroids[:, 0], centroids[:, 1], 'b^')
    # pl.legend()
    # pl.xlim([0 , 1000000])
    # pl.savefig(os.path.join(dir_path,'s1_compare_GT_TethA_KM++.png'), dpi=300)
    # pl.show()


def show_points(X, title='initial'):
    pl.figure()
    pl.axis('equal')
    pl.title(title)
    pl.scatter(X[:, 0], X[:, 1], alpha=0.1)


def show_points_two(X, Y, title='overlap'):
    pl.figure()
    pl.axis('equal')
    pl.title(title)
    pl.scatter(X[:, 0], X[:, 1], color='b', alpha=0.1)
    pl.scatter(Y[:, 0], Y[:, 1], color='g', alpha=0.1)


# X = np.loadtxt(os.path.join(dir_path, 's4.txt'))
# gt_c = np.loadtxt(os.path.join(dir_path, 's4-cb.txt'))

X, gt_c = grid_of_gaussians(250, blob_dis=5)

kppcenters, kppindices = kmeans_plusplus(X, n_clusters=100, random_state=0)

kpplabels = assign(X, kppcenters, metric='euclidean')

show_clusters(X, 'kpp', gt_c, kpplabels, kppcenters)

show_c = True

X, gt_c = random_md_gaussians(n_local_samples=250,
                              rcenters=100,
                              scenters=60,
                              scov=40, ndim=80)

print(X.shape)

np.random.shuffle(X)

X2 = X.copy()

# X = X[:int(0.1 * len(X))]

# X = np.ascontiguousarray(X)

# show_points(X)

theta = 4 * 5 * 4 * 2

# theta = 5 * 10 * 2
# theta = 100000
print('starting')
centroids, labels, summs, clusters_sizes, dtravel = theta_g1t(X, theta=theta,
                                                              metric='euclidean',
                                                              all_out=True)
print('done')
if show_c:
    show_clusters(X, 'TSG', gt_c, labels, centroids)

# final_labels = gravity(X, centroids, clusters_sizes, theta/2., metric='euclidean')

def labels_to_centroids(data, labels):

    unique_labels, unique_counts = np.unique(labels,
                                             return_counts=True)

    centroids = np.zeros((len(unique_labels), data.shape[1]))

    for (i, (u, c)) in enumerate(zip(unique_labels, unique_counts)):
        centroids[i] = np.sum(data[labels == u], axis=0) / c

    return centroids

# final_centroids = labels_to_centroids(X, final_labels)
# show_clusters(X, 'TSG-G', gt_c, final_labels, None, final_centroids)


def cluster_distribution(X, centroids, labels, clusters_sizes):

    distr = {}
    sumd = []
    sigd = []

    for c in range(centroids.shape[0]):
        distr[c] = []

    for i, l in enumerate(labels):
        distr[l].append(np.linalg.norm(X[i] - centroids[labels[i]]))

    pl.figure()
    pl.title('distr')
    max_length = 0
    for c in range(centroids.shape[0]):
        distr[c] = np.sort(distr[c])[::-1]
        current_length = len(distr[c])
        if current_length > max_length:
            max_length = current_length

    fdistr = np.zeros((centroids.shape[0], max_length), 'f8')
    integrals = []

    for c in range(centroids.shape[0]):
        tmp = np.zeros(max_length, 'f8')
        current_length = len(distr[c])
        tmp[:current_length] = distr[c]
        sumd.append(np.mean(distr[c]))
        sigd.append(np.std(distr[c]))
        distr[c] = tmp.copy()
        fdistr[c] = tmp.copy()
        integrals.append(np.sum(tmp))
        pl.plot(distr[c], alpha=0.2)
        # print(c, distr)

    pl.figure()
    pl.title('integrals')
    pl.plot(integrals, '.')

    return distr, sumd, fdistr, integrals


def cluster_distribution_hist(X, centroids, labels, theta):

    distr = {}

    for c in range(centroids.shape[0]):
        distr[c] = []

    for i, l in enumerate(labels):
        distr[l].append(np.linalg.norm(X[i] - centroids[l]))

    for c in range(centroids.shape[0]):
        distr[c], b = np.histogram(distr[c], bins=50, range=(0, 2*theta))

    # pl.figure()
    # pl.title('distr - plot')
    arr = np.zeros((len(distr), distr[0].shape[0]))
    proj = np.zeros(arr.shape[0])
    tmp = np.log(np.arange(arr.shape[1]) + 2)[::-1]
    for c in range(centroids.shape[0]):
        arr[c] = distr[c]
        proj[c] = np.sum(arr[c] * tmp)
    #     pl.plot(arr[c], label=c, alpha=0.5)
    # pl.legend()

    pl.figure()
    pl.title('distr - image')
    pl.imshow(arr)

    pl.figure()
    pl.title('projections')
    pl.plot(proj, '.')

    pl.figure()
    pl.title('sorted projections')
    pl.grid()
    pl.plot(np.sort(proj)[::-1], '.')

    return distr, arr, proj


# distr, sumd, fdistr, integrals = cluster_distribution(X, centroids, labels, clusters_sizes)

distrh, arrh, projh = cluster_distribution_hist(X, centroids, labels, theta)

labels_proj = np.zeros(X.shape[0], dtype='i8')
res_bool = projh > 600
for i, l in enumerate(labels):

    if res_bool[l]:
        labels_proj[i] = 0
    else:
        labels_proj[i] = 1

if show_c:
    show_clusters(X, 'projection', gt_c, labels_proj, centroids)

pl.figure()
pl.title('Sizes of clusters')
pl.plot(clusters_sizes, '.')

pl.figure()
pl.title('Sorted sizes of clusters')
pl.grid()
pl.plot(np.sort(clusters_sizes)[::-1], '.')


pl.figure()
pl.title('dtravel')
pl.grid()
pl.plot(dtravel, '.')

pl.figure()
pl.title('Sorted dtravel')
pl.grid()
pl.plot(np.sort(dtravel)[::-1], '.')

pl.figure()
pl.title('dtravel normalized')
pl.grid()
pl.plot(dtravel/clusters_sizes, '.')


if show_c:
    from sklearn.cluster import DBSCAN
    db = DBSCAN(eps=.8, min_samples=10).fit(X)
    show_clusters(X, 'dbscan', gt_c, db.labels_, None)

# theta2 = theta

# centroids2, labels2, summs2, clusters_sizes2 = theta_g1(fdistr, theta=theta2,
#                                                         metric='euclidean',
#                                                         all_out=True)
# print('Centroids2',  centroids2.shape)
# print('Fdistr',  fdistr.shape)

# integrals = np.array(integrals)[:, None]

# theta3 = np.mean(integrals) / 4

# centroids3, labels3, summs3, clusters_sizes3 = theta_g1(integrals, theta=theta3,
#                                                         metric='euclidean',
#                                                         all_out=True)
# print('Centroids3',  centroids3.shape)
# print('Labels3',  labels3.shape)
# print('Clusters_sizes3',  clusters_sizes3.shape)

# small_clusters_indices = np.where(clusters_sizes3 < np.mean(clusters_sizes3) / 2)[0]

# flagged = np.zeros(labels.shape, 'i8')
# for si in small_clusters_indices:
#     flagged[labels == si] = 1

# final_labels = calibrate(X, centroids, labels,
#                          flagged, metric='euclidean')

# final_centroids = labels_to_centroids(X, final_labels)

# show_clusters(X, 'TSG', gt_c, final_labels, None, final_centroids)

pl.show()

from scipy.spatial import distance_matrix

D = distance_matrix(centroids, gt_c)

minD = np.min(D, axis=1)
print(np.max(minD))


bkp = 1

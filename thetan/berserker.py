from time import time
import numpy as np
from patchify import patchify
from scipy.ndimage import gaussian_gradient_magnitude, label, distance_transform_edt, binary_fill_holes
from scipy.ndimage import binary_erosion, binary_dilation, binary_opening, binary_closing, laplace, generic_gradient_magnitude
from scipy.ndimage import generate_binary_structure, gaussian_filter, center_of_mass, median_filter, gaussian_laplace
from thetan.theta import theta_g1, reassign_uncertain_update_labels, centroids_from_labels
from thetan.theta import theta_means, candidate_labels_c, candidate_labels_outliers_c
from thetan.theta import img3d_to_ijkv_features, img2d_to_ijv_features, assign, distance_matrix
from skimage.segmentation import flood_fill
from skimage.morphology import reconstruction


def check_arr(a):
    print(a.shape, a.dtype, a.min(), a.max())

def check_module_load():

    print('yep')


def candidate_labels(centroids, labels, initial_labels):
    """ Find labels for removal in cases of conflict
    """
    nice = np.ones(initial_labels.shape[0], dtype=initial_labels.dtype)
    updated_labels = initial_labels.copy()
    clusters = np.empty((centroids.shape[0], 0)).tolist()
    for i in range(labels.shape[0]):
        clusters[labels[i]].append(i)

    candidates = []
    # for c in clusters:
    #     if len(c) > 1:
    #         maxc = np.max(c)
    #         candidates.append(maxc)

    for c in clusters:
        if len(c) > 1:
            cs = np.sort(c)
            candidates.extend(cs[1:])

    # caution here, deleting will change the labels of other Xs too which is not ideal
    # centroids_reduced = np.delete(centroids, candidates, axis=0)

    for i, l in enumerate(initial_labels):
        for can in candidates:
            if l == can:
                nice[i] = 0
                updated_labels[i] = -1

    for i, l in enumerate(initial_labels):
        for can in candidates:
            if l > can:
                updated_labels[i] -= 1


    return nice, candidates, clusters, updated_labels


def candidate_labels_outliers(centroids, candidates, initial_labels):
    """ Find labels for removal in cases of outliers
    """
    nice = np.ones(initial_labels.shape[0], dtype=initial_labels.dtype)
    updated_labels = initial_labels.copy()

    # caution here, deleting will change the labels of other Xs too which is not ideal
    # centroids_reduced = np.delete(centroids, candidates, axis=0)

    for i, l in enumerate(initial_labels):
        for can in candidates:
            if l == can:
                nice[i] = 0
                updated_labels[i] = -1

    # The above can be written in this way too
    # can_mask = np.in1d(initial_labels, candidates)
    # nice[can_mask] = 0
    # updated_labels[can_mask] = -1

    for i, l in enumerate(initial_labels):
        for can in candidates:
            if l > can:
                updated_labels[i] -= 1

    return nice, updated_labels


def labels_to_centroids(data, labels):

    unique_labels, unique_counts = np.unique(labels,
                                             return_counts=True)

    centroids = np.zeros((len(unique_labels), data.shape[1]))

    for (i, (u, c)) in enumerate(zip(unique_labels, unique_counts)):
        centroids[i] = np.sum(data[labels == u], axis=0) / c

    return centroids


def tsg_randomized(X, theta, metric='euclidean', num_shuffles=10):
    sum_of_time = 0
    all_centroids = {}
    X_work = X.copy()

    num_centroids = []
    timings = []
    for s in range(num_shuffles):

        np.random.shuffle(X_work)
        X_work_cp = np.ascontiguousarray(X_work)
        t1 = time()
        centroids, _ = theta_g1(X_work_cp, theta=theta, metric=metric)
        t2 = time()
        timings.append(t2-t1)

        all_centroids[s] = centroids
        num_centroids.append(len(centroids))

    print('Time for tg1 calls', np.sum(timings))
    sum_of_time += np.sum(timings)
    total_num_centroids = np.sum(num_centroids)
    X_centroids = np.zeros((total_num_centroids, X_work.shape[1]), dtype='f8')

    offset = 0

    for s in range(num_shuffles):

        X_centroids[offset:offset + all_centroids[s].shape[0]] = all_centroids[s]
        offset += all_centroids[s].shape[0]

    t1 = time()
    new_centroids, _ = theta_g1(X_centroids, theta=theta, metric=metric)
    t2 = time()
    sum_of_time += t2 - t1
    print('Time for tg1 on centroids', t2 - t1)


    t1 = time()
    final_indices = assign(X, new_centroids, metric=metric)
    t2 = time()
    sum_of_time += t2 - t1
    print('Time for assign calls', t2 - t1)
    final_centroids = labels_to_centroids(X, final_indices)

    return final_centroids, final_indices, sum_of_time


def berserker(points, theta_value, metric='euclidean', all_out=True, verbose=True,
              manage_outliers={'mode':'min_size', 'size_limit':60}, quality_over_speed=False):


    info_dict = {}
    t0 = time()

    np.random.shuffle(points)

    t1 = time()
    info_dict['Shuffle 1 X'] = t1 - t0

    initial_centroids, initial_labels, _, _ = theta_g1(points, theta=theta_value, metric=metric,
                                                       all_out=True)

    t2 = time()

    info_dict['Shuffle 1 TG1 X'] = t2 - t1

    print('ic', initial_centroids.shape)

    t3 = time()
    c_centroids, c_labels, _, _ = theta_g1(initial_centroids, theta=theta_value, metric=metric,
                                           all_out=True)

    t4 = time()

    info_dict['Shuffle 1 TG1 C'] = t4 - t3

    t5 = time()

    # nice, candidates, _clusters, updated_labels = candidate_labels(c_centroids, c_labels, initial_labels)
    nice, candidates, _clusters, updated_labels = candidate_labels_c(c_centroids, c_labels, initial_labels)

    t6 = time()

    info_dict['Shuffle 1 CL'] = t6 - t5

    even_centroids = np.delete(initial_centroids, candidates, axis=0)
    print('ec', even_centroids.shape)

    t7 = time()

    info_dict['Shuffle 1 DEL'] = t7 - t6


    updated_labels2 = reassign_uncertain_update_labels(points, even_centroids,
                                                       updated_labels, nice,
                                                       metric=metric)

    t8 = time()

    info_dict['Shuffle 1 REA X'] = t8 - t7

    even_centroids_ex = centroids_from_labels(points, updated_labels2)

    t9 = time()

    info_dict['Shuffle 1 CEN_LA X'] = t9 - t8


    # This second shuffle resolves many problems with centroids appearing between clusters
    np.random.shuffle(points)

    t10 = time()

    info_dict['Shuffle 2 X'] = t10 - t9

    p_centroids, p_labels, _, _ = theta_means(points, even_centroids_ex, theta=theta_value,
                                              metric=metric,
                                              all_out=True)

    t11 = time()
    print('p_centroids', p_centroids.shape)

    info_dict['Shuffle 2 TM X'] = t11 - t10


    t12 = time()
    c_centroids2, c_labels2, _, _ = theta_g1(p_centroids, theta=theta_value,
                                             metric=metric, all_out=True)

    t13 = time()


    info_dict['Shuffle 2 TG1 PC'] = t13 - t12

    # nice2, candidates2, _clusters2, updated_labels2x = candidate_labels(c_centroids2, c_labels2, p_labels)
    nice2, candidates2, _clusters2, updated_labels2x = candidate_labels_c(c_centroids2, c_labels2, p_labels)

    t14 = time()

    info_dict['Shuffle 2 CL'] = t14 - t13


    even_centroids2 = np.delete(p_centroids, candidates2, axis=0)

    t15 = time()
    print('ec2 ', even_centroids2.shape)

    info_dict['Shuffle 2 DEL'] = t15 - t14

    updated_labels_ex2 = reassign_uncertain_update_labels(points, even_centroids2,
                                                          updated_labels2x, nice2,
                                                          metric=metric)
    t16 = time()

    info_dict['Shuffle 2 REA X'] = t16 - t15

    even_centroids_ex2 = centroids_from_labels(points, updated_labels_ex2)
    t17 = time()
    print('ec2x ', even_centroids_ex2.shape)

    info_dict['Shuffle 2 CEN_LA'] = t17 - t16


    if quality_over_speed:

        print('Quality over speed enabled')

        print('------------------')

        for i in range(3):

            np.random.shuffle(points)

            t10 = time()

            info_dict['Shuffle ' + str(3 + i)  + ' X'] = t10 - t9

            p_centroids, p_labels, _, _ = theta_means(points, even_centroids_ex2, theta=theta_value,
                                                    metric=metric,
                                                    all_out=True)

            t11 = time()
            print('p_centroids', p_centroids.shape)

            info_dict['Shuffle ' + str(3 + i)  + ' TM X'] = t11 - t10


            t12 = time()
            c_centroids2, c_labels2, _, _ = theta_g1(p_centroids, theta=theta_value,
                                                    metric=metric, all_out=True)

            t13 = time()


            info_dict['Shuffle ' + str(3 + i)  + ' TG1 PC'] = t13 - t12

            # nice2, candidates2, _clusters2, updated_labels2x = candidate_labels(c_centroids2, c_labels2, p_labels)
            nice2, candidates2, _clusters2, updated_labels2x = candidate_labels_c(c_centroids2, c_labels2, p_labels)

            t14 = time()

            info_dict['Shuffle ' + str(3 + i)  + ' CL'] = t14 - t13


            even_centroids2 = np.delete(p_centroids, candidates2, axis=0)

            t15 = time()
            print('ec2 ', even_centroids2.shape)

            info_dict['Shuffle ' + str(3 + i)  + ' DEL'] = t15 - t14

            updated_labels_ex2 = reassign_uncertain_update_labels(points, even_centroids2,
                                                                updated_labels2x, nice2,
                                                                metric=metric)
            t16 = time()

            info_dict['Shuffle ' + str(3 + i)  + ' REA X'] = t16 - t15

            even_centroids_ex2 = centroids_from_labels(points, updated_labels_ex2)
            t17 = time()
            print('ec2x ', even_centroids_ex2.shape)

            info_dict['Shuffle ' + str(3 + i)  + ' CEN_LA'] = t17 - t16


            if manage_outliers['mode'] == 'min_size':

                unique_labels_ex2, unique_counts_ex2 = np.unique(updated_labels_ex2,
                                                                return_counts=True)

                candidates3 = unique_labels_ex2[unique_counts_ex2 <= manage_outliers['size_limit']]

                # nice3, updated_labels3 = candidate_labels_outliers(even_centroids_ex2, candidates3, updated_labels_ex2)
                nice3, updated_labels3 = candidate_labels_outliers_c(even_centroids_ex2, candidates3, updated_labels_ex2)

                even_centroids_ex3 = np.delete(even_centroids_ex2, candidates3, axis=0)

                updated_labels_ex4 = reassign_uncertain_update_labels(points, even_centroids_ex3, updated_labels3, nice3,
                                                                    metric=metric)

                even_centroids_ex4 = centroids_from_labels(points, updated_labels_ex4)

                even_centroids_ex2 = even_centroids_ex4

                updated_labels_ex2 = updated_labels_ex4

                t18 = time()

                info_dict['Manage Outliers ' + str(i)] = t18 - t17

                print('Final Shape:', even_centroids_ex2.shape)


    else:

        if manage_outliers['mode'] == 'min_size':

            unique_labels_ex2, unique_counts_ex2 = np.unique(updated_labels_ex2,
                                                            return_counts=True)

            candidates3 = unique_labels_ex2[unique_counts_ex2 <= manage_outliers['size_limit']]

            # nice3, updated_labels3 = candidate_labels_outliers(even_centroids_ex2, candidates3, updated_labels_ex2)
            nice3, updated_labels3 = candidate_labels_outliers_c(even_centroids_ex2, candidates3, updated_labels_ex2)

            even_centroids_ex3 = np.delete(even_centroids_ex2, candidates3, axis=0)

            updated_labels_ex4 = reassign_uncertain_update_labels(points, even_centroids_ex3, updated_labels3, nice3,
                                                                metric=metric)

            even_centroids_ex4 = centroids_from_labels(points, updated_labels_ex4)

            even_centroids_ex2 = even_centroids_ex4

            updated_labels_ex2 = updated_labels_ex4

            t18 = time()

            info_dict['Manage Outliers '] = t18 - t17

            print('Final Shape:', even_centroids_ex2.shape)



    print('Times')
    print(info_dict)
    t = 1

    if all_out:
        return even_centroids_ex2, updated_labels_ex2, initial_centroids, even_centroids, even_centroids_ex, p_centroids, p_labels, even_centroids2

    else:
        return even_centroids_ex2, updated_labels_ex2


def berserker2(points, theta_value, metric='euclidean', all_out=True, verbose=True):

    info_dict = {}
    t0 = time()

    # np.random.shuffle(points)

    ind = np.random.choice(np.arange(points.shape[0]), points.shape[0], replace=False)
    points2 = points[ind].copy()
    ind_inv = np.argsort(ind)


    t1 = time()
    info_dict['Shuffle 1 X'] = t1 - t0

    initial_centroids, initial_labels, _, _ = theta_g1(points2, theta=theta_value, metric=metric,
                                                       all_out=True)

    t2 = time()

    info_dict['Shuffle 1 TG1 X'] = t2 - t1

    if verbose:
        print('ic', initial_centroids.shape)

    t3 = time()
    c_centroids, c_labels, _, _ = theta_g1(initial_centroids, theta=theta_value, metric=metric,
                                           all_out=True)

    t4 = time()

    info_dict['Shuffle 1 TG1 C'] = t4 - t3

    t5 = time()

    nice, candidates, _clusters, updated_labels = candidate_labels_c(c_centroids, c_labels, initial_labels)

    t6 = time()

    info_dict['Shuffle 1 CL'] = t6 - t5

    even_centroids = np.delete(initial_centroids, candidates, axis=0)
    if verbose:
        print('ec', even_centroids.shape)

    t7 = time()

    info_dict['Shuffle 1 DEL'] = t7 - t6

    updated_labels2 = reassign_uncertain_update_labels(points2, even_centroids,
                                                       updated_labels, nice,
                                                       metric=metric)

    t8 = time()

    info_dict['Shuffle 1 REA X'] = t8 - t7

    even_centroids_ex = centroids_from_labels(points2, updated_labels2)

    t9 = time()

    info_dict['Shuffle 1 CEN_LA X'] = t9 - t8


    # This second shuffle resolves many problems with centroids appearing between clusters
    # np.random.shuffle(points)
    ind2 = np.random.choice(np.arange(points.shape[0]), points.shape[0], replace=False)
    points3 = points[ind2].copy()
    ind2_inv = np.argsort(ind2)

    t10 = time()

    info_dict['Shuffle 2 X'] = t10 - t9

    p_centroids, p_labels, _, _ = theta_means(points3, even_centroids_ex, theta=theta_value,
                                              metric=metric,
                                              all_out=True)

    t11 = time()
    if verbose:
        print('p_centroids', p_centroids.shape)

    info_dict['Shuffle 2 TM X'] = t11 - t10


    t12 = time()
    c_centroids2, c_labels2, _, _ = theta_g1(p_centroids, theta=theta_value,
                                             metric=metric, all_out=True)

    t13 = time()


    info_dict['Shuffle 2 TG1 PC'] = t13 - t12

    # nice2, candidates2, _clusters2, updated_labels2x = candidate_labels(c_centroids2, c_labels2, p_labels)
    nice2, candidates2, _clusters2, updated_labels2x = candidate_labels_c(c_centroids2, c_labels2, p_labels)

    t14 = time()

    info_dict['Shuffle 2 CL'] = t14 - t13

    even_centroids2 = np.delete(p_centroids, candidates2, axis=0)

    t15 = time()
    if verbose:
        print('ec2 ', even_centroids2.shape)

    info_dict['Shuffle 2 DEL'] = t15 - t14

    updated_labels_ex2 = reassign_uncertain_update_labels(points3, even_centroids2,
                                                          updated_labels2x, nice2,
                                                          metric=metric)
    t16 = time()

    info_dict['Shuffle 2 REA X'] = t16 - t15

    # bring labels back from original points (not points3)
    updated_labels_ex2 = updated_labels_ex2[ind2_inv].copy()
    even_centroids_ex2 = centroids_from_labels(points, updated_labels_ex2)

    t17 = time()

    if verbose:
        print('ec2x ', even_centroids_ex2.shape)

    info_dict['Shuffle 2 CEN_LA *'] = t17 - t16

    # quality over speed option not available in this version yet
    if verbose:
        print('Times')
        print(info_dict)

    t = 1

    if all_out:
        return even_centroids_ex2, updated_labels_ex2, initial_centroids, even_centroids, even_centroids_ex, p_centroids, p_labels, even_centroids2

    else:
        return even_centroids_ex2, updated_labels_ex2


def berserker2_to_3D_labels(X, theta_value, data_dims, verbose=False, mask=None):
    """
    Deploying berserker v2 in imaging data and
    returning labels as 3D volumes.
    A mask can be provided in 1D form (ravelled)
    """

    if mask is not None:
        Xp = X[mask, :].copy()
        imask = 1 - mask
    else:
        Xp = X

    centroids, labels = berserker2(Xp, theta_value, all_out=False, verbose=verbose)

    if mask is not None:
        # We can calculate the centroids for label -1 on our own
        labelsp = np.zeros(np.prod(data_dims))
        labelsp[imask == 1] = -1
        labelsp[mask == 1] = labels.copy()
        centroid_neg1 = np.mean(X[imask], axis=0)
        centroids = np.vstack((centroid_neg1, centroids))
    else:
        labelsp = labels

    data_labels = labelsp.copy()
    data_labels = data_labels.reshape(data_dims)
    data_labels = data_labels.astype('f8')
    return data_labels, centroids


def X_to_many_3D(X, data_dims):
    """ Currently assuming only 4 features"""

    X0 = X[:, 0].reshape(data_dims)
    X1 = X[:, 1].reshape(data_dims)
    X2 = X[:, 2].reshape(data_dims)
    X3 = X[:, 3].reshape(data_dims)

    return X0, X1, X2, X3


def berserker2_to_2D_labels(X, theta_value, data_dims):

    centroids, labels = berserker2(X, theta_value, all_out=False)

    data_labels = labels.copy()
    data_labels = data_labels.reshape(data_dims)
    data_labels = data_labels.astype('f8')
    return data_labels, centroids


def X_to_many_2D(X, data_dims):
    """ Currently assuming only 4 features"""

    X0 = X[:, 0].reshape(data_dims)
    X1 = X[:, 1].reshape(data_dims)
    X2 = X[:, 2].reshape(data_dims)

    return X0, X1, X2


def prepare_data_2d(data, enh=2.):
    """ Returns feature space X(rows, cols, data) with rows and cols
    coordinates originating at the center of the image."""

    data = np.ascontiguousarray(data)

    ijv = img2d_to_ijv_features(data)

    ci, cj = np.asarray(data.shape) / 2.0

    X = ijv.copy()
    X[:, 0] -= ci
    X[:, 1] -= cj
    X[:, 2] *= enh

    return X


def prepare_data_3d(data, affine, perc=(2, 99), mult=300, enh=3):

    if perc is None:
        data2 = data
    else:
        value_range = np.percentile(data, q=perc)
        data2 = np.interp(data, value_range, (0, 1))
        data2 =  mult * np.ascontiguousarray(data2, dtype='f8')

    X = img3d_to_ijkv_features(data2)
    # move to world coordinates W (xyz)

    W = np.dot(affine[:3, :3], X[:, :3].T) + affine[:3, 3][:, None]
    X[:, :3] = W.T

    # Intensity more important than location
    X[:, 3] = enh * X[:, 3]

    return X


def prepare_data_3d_as_2d_series(data, affine, perc=(2, 99), mult=300, enh=3):

    if perc is None:
        data2 = data
    else:
        value_range = np.percentile(data, q=[2, 99])
        data2 = np.interp(data, value_range, (0, 1))
        data2 =  mult * np.ascontiguousarray(data2, dtype='f8')

    X = img3d_to_ijkv_features(data2)
    # move to world coordinates W (xyz)

    # W = np.dot(affine[:3, :3], X[:, :3].T) + affine[:3, 3][:, None]
    # X[:, :3] = W.T

    # Intensity more important than location
    X[:, 3] = enh * X[:, 3]

    X = np.concatenate([X[..., :2], X[..., -1:]], axis=-1)

    return X


def analyze_labels(data_labels_focused, num_focused, verbose=False, return_sizes=False):

    sizes = []
    for i in range(num_focused):
        cur_sz = (data_labels_focused == i).sum()
        sizes.append(cur_sz)

    sizes = np.asarray(sizes)
    if verbose:
        print(sizes)
    favorites = np.argsort(sizes)[::-1][:3]
    favorite = favorites[1] # 0 most likely bk
    print('Favorite index')
    print(favorite, sizes[favorite])
    if return_sizes:
        return sizes
    return favorite


def mask_lt(data_labels, centroids, bk_thr=300, updated_labels=False):
    """ Remove low intensities using label centroids
    """
    # bk_thr = 300 # 400 or 250 interesting too
    data_labels_u = data_labels.copy()
    indx = np.where(centroids[:, -1] < bk_thr)[0]
    min_indx = np.argmin(centroids[:, -1])

    for indx_ in indx:
        data_labels_u[data_labels == indx_] = min_indx

    mask = data_labels_u == min_indx
    maskb = mask.copy()
    mask = mask.astype('f8')

    if updated_labels:
        return mask, data_labels_u
    return mask


def mask_gt(data_labels, centroids, bk_thr=300, updated_labels=False):
    """ Keep high intensities using label centroids
    """

    data_labels_u = data_labels.copy()
    indx = np.where(centroids[:, -1] > bk_thr)[0]
    max_indx = np.argmax(centroids[:, -1])

    for indx_ in indx:
        data_labels_u[data_labels == indx_] = max_indx

    mask = data_labels_u == max_indx
    maskb = mask.copy()
    mask = mask.astype('f8')
    if updated_labels:
        return mask, data_labels_u
    return mask


def print_centroids(centroids):
    for idx, c in zip(range(centroids.shape[0]), centroids[:, -1]):
        print(idx, c)
    print()


def patch_sums(data, patch):
    """ Sums in linearized patches
    """

    data2 = np.pad(data, patch[0]//2)
    data2_enh = patchify(data2, patch)
    data2_enh = data2_enh.reshape(data2_enh.shape[0],
                                  data2_enh.shape[1],
                                  data2_enh.shape[2],
                                  patch[0] * patch[1] * patch[2])

    return np.sum(data2_enh, axis=-1)


def patch_median(data, patch):
    """ Sums in linearized patches
    """

    data2 = np.pad(data, patch[0]//2)
    data2_enh = patchify(data2, patch)
    data2_enh = data2_enh.reshape(data2_enh.shape[0],
                                  data2_enh.shape[1],
                                  data2_enh.shape[2],
                                  patch[0] * patch[1] * patch[2])

    return np.median(data2_enh, axis=-1)


def labels_sp_centroids(labels, centroids):
    """ count labels in patch
    """
    tmp = labels.copy()

    for i in range(centroids.shape[0]):

        tmp[labels == i] = centroids[i][3]

    return tmp


def ball(radius, strict=True, dtype=np.uint8):
    n = 2 * radius + 1
    Z, Y, X = np.mgrid[-radius:radius:n * 1j,
                        -radius:radius:n * 1j,
                        -radius:radius:n * 1j]
    s = X ** 2 + Y ** 2 + Z ** 2
    if not strict:
        radius += 0.5
    return np.array(s <= radius * radius, dtype=dtype)


def throwing_planets(mask, W, radius=10, samples=100, perc=5):
    # keep the spheres that have overlaping volume of less than percentage above

    mask_orig = mask.copy()
    print('Starting throwing planets ... light version ')
    t = time()

    # mask = binary_closing(mask, iterations=1)
    # mask = binary_dilation(mask, iterations=1)

    # TODO: this part needs work
    # mask = gaussian_filter(10 * mask, sigma=3)
    # mask_tmp = mask.copy()
    # mask = (mask > 0.1).astype('int32')

    tmp = mask * 0
    kernel = ball(radius)
    kernel_sum = kernel.sum()
    print('>> Kernel sum', kernel_sum)

    kI, kJ, kK = np.asarray(kernel.shape) // 2
    I, J, K = tmp.shape

    volcov = np.prod(kernel.shape)
    print('>> Kernel ratio ', kernel_sum / volcov)

    perc = perc / 100.

    i_s = np.random.randint(radius, I - radius, size=samples)
    j_s = np.random.randint(radius, J - radius, size=samples)
    k_s = np.random.randint(radius, K - radius, size=samples)

    for i, j, k in zip(i_s, j_s, k_s):

        hit = (mask[i - kI:i + kI + 1, j - kJ:j + kJ + 1, k - kK : k + kK + 1] * kernel).sum()
        if hit < perc * kernel_sum:
            tmp[i - kI:i + kI + 1, j - kJ:j + kJ + 1, k - kK : k + kK + 1] += kernel

    print('Duration ', time() - t)
    print()

    return tmp


def centroid_images(data_labels, centroids):

    labels = np.unique(data_labels)
    centroids_image = np.zeros(data_labels.shape)
    for i, l in enumerate(labels):
        centroids_image[data_labels == l] = centroids[i, -1]

    return centroids_image


def medroid_images(data_labels, data):

    labels = np.unique(data_labels)
    medroids_image = np.zeros(data_labels.shape)
    for i, l in enumerate(labels):
        medroids_image[data_labels == l] = np.median(data[data_labels == l])
    return medroids_image


def cluster_analysis(data_labels_pg, centroids_pg, lod = 0.8):

    data_labels_pg_u = data_labels_pg.copy()
    unq, cnts = np.unique(data_labels_pg_u.ravel(), return_counts=True)

    print('Further analyzing clusters')
    # from dipy.viz import horizon
    total = np.zeros(data_labels_pg.shape)
    total_cnt = np.zeros(data_labels_pg.shape)
    t = time()

    cnt = 1
    for i, u in enumerate(unq):

        print('Unique label ', u, ' centroid ', centroids_pg[i, 3])
        tmp = (data_labels_pg_u == u)
        print(tmp.min(), tmp.max())
        dt_tmp = distance_transform_edt(tmp)

        tmp_o = dt_tmp > lod # 1 will darken most smaller structures
        print(tmp_o.min(), tmp_o.max())

        # ratio = tmp_o.sum() / np.prod(data_labels_pg.shape)
        # print('>>', ratio)
        # ratio2 = tmp_o.sum() / tmp.sum()
        # print('>>', ratio2)

        total[tmp_o == 1] = centroids_pg[int(u), 3]

    print('Duration ', time() - t)

    # total_blur = np.zeros(data.shape)
    # total_blur[total > 0] = 10.0
    total_blur = total.copy()
    total_blur = gaussian_filter(total_blur, sigma=1)

    return total, total_blur


def labels_analysis(data_labels, mask, grand_thr=100000):

    print('Focusing on the grand objects ...')
    t = time()
    unq, cnts = np.unique(data_labels.ravel(), return_counts=True)
    lrg = grand_thr
    print(cnts)
    label_cnt = 1
    grand_labels = 0 * data_labels
    from dipy.viz import horizon
    for l in unq:

        biggest = detect_largest_object(data_labels == l, structure=np.ones((3, 3, 3))) # second largest
        if biggest.sum() > grand_thr:
            kernel = ball(1)
            biggest = binary_opening(biggest, structure=kernel, iterations=1, border_value=0)
            grand_labels[biggest > 0] = label_cnt
            print(biggest.max())
            # horizon(images=[(biggest.astype('f8'), np.eye(4), 'current')])
            label_cnt += 1

    # t = 1
    print('Duration ', time() - t)
    return grand_labels


def labels_analysis2(data_labels, mask, grand_thr=100000):

    print('Focusing on the grand objects ...')
    t = time()
    unq, cnts = np.unique(data_labels.ravel(), return_counts=True)
    lrg = grand_thr
    print(cnts)
    label_cnt = 1
    grand_labels = 0 * data_labels
    # from dipy.viz import horizon
    for l in unq:

        if l > -1:
            biggest = detect_largest_object(data_labels == l, structure=np.ones((3, 3, 3))) # second largest

            kernel = ball(1)
            biggest = binary_opening(biggest, structure=kernel, iterations=1, border_value=0)
            grand_labels[biggest > 0] = label_cnt
            print(biggest.max())
            # horizon(images=[(biggest.astype('f8'), np.eye(4), 'current')])
            label_cnt += 1

    # t = 1
    print('Duration ', time() - t)
    return grand_labels


def weighted_image(dt_pg, W):

    weights = dt_pg.copy()
    weights[dt_pg > 5] = 5
    weights_tmp = weights.copy()
    weights = gaussian_filter(weights, sigma=1)
    weights_tmp2 = weights.copy()
    weights = weights * W

    return weights


def weighted_image2(dt, W, idt, masko):

    weights = np.zeros(dt.shape)
    weights[dt > 6] = 6

    # weights = gaussian_filter(weights, sigma=1)
    # weights[weights < .1] = 0
    # weights[weights >= .1] = 1

    data_labels_focused, num_focused = label(weights)
    favorite = analyze_labels(data_labels_focused, num_focused)

    focus  = np.zeros(dt.shape)
    focus[data_labels_focused == favorite] = 1

    cm = center_of_mass(focus)
    cx, cy, cz = np.round(cm).astype('int32')

    cm_im = np.ones(dt.shape)
    cm_im[cx, cy, cz] = 0

    dt2 = distance_transform_edt(cm_im)

    tmp = dt2 + 50 * masko
    new_mask = np.zeros(dt.shape)
    new_mask[tmp > 100] = 1

    t = 1
    # weights_tmp = weights.copy()
    # weights = gaussian_filter(weights, sigma=1)
    # weights_tmp2 = weights.copy()
    return focus * W


def berserker2_3D(data, affine, theta_value=120., perc=(0, 98), mult=300., enh=3., medroids=False):
    "Thetan Berserker2 applied on 3D images"

    data = np.ascontiguousarray(data, dtype='f8')

    print('Preparing data')
    X = prepare_data_3d(data, affine, perc=perc, mult=mult, enh=enh)
    Ivol, Jvol, Kvol, W = X_to_many_3D(X, data.shape)

    print('Applying Berserker 2 on X (i, j, k, 3w)')
    t = time()
    data_labels, centroids = berserker2_to_3D_labels(X, theta_value, data_dims=data.shape, verbose=False)
    print('Duration ', time() - t)
    print()

    print_centroids(centroids)
    centroids_im = centroid_images(data_labels, centroids)

    return data_labels, centroids, centroids_im, W


def berserker2_brain_seg_3d_as_2d_series(data, affine, return_all=False, orig_est=100, m_est=3, theta_value=120., percb=1., pg=False, ng=False):
    """ Process each slice of the volume independently """

    data = np.ascontiguousarray(data, dtype='f8')

    print('Preparing data')
    X = prepare_data_3d_as_2d_series(data, affine, perc=(0, 98), mult=300., enh=2.)
    W = data

    print('Applying Berserker 2 on X (i, j, k, 3w)')
    t = time()
    data_labels, centroids = berserker2_to_3D_labels(X, theta_value, data_dims=data.shape, verbose=False)
    print('Duration ', time() - t)
    print()

    print_centroids(centroids)

    mask = mask_lt(data_labels, centroids, bk_thr= orig_est * m_est)
    dt = distance_transform_edt(1 - mask)

    mask_orig = W < orig_est
    W[mask_orig] = 0
    mask_orig = mask_orig.astype('uint8')

    if pg or ng:
        print('Calculating gradients')
        Wgrad = gaussian_gradient_magnitude(W, sigma=1)

    if pg:
        Xpg = X.copy()
        Xpg[:, 3] += 1.4 * Wgrad.ravel()
        # Ivol, Jvol, Kvol, Wpg = X_to_many_3D(Xpg, data.shape)

        print('Applying Berserker 2 on Xpg (i, j, k, 3w + 2gradW)')
        t = time()
        data_labels_pg, centroids_pg = berserker2_to_3D_labels(Xpg, theta_value, data_dims=data.shape, verbose=False)
        print('Duration ', time() - t)
        print()

        print_centroids(centroids_pg)

        mask_pg = mask_lt(data_labels_pg, centroids_pg, bk_thr= orig_est * m_est)
        dt_pg = distance_transform_edt(1 - mask_pg)

    if ng:
        Xng = X.copy()
        Xng[:, 3] -= 1.4 * Wgrad.ravel()
        # Ivol, Jvol, Kvol, Wng = X_to_many_3D(Xng, data.shape)
        print('Applying Berserker 2 on Xng (i, j, k, 3w - 2gradW)')
        t = time()
        data_labels_ng, centroids_ng = berserker2_to_3D_labels(Xng, theta_value, data_dims=data.shape, verbose=False)
        print('Duration ', time() - t)
        print()

        print_centroids(centroids_ng)

        mask_ng = mask_lt(data_labels_ng, centroids_ng, bk_thr= orig_est * m_est)
        dt_ng = distance_transform_edt(1 - mask_ng)

    centroids_im = centroid_images(data_labels, centroids)
    centroids_median_im = median_filter(centroids_im, size=3)

    standard_filters = False
    if standard_filters:

        centroids_imb = gaussian_filter(centroids_im, sigma=1)
        centroids_imb2 = gaussian_filter(centroids_im, sigma=2)
        centroids_imm = median_filter(centroids_im, size=3)


    print('Applying Berserker 2 on X masked (i, j, k, 3w)')
    t = time()
    maskr = mask.ravel().astype('bool')
    data_labels_mask, centroids_mask = berserker2_to_3D_labels(X, theta_value / 2.0, data_dims=data.shape, verbose=False, mask=maskr)
    print('Duration ', time() - t)
    print()

    centroids_mask_im = centroid_images(data_labels_mask, centroids_mask)
    centroids_mask_median_im = median_filter(centroids_mask_im, size=3)
    # sums_im = triangular_variance(mask_orig, Ivol, Jvol, Kvol, W, msamples=100, osamples=5000)

    # w_im = weighted_image2(dto, centroids_im, idto, mask_orig)

    aff = np.eye(4)
    images = []
    images.append((mask_orig, aff, 'mask_orig'))
    images.append((mask, aff, 'mask'))

    if pg:
        images.append((mask_pg, aff, 'mask_pg'))
        images.append((data_labels_pg, aff, 'dl_pg'))
        images.append((W+Wgrad*1.4, aff, 'Wpg'))

    if standard_filters:
        images.append((centroids_im, aff, 'centroids_im'))
        images.append((centroids_imb, aff, 'centroids_imb'))
        images.append((centroids_imb2, aff, 'centroids_imb2'))
        images.append((centroids_imm, aff, 'centroids_imm'))
    images.append((data_labels, aff, 'dl'))

    if pg or ng:
        images.append((Wgrad, aff, 'Wgrad'))

    # images.append((sums_im, aff, 'sums_im'))

    images.append((centroids_im, aff, 'centroids_im'))
    images.append((centroids_median_im, aff, 'centroids_median_im'))
    images.append((centroids_mask_im, aff, 'centroids_mask_im'))
    images.append((centroids_mask_median_im, aff, 'centroids_mask_median_im'))

    images.append((data_labels_mask, aff, 'dl_mask'))
    images.append((W, aff, 'W'))
    images.append((data, aff, 'data'))


    t = 1

    return images


def berserker2_seg_2d(data, theta_value, enh=2, pg=False, ng=False):
    """Assuming data are gray scale and 2D"""

    X = prepare_data_2d(data, enh)
    I, J, W = X_to_many_2D(X, data.shape)

    if pg:
        gradW = gaussian_gradient_magnitude(W, sigma=1)
        X[:, 2] += 3 * gradW.ravel()

    if ng:
        gradW = gaussian_gradient_magnitude(W, sigma=1)
        X[:, 2] -= 3 * gradW.ravel()

    data_labels, centroids = berserker2_to_2D_labels(X, theta_value, data.shape)
    centroids_im = centroid_images(data_labels, centroids)

    return data_labels, centroids, centroids_im


def berserker2_recovery(data, affine, theta_value=120., perc=(0, 98), mult=300., enh=3, reps=100, medroids=False):
    """Signal recovery"""

    t_start = time()
    c_f = np.zeros(data.shape, dtype='f8')
    m_f = np.zeros(data.shape, dtype='f8')

    for i in range(reps):
        print('>>> ', i)
        print('Starting Thetan Berserker 2 applied to 3D images...')
        t = time()
        data_labels, centroids, centroids_im, W = berserker2_3D(data, affine, theta_value=theta_value, perc=perc, mult=mult, enh=enh)

        print('TB Duration ', time() - t)
        print()
        c_f += centroids_im
        if medroids:
            medroid_im = medroid_images(data_labels, W)
            m_f += medroid_im

    c_f /= reps
    if medroids:
        m_f /= reps
        return c_f, W, m_f

    print('All runs ', time() - t_start)



    return c_f, W


def detect_largest_object(data, structure=np.ones((3, 3, 3)), actually=False):
    # returns the largest object (second largest after background) found in a binary image (data)
    # or first if actually is True.
    data_labels_t, num_t = label(data, structure=structure)
    dlt_unq, dlt_cnts = np.unique(data_labels_t, return_counts=True)
    if actually:
        dl_final = dlt_unq[np.argsort(dlt_cnts)[-1]]
    else:
        dl_final = dlt_unq[np.argsort(dlt_cnts)[-2]]
    final = np.zeros(data.shape, dtype='f8')
    final[data_labels_t == dl_final] = 1.
    return final


def gaussian_gradients(data, sigma=1):
    # calculate gaussian gradients
    # data should be f8

    dx = gaussian_filter(data, sigma=sigma, order = (1, 0, 0))
    dy = gaussian_filter(data, sigma=sigma, order = (0, 1, 0))
    dz = gaussian_filter(data, sigma=sigma, order = (0, 0, 1))

    gm = np.sqrt(dx**2 + dy**2 + dz**2)

    return dx, dy, dz, gm


def auto_theta_rw(X):
    # randomly pick M number of features and calculate the direct distance
    # study the distribution of distances
    pass


def breaking_phase(data, affine, center, vals, orig_est, poi, centroids_median_im_rnd, X_median, theta_value_small):
    
    ci, cj, ck = center
    puzz_mask = np.zeros(data.shape)
    puzz_mask_cnt = np.zeros(data.shape)

    fc = np.array([[ci, cj, ck]])
    fc_w = np.dot(affine[:3, :3], fc.T) + affine[:3, 3][:, None]
    fc_w = fc_w.T[0]

    cnt = 1

    relabel_breaking = np.zeros(data.shape)

    cnt_r = 1

    all_puzz_centroids = []
    for v in vals:

        print(v)

        if v > orig_est: # original mask intensity

            mask_tmp = np.zeros(data.shape)
            mask_tmp[centroids_median_im_rnd == v] = 1

            data_labels_tmp, centroids_tmp = berserker2_to_3D_labels(X_median, theta_value_small, data_dims=data.shape, verbose=False, mask=mask_tmp.ravel().astype('bool'))
            dl_tmp, dl_tmp_cnts = np.unique(data_labels_tmp, return_counts=True)
            # horizon(images=[(data_labels_tmp, aff, 'dl_tmp')])

            # starting from 1 because we have -1 in dl_tmp
            puzz = np.ascontiguousarray(centroids_tmp[1:, :3])
            all_puzz_centroids.extend(puzz)

            for i, p in enumerate(puzz):
                relabel_breaking[data_labels_tmp==i] = cnt_r
                cnt_r += 1

            puzz2center = []
            for p in puzz:
                dc = np.sqrt(np.sum((fc_w - p) ** 2))
                puzz2center.append(dc)
            puzz2center = np.asarray(puzz2center)

            poi = np.ascontiguousarray(poi)
            puzz2poi_mat = distance_matrix(puzz, poi, 'euclidean')

            # TODO: get a percentage of the closest ones
            for i, d in enumerate(puzz2center):
                if d < 100:
                    puzz_mask[data_labels_tmp == dl_tmp[i + 1]] = 1
                    puzz_mask_cnt[data_labels_tmp == dl_tmp[i + 1]] = cnt
                    cnt += 1

            # data_labels_tmp, num_tmp = label(mask_tmp, structure=None)
            # horizon(images=[(data_labels_tmp, aff, 'dl_tmp')])
            # dl_tmp, _ = np.unique(data_labels_tmp, return_counts=True)
            # for l in dl_tmp:
            #     centroids_im_sp[data_labels_tmp == l]

    return relabel_breaking, all_puzz_centroids, puzz_mask, puzz_mask_cnt


def berserker2_brain_seg(data, affine, return_all=False, orig_est=100, m_est=3, theta_value=120., percb=1., pg=False, ng=False):

    data = np.ascontiguousarray(data, dtype='f8')

    print('Preparing data')
    X = prepare_data_3d(data, affine, perc=(0, 98), mult=300., enh=3.)
    Ivol, Jvol, Kvol, W = X_to_many_3D(X, data.shape)

    print('Applying Berserker 2 on X (i, j, k, 3w)')
    t = time()
    data_labels, centroids = berserker2_to_3D_labels(X, theta_value, data_dims=data.shape, verbose=False)
    print('Duration ', time() - t)
    print()

    print('Intensity estimates ...')
    print_centroids(centroids)

    mask_ci = mask_lt(data_labels, centroids, bk_thr= orig_est * 1.5) # 1.2 previously - this parameter is important - it may need to be a hyperparameter
    mask_ext = mask_lt(data_labels, centroids, bk_thr= orig_est * m_est)

    # dt = distance_transform_edt(1 - mask_ext)

    mask_orig = W < orig_est
    W[mask_orig] = 0
    mask_orig = mask_orig.astype('uint8')

    # mask_diff = mask_ext - mask_orig

    if pg or ng:
        print('Calculating gradients')
        Wgrad = gaussian_gradient_magnitude(W, sigma=1)

    if pg:
        Xpg = X.copy()
        Xpg[:, 3] += 2 * Wgrad.ravel()
        Ivol, Jvol, Kvol, Wpg = X_to_many_3D(Xpg, data.shape)

        print('Applying Berserker 2 on Xpg (i, j, k, 3w + 2gradW)')
        t = time()
        data_labels_pg, centroids_pg = berserker2_to_3D_labels(Xpg, theta_value, data_dims=data.shape, verbose=False)
        print('Duration ', time() - t)
        print()

        print_centroids(centroids_pg)

        mask_pg = mask_lt(data_labels_pg, centroids_pg, bk_thr= orig_est * m_est)
        dt_pg = distance_transform_edt(1 - mask_pg)

    if ng:
        Xng = X.copy()
        Xng[:, 3] -= 2 * Wgrad.ravel()
        Ivol, Jvol, Kvol, Wng = X_to_many_3D(Xng, data.shape)
        print('Applying Berserker 2 on Xng (i, j, k, 3w - 2gradW)')
        t = time()
        data_labels_ng, centroids_ng = berserker2_to_3D_labels(Xng, theta_value, data_dims=data.shape, verbose=False)
        print('Duration ', time() - t)
        print()

        print_centroids(centroids_ng)

        mask_ng = mask_lt(data_labels_ng, centroids_ng, bk_thr= orig_est * m_est)
        dt_ng = distance_transform_edt(1 - mask_ng)

    lightsoff = False
    if lightsoff:
        print('Calculating weights')
        weights = weighted_image(dt_pg, W)
        print('Applying Berserker 2 on X (i, j, k, 3w)')
        t = time()
        data_labels, centroids = berserker2_to_3D_labels(X, theta_value, data_dims=data.shape, verbose=False)
        print('Duration ', time() - t)
        print()

        print_centroids(centroids)

        mask = mask_lt(data_labels, centroids, bk_thr= orig_est * m_est)
        dt = distance_transform_edt(1 - mask)

        # TODO: this is useful needs cleaning
        Xw = X.copy()
        Xw[:, 3] = weights.ravel()
        print('Applying Berserker 2 on Xw (re-weighted)')
        t = time()
        data_labels_w, centroids_w = berserker2_to_3D_labels(Xw, theta_value * 5, data_dims=data.shape, verbose=False)
        print('Duration ', time() - t)
        print()

        print_centroids(centroids_w)
        mask_w = mask_lt(data_labels_w, centroids_w, bk_thr= 5 * orig_est * m_est)
        imask_w = 1 - mask_w

    auto_edt_rm = False
    if auto_edt_rm:
        print('Starting cluster analysis of dl_pg ..')
        total_pg, total_blur_pg = cluster_analysis(data_labels_pg, centroids_pg, 0.9)

        print('Starting cluster analysis of dl ..')
        total, total_blur = cluster_analysis(data_labels, centroids, 0.9)

    centroids_im = centroid_images(data_labels, centroids)
    centroids_median_im = median_filter(centroids_im, size=3)

    if pg:
        centroids_im_pg = centroid_images(data_labels_pg, centroids_pg)

    if ng:
        centroids_im_ng = centroid_images(data_labels_ng, centroids_ng)

    masked_berserker = False
    if masked_berserker:
        print('Applying Berserker 2 on X masked (i, j, k, 3w)')
        t = time()
        maskr = mask_orig.ravel().astype('bool')
        data_labels_mask, centroids_mask = berserker2_to_3D_labels(X, theta_value / 2.0, data_dims=data.shape, verbose=False, mask=maskr)
        print('Duration ', time() - t)
        print()

        # centroids_mask_im = centroid_images(data_labels_mask, centroids_mask)
        # centroids_mask_median_im = median_filter(centroids_mask_im, size=3)
        # sums_im = triangular_variance(mask_orig, Ivol, Jvol, Kvol, W, msamples=100, osamples=5000)

        # w_im = weighted_image2(dto, centroids_im, idto, mask_orig)

    aff = np.eye(4)
    images = []

    images.append((W, aff, 'W'))
    images.append((data, aff, 'data'))
    images.append((mask_orig, aff, 'mask_orig'))
    images.append((mask_ext, aff, 'mask_ext'))
    # images.append((mask_diff, aff, 'mask_diff'))


    # images.append((mask_ci, aff, 'mask_ci'))


    if masked_berserker:
        images.append((data_labels_mask, aff, 'dl_mask'))

        t = 2
    if auto_edt_rm:
        images.append((total_pg, aff, 'total_pg'))
        images.append((total_blur_pg, aff, 'total_blur_pg'))
        images.append((total, aff, 'total'))
        images.append((total_blur, aff, 'total_blur'))
    if pg:
        images.append((mask_pg, aff, 'mask_pg'))
        images.append((data_labels_pg, aff, 'dl_pg'))
        images.append((Wpg, aff, 'Wpg'))
    if lightsoff:
        images.append((weights, aff, 'weights'))
        images.append((data_labels_w, aff, 'dl_w'))
        images.append((imask_w, aff, 'imask_w'))

    images.append((data_labels, aff, 'dl'))

    if pg or ng:
        images.append((Wgrad, aff, 'Wgrad'))

    # images.append((sums_im, aff, 'sums_im'))

    images.append((centroids_im, aff, 'centroids_im'))
    images.append((centroids_median_im, aff, 'centroids_median_im'))

    if pg:
        images.append((centroids_im_pg, aff, 'centroids_im_pg'))
    if ng:
        images.append((centroids_im_ng, aff, 'centroids_im_ng'))

    negone = False # requires masked berserker
    if negone:
        overshoot = np.zeros(data.shape)
        # combined_mask = (data_labels_mask > -1) & (centroids_mask_im < 0.5)
        # overshoot[combined_mask] = 1
        kernel = ball(5)
        # overshoot2 = binary_closing(overshoot, structure=kernel, iterations=1).astype('f8')
        # kernel = ball(5)

        # overshoot3 = binary_closing(overshoot2, structure=kernel, iterations=1).astype('f8')

        # kernel = ball(5)
        # closing = binary_closing(overshoot, structure=kernel, iterations=1).astype('f8')

        # close same labels that are not -1
        unql, cntsl = np.unique(data_labels_mask, return_counts=True)
        print('Unique >-1 labels')
        print(unql)
        print(cntsl)

        together = np.zeros(data.shape, dtype='f8')
        for l in unql:
            print(l)
            if l > -1:
                together += binary_closing(data_labels_mask==l, structure=kernel, iterations=1, border_value=-2).astype('f8')
                # TODO: the closing above should be applied to a padded input otherwise a border is drawn

        blacky = together==0

        # provides the largest object (second largest after background) in a binary image
        final = detect_largest_object(blacky, structure=np.ones((3, 3, 3)))
        # TODO: remove the piece below
        # data_labels_t, num_t = label(blacky, structure=np.ones((3, 3, 3)))
        # dlt_unq, dlt_cnts = np.unique(data_labels_t, return_counts=True)
        # dl_final = dlt_unq[np.argsort(dlt_cnts)[-2]]
        # final = np.zeros(data.shape, dtype='f8')
        # final[data_labels_t == dl_final] = 1.

        # images.append((overshoot, aff, 'overshoot'))
        images.append((together, aff, 'together'))
        images.append((blacky.astype('f8'), aff, 'blacky'))
        # images.append((data_labels_t, aff, 'dl_t'))
        images.append((final, aff, 'final'))

    # mask2 = mask_lt(data_labels, centroids, bk_thr=orig_est * 1.8)
    mask2 = mask_orig.copy()
    mask2 = mask2.astype('f8')

    # images.append((mask2, aff, 'mask2'))

    # currently using original mask
    tbl0 = throwing_planets(mask2, W=W, radius=28, samples=20000, perc=2)
    tbl0 = tbl0 / tbl0.max()

    images.append((tbl0, aff, 'tbl0'))

    mask3 = mask2.copy()
    # mask3 = mask_ext.copy()
    # mask3 = mask_ci.copy()

    # keep only a highly focus and central area
    potential_focus = tbl0 > 0.2
    # find largest object (after bk)
    focus = detect_largest_object(potential_focus, None)
    images.append((focus, aff, 'focus'))

    # calculate center of mass
    ci, cj, ck = np.round(center_of_mass(focus), 0).astype(int)
    print('Nicer center ...', ci, cj, ck)

    # make mask3 zero in the focus areas to remove errors from large ventricles etc.
    mask3[focus.astype('bool')] = 0
    images.append((mask3, aff, 'mask3'))

    # use opening to increase mask3 (cleared original mask)
    kernel = ball(2)
    mask3_bo = binary_opening(mask3, structure=kernel, iterations=1, border_value=1) # check borders

    # getting the actual background (assuming it is the largest object) TODO: add a secondary method for detecting bk
    bk = detect_largest_object(mask3_bo, structure=None, actually=True)
    images.append((bk, aff, 'bk'))

    kernel = ball(8)
    bk2 = binary_dilation(bk, structure=kernel, iterations=1)

    images.append((bk2.astype('f8'), aff, 'bk2'))

    # remove actual background.
    mask_nobk = mask3.copy()
    mask_nobk[bk2 > 0] = 0

    # remove any boundary issues
    mask_nobk[:2, :, :] = 0
    mask_nobk[-2:, :, :] = 0
    mask_nobk[:, :2, :] = 0
    mask_nobk[:,-2:, :] = 0
    mask_nobk[:, :, :2] = 0
    mask_nobk[:, :, -2:] = 0

    images.append((mask_nobk, aff, 'mask_nobk'))

    print('Applying Berserker 2 on X with nobk mask')
    t = time()
    maskr = mask_nobk.ravel().astype('bool')
    data_labels_nobk, centroids_nobk = berserker2_to_3D_labels(X, theta_value / 4.0, data_dims=data.shape, verbose=False, mask=maskr)
    print('Duration ', time() - t)
    print()

    centroids_im_nobk = centroid_images(data_labels_nobk, centroids_nobk)
    images.append((data_labels_nobk, aff, 'data_labels_nobk'))

    poi = centroids_nobk[:, :3]
    fc = np.array([[ci, cj, ck]])
    fc_w = np.dot(affine[:3, :3], fc.T) + affine[:3, 3][:, None]
    fc_w = fc_w.T[0]

    poi2center = []
    for p in poi:
        dc = np.sqrt(np.sum((fc_w - p) ** 2))
        poi2center.append(dc)

    poi2center = np.asarray(poi2center)
    centroids_nobk2 = centroids_nobk.copy()
    for i, d in enumerate(poi2center):
        if d < 120:
            centroids_nobk2[i, 3] = 1.

    mask_reduced_nobk2 = centroid_images(data_labels_nobk, centroids_nobk2)
    images.append((mask_reduced_nobk2, aff, 'mask_reduced_nobk2'))


    centroids_median_im_rnd  = np.round(centroids_median_im, 0).astype('i8')
    vals, counts = np.unique(centroids_median_im_rnd, return_counts=True)
    images.append((centroids_median_im_rnd, aff, 'centroids_median_im_rnd'))

    print('Starting operating on median puzzle pieces ...')
    X_median = X.copy()
    X_median[:, 3] = centroids_median_im_rnd.ravel().astype('f8')

    print('Breaking phase ... with berserker 2 ...')
    t = time()

    theta_value_small = theta_value / 8.
    relabel_breaking, all_puzz_centroids, puzz_mask, puzz_mask_cnt = breaking_phase(data, affine, (ci, cj, ck),
                                                                                    vals, orig_est, poi, centroids_median_im_rnd,
                                                                                    X_median, theta_value_small)

    print('Duration ', time() - t)
    print()

    images2 = []
    images2.append((W, aff, 'W'))
    images2.append((mask_reduced_nobk2.astype('f8'), aff, 'mask_reduced_nobk2'))
    images2.append((relabel_breaking.astype('f8'), aff, 'relabel_breaking'))
    images2.append((puzz_mask, aff, 'puzz_mask'))
    images2.append((puzz_mask_cnt, aff, 'puzz_mask_cnt'))


    # Remove obvious things from puzz_mask_cnt
    puzz_mask_cnt2 = puzz_mask_cnt.copy()
    puzz_mask_cnt2[bk2 == 1] = 0
    puzz_mask_cnt2[mask_nobk == 1] = 0
    images2.append((puzz_mask_cnt2, aff, 'puzz_mask_cnt2'))

    puzz_mask2 = np.zeros(data.shape)
    
    # images2.append((mask_puzz_mask_cnt2.astype('f8'), aff, 'mask_puzz_mask_cnt2'))

    puzz_mask2[puzz_mask_cnt2 > 0] = 1
    images2.append((puzz_mask2.astype('f8'), aff, 'puzz_mask2'))

    puzz_mask_smooth = gaussian_filter(10 * puzz_mask2.astype('f8'), sigma=4)
    images2.append((puzz_mask_smooth.astype('f8'), aff, 'puzz_mask_smooth'))

    boundary = ((puzz_mask_smooth > 0.2) & (puzz_mask_smooth < 1.2)).astype('f8')
    images2.append((boundary.astype('f8'), aff, 'boundary'))

    print('Applying Berserker 2 on X with internal bk mask')
    t = time()
    data_labels_intbk, centroids_intbk = berserker2_to_3D_labels(X, theta_value / 8.0, data_dims=data.shape, verbose=False, mask=maskr)
    print('Duration ', time() - t)
    print()

    images2.append((data_labels_intbk.astype('f8'), aff, 'data_labels_intbk'))


    print('Applying Berserker 2 on X external boundary')
    t = time()
    maskr3 = boundary.ravel().astype('bool')
    data_labels_boundary, centroids_boundary = berserker2_to_3D_labels(X, theta_value / 4.0, data_dims=data.shape, verbose=False, mask=maskr3)
    print('Duration ', time() - t)
    print()

    images2.append((data_labels_boundary.astype('f8'), aff, 'data_labels_boundary'))

    all_puzz_centroids = np.ascontiguousarray(all_puzz_centroids)


    tmp_cintbk = np.ascontiguousarray(centroids_intbk[:, :3])

    allpuzz2intbk_mat = distance_matrix(all_puzz_centroids, tmp_cintbk, 'euclidean')
    print('Puzz2intbk Matrix shape', all_puzz_centroids.shape)

    tmp_cboundary = np.ascontiguousarray(centroids_boundary[:, :3])

    allpuzz2bound_mat = distance_matrix(all_puzz_centroids, tmp_cboundary, 'euclidean')
    print('Puzz2bound Matrix shape', allpuzz2bound_mat.shape)

    # TODO: how many times do we have neighbors from the boundary at less than e.g. 20?

    mins_int2bk = np.min(allpuzz2intbk_mat, axis=-1)
    argmins_int2bk = np.argmin(allpuzz2intbk_mat, axis=-1)

    dist_puzz_centroids_from_center = np.sqrt(np.sum(all_puzz_centroids ** 2, axis=-1))
    corresponding_int2bk = tmp_cintbk[argmins_int2bk]

    dist_corresponding_from_center = np.sqrt(np.sum(corresponding_int2bk ** 2, axis=-1))

    mins_bound = np.min(allpuzz2bound_mat, axis=-1)
    argmins_bound = np.argmin(allpuzz2bound_mat, axis=-1)
    corresponding_bound = tmp_cintbk[argmins_bound]

    dist_bound_from_center = np.sqrt(np.sum(corresponding_bound ** 2, axis=-1))

    between_region = np.zeros(data.shape)
    enum_range = range(int(relabel_breaking.max()))
    for i, dpc, dcc, dpb in zip(enum_range, dist_puzz_centroids_from_center, dist_corresponding_from_center, dist_bound_from_center):
    
        if dpc < 100:
            if (dpc < dcc):
                between_region[relabel_breaking == i + 1] = 1
   
    images2.append((between_region.astype('f8'), aff, 'between_region'))

    # TODO: calculate triangles
    t = 1

    show_points = True
    if show_points:
        from fury import actor, window, colormap
        scene = window.Scene()

        # colors = colormap.create_colormap(poi2center, name='plasma', auto=True)
        # points_ = actor.point(np.array([fc_w]), (1, 1, 1), point_radius=2.)
        # points2_ = actor.point(poi, colors, point_radius=2.)
        # colors = colormap.create_colormap(poi2center, name='plasma', auto=True)
        center_ = actor.point(np.array([[ci, cj, ck]]), (1, 0, 0), point_radius=4.)
        points_ = actor.point(tmp_cintbk, (0, 1, 0), point_radius=2.)
        # points2_ = actor.point(tmp_cboundary, (1, 0, 0), point_radius=2.)
        points3 = actor.point(all_puzz_centroids, (0, 0, 1), point_radius=1.)
        scene.add(actor.axes())
        scene.add(center_)
        scene.add(points_)
        # scene.add(points2_)
        scene.add(points3)
        window.show(scene)

    t = 2

    return images, images2

    print('Opening and detecting largest object ... ')
    # kernel = ball(6)
    # kernel = ball(8)
    final2o = binary_opening(final2, structure=kernel, iterations=1)

    images2.append((final2o.astype('f8'), aff, 'final2o'))

    final2o_main = detect_largest_object(final2o, structure=None)
    images2.append((final2o_main.astype('f8'), aff, 'final2o_main'))



    mask_reduced = binary_opening(mask_reduced_nobk2, structure=None, iterations=1)
    images2.append((mask_reduced.astype('f8'), aff, 'masked_reduced'))

    mask_reduced2 = (boundary + mask_reduced).astype('bool')
    images2.append((mask_reduced2.astype('f8'), aff, 'masked_reduced2'))

    kernel = ball(10)
    mask_reduced3 = binary_closing(mask_reduced2, structure=kernel, iterations=1)
    images2.append((mask_reduced3.astype('f8'), aff, 'masked_reduced3'))


    print('Connected ', cnt, ' puzzles')

    # # TODO: Rather than dividing by 4 or 5 just break labels in parts?
    # print('Applying Berserker 2 on X2 (i, j, k, 3w) ...')
    # t = time()
    # # TODO: only on the sphere (d < 120)
    # mask_X2 = (1 - centroids_im_nobk2)
    # data_labels2, centroids2 = berserker2_to_3D_labels(X2, theta_value / 4., data_dims=data.shape, verbose=False, mask=mask_X2.ravel().astype('bool'))
    # print('Duration ', time() - t)
    # print()

    return images, images2




    t = 2

    # dx, dy, dz, gm = gaussian_gradients(mask2.astype('f8'), sigma=1)
    # ggm = gaussian_gradient_magnitude(mask2.astype('f8'), sigma=1)

    # images.append((dx, aff, 'dx'))
    # images.append((dy, aff, 'dy'))
    # images.append((dz, aff, 'dz'))
    # images.append((gm, aff, 'gm'))
    # images.append((ggm, aff, 'ggm'))
    # glap = gaussian_laplace(mask2.astype('f8'), sigma=2)
    # images.append((glap, aff, 'glap'))

    la = False
    if la:
        big_objects = labels_analysis(data_labels, mask3_bc)
        images.append((big_objects, aff, 'big_objects'))




    return images



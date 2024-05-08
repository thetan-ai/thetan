from time import time
import numpy as np
from patchify import patchify
from scipy.ndimage import gaussian_gradient_magnitude, label, distance_transform_edt
from scipy.ndimage import binary_erosion, binary_dilation, binary_opening, binary_closing
from scipy.ndimage import generate_binary_structure, gaussian_filter, center_of_mass, median_filter
from thetan.theta import theta_g1, reassign_uncertain_update_labels, centroids_from_labels
from thetan.theta import theta_means, candidate_labels_c, candidate_labels_outliers_c
from thetan.theta import img3d_to_ijkv_features
from skimage.segmentation import flood_fill


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
        return even_centroids_ex2


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

    updated_labels_ex2 = reassign_uncertain_update_labels(points3, even_centroids2,
                                                          updated_labels2x, nice2,
                                                          metric=metric)
    t16 = time()

    info_dict['Shuffle 2 REA X'] = t16 - t15

    # bring labels back from original points (not points3)
    updated_labels_ex2 = updated_labels_ex2[ind2_inv].copy()
    even_centroids_ex2 = centroids_from_labels(points, updated_labels_ex2)

    t17 = time()

    print('ec2x ', even_centroids_ex2.shape)

    info_dict['Shuffle 2 CEN_LA *'] = t17 - t16

    # quality over speed option not available in this version yet

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

    centroids, labels = berserker2(Xp, theta_value, all_out=False)

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

def prepare_data(data, affine, perc=(2, 99), mult=300, enh=3):

    if perc is None:
        data2 = data
    else:
        value_range = np.percentile(data, q=[2, 99])
        data2 = np.interp(data, value_range, (0, 1))
        data2 =  mult * np.ascontiguousarray(data2, dtype='f8')

    X = img3d_to_ijkv_features(data2)
    # move to world coordinates W (xyz)

    W = np.dot(affine[:3, :3], X[:, :3].T) + affine[:3, 3][:, None]
    X[:, :3] = W.T

    # Intensity more important than location
    X[:, 3] = enh * X[:, 3]

    return X


def prepare_data_2d(data, affine, perc=(2, 99), mult=300, enh=3):

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


def throwing_balls(mask, W, radius=10, samples=100, perc=5):

    mask_orig = mask.copy()
    print('Starting throwing solid balls in 3D space ... ')
    t = time()

    mask = binary_closing(mask, iterations=1)
    # mask = binary_dilation(mask, iterations=1)

    # TODO: this part needs work
    # mask = gaussian_filter(10 * mask, sigma=3)
    mask_tmp = mask.copy()
    # mask = (mask > 0.1).astype('int32')

    tmp = mask.copy() * 0
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

        # if hit == 0:
        if hit < perc  * kernel_sum:
            tmp[i - kI:i + kI + 1, j - kJ:j + kJ + 1, k - kK : k + kK + 1] += kernel

    # from dipy.viz import horizon
    # aff = np.eye(4)
    # horizon(images=[(mask_orig, aff, 'mask_orig'), (mask.astype('int32'), aff, 'mask'), (tmp, aff, 'tmp')])
    print('Duration ', time() - t)
    print()

    new_mask = np.zeros(mask.shape)
    new_mask[tmp > 100] = 1

    data_labels_nm, num_focused_nm = label(new_mask)
    favorite_nm = analyze_labels(data_labels_nm, num_focused_nm)

    focused = np.zeros(mask.shape)
    focused[data_labels_nm==favorite_nm] = 1

    focused2 = focused.copy()

    focused2 += mask_orig

    new_mask_dt = distance_transform_edt(1 - focused2)

    new_mask2 = np.zeros(mask.shape)
    new_mask2[new_mask_dt > 5] = 1

    # new_mask2 = binary_opening(new_mask2, iterations=1)

    data_labels_nm2, num_focused_nm2 = label(new_mask2)
    print(num_focused_nm2)

    unq2, cnts2 = np.unique(data_labels_nm2, return_counts=True)

    sizes = []
    for i in range(num_focused_nm2):
        sz = (data_labels_nm2==i).sum()
        # print('>>', i, sz)
        sizes.append(sz)

    print(sorted(sizes))

    from dipy.viz import horizon

    a = np.eye(4)
    images=[(mask_orig.astype('i4'), a, 'masko'),
            (mask.astype('i4'), a, 'mask'),
            (tmp, a, 'tmp'),
            (new_mask, a, 'new_mask'),
            (data_labels_nm, a, 'dl_nm'),
            (focused, a, 'focused'),
            (focused2, a, 'focused2'),
            (new_mask_dt, a, 'new_mask_dt'),
            (new_mask2, a, 'new_mask2'),
            (data_labels_nm2, a, 'dl_nm2'),
            (W, a, 'W')]
    horizon(images=images)


    t = 1

    return tmp, new_mask, new_mask_dt


def centroid_images(data_labels, centroids, round=False):

    labels = np.unique(data_labels)
    centroids_image = np.zeros(data_labels.shape)
    # for i in range(centroids.shape[0]):
    #     centroids_image[data_labels == i] = centroids[i, -1]
    for i, l in enumerate(labels):
        centroids_image[data_labels == l] = centroids[i, -1]

    if round:
        return np.round(centroids_image, 0)
    return centroids_image


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


def triangular_variance(mask, X, Y, Z, W, msamples=100, osamples=1000):

    mask2 = binary_opening(mask, iterations=1).astype('i4')
    mask3 = binary_closing(mask2, iterations=1).astype('i4')
    # mask3 = flood_fill(mask2, (1, 1, 1), 2, connectivity=1)
    # mask4 = (mask3 == 1).astype('i4')

    lab3, num3 = label(mask3)
    fav = analyze_labels(lab3, num3)

    mask4 = np.zeros(mask.shape)
    mask4[lab3 == fav] = 1

    mask5 = np.zeros(mask.shape)
    mask5[mask4 == 1] = 1

    a = np.eye(4)
    from dipy.viz import horizon
    horizon(images=[(mask5, a, 'mask5'), (lab3, a, 'lb3'), (mask4, a, 'm4'), (mask3, a, 'm3'), (mask2, a, 'm2'), (mask, a, 'm'), (W, a, 'W')])

    Xr = X.ravel()
    Yr = Y.ravel()
    Zr = Z.ravel()
    Wr = W.ravel()
    maskr = mask.ravel()

    mXr = Xr[maskr == 1]
    mYr = Yr[maskr == 1]
    mZr = Zr[maskr == 1]



    mlength = mXr.shape[0]
    msel = np.random.randint(0, mlength, size=msamples)

    mXrsel = mXr[msel]
    mYrsel = mYr[msel]
    mZrsel = mZr[msel]

    oXr = Xr[maskr == 0]
    oYr = Yr[maskr == 0]
    oZr = Zr[maskr == 0]

    olength = oXr.shape[0]
    osel = np.random.randint(0, olength, size=osamples)

    oXrsel = oXr[osel]
    oYrsel = oYr[osel]
    oZrsel = oZr[osel]

    t = 1

    sums = []
    for o in range(oXrsel.shape[0]):
        res = np.sqrt((oXrsel[o] * mXrsel) ** 2 + (oYrsel[o] * mYrsel) ** 2 + (oZrsel[o] * mZrsel) ** 2)
        sums.append(res.sum())
        t = 2
    sums = np.asarray(sums)

    sums_im = np.zeros(W.shape)


    imask = maskr == 0
    imask = imask.astype('i4')
    imask[osel] = 2
    imask2 = imask.reshape(W.shape)

    sums_im[imask2==2] = sums

    return sums_im


def multi_scale_patches(mask):

    mask2 = patchify(mask, (10, 10, 10), step=10)

    t = 1


    pass


def berserker2_brain_seg(data, affine, return_all=False, orig_est=100, m_est=3, theta_value=120., percb=1., pg=False, ng=False):

    data = np.ascontiguousarray(data, dtype='f8')

    print('Preparing data')
    X = prepare_data(data, affine, perc=(0, 98), mult=300., enh=3.)
    Ivol, Jvol, Kvol, W = X_to_many_3D(X, data.shape)

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
    # dto = distance_transform_edt(1 - mask_orig.astype('f4'))
    # idto = distance_transform_edt(mask_orig.astype('f4'))

    # psums = patch_sums(mask_orig, (5, 5, 5))
    # mask2 = np.zeros(mask_orig.shape)
    # mask2[psums > 25] = 1

    # print('Time to start throwing solid balls ..')
    # ball_mask, bm2, bm2_dt = throwing_balls(mask_orig, W, 20, 100000, perc=percb)

    # TODO: I need to chose between this weighting strategy or the one
    # below. I also need to use label to find the largest object
    # Ideally the weighting process needs a tiny bit of optimization

    # brain_mask = np.zeros(data.shape)
    # brain_mask[ball_mask > 5] = 1

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
    # images.append((psums, aff, 'patch_sums'))
    # images.append((ball_mask, aff, 'ball_mask'))
    # images.append((bm2, aff, 'bm2'))
    # images.append((bm2_dt, aff, 'bm2_dt'))

    # images.append((brain_mask, aff, 'brain_mask'))

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


def berserker2_brain_seg_2d(data, affine, return_all=False, orig_est=100, m_est=3, theta_value=120., percb=1., pg=False, ng=False):
    """ Process each slice of the volume independently """

    data = np.ascontiguousarray(data, dtype='f8')

    print('Preparing data')
    X = prepare_data_2d(data, affine, perc=(0, 98), mult=300., enh=2.)
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
    # images.append((psums, aff, 'patch_sums'))
    # images.append((ball_mask, aff, 'ball_mask'))
    # images.append((bm2, aff, 'bm2'))
    # images.append((bm2_dt, aff, 'bm2_dt'))

    # images.append((brain_mask, aff, 'brain_mask'))

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


def berserker2_seg_2d(data, theta_value):

    pass

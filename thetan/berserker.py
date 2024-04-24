from time import time
import numpy as np
from patchify import patchify
from skimage.morphology import ball
from scipy.ndimage import gaussian_gradient_magnitude, label, distance_transform_edt
from scipy.ndimage import binary_erosion, binary_dilation, binary_opening, binary_closing
from scipy.ndimage import generate_binary_structure, gaussian_filter
from thetan.theta import theta_g1, reassign_uncertain_update_labels, centroids_from_labels
from thetan.theta import theta_means, candidate_labels_c, candidate_labels_outliers_c
from thetan.theta import img3d_to_ijkv_features


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



def berserker2_to_3D_labels(X, theta_value, data_dims, verbose=False):
    """
    Deploying berserker v2 in imaging data and
    returning labels as 3D volumes
    """
    centroids, labels = berserker2(X, theta_value, all_out=False)
    data_labels = labels.copy()
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
    indx = np.where(centroids[:, 3] < bk_thr)[0]
    min_indx = np.argmin(centroids[:, 3])

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
    indx = np.where(centroids[:, 3] > bk_thr)[0]
    max_indx = np.argmax(centroids[:, 3])

    for indx_ in indx:
        data_labels_u[data_labels == indx_] = max_indx

    mask = data_labels_u == max_indx
    maskb = mask.copy()
    mask = mask.astype('f8')
    if updated_labels:
        return mask, data_labels_u
    return mask


def print_centroids(centroids):
    for idx, c in zip(range(centroids.shape[0]), centroids[:, 3]):
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


def ball_overlap(mask, radius=10, samples=100, perc=5):

    print('Starting throwing solid balls in 3D space ... ')
    t = time()   

    # TODO: this part needs work
    mask = gaussian_filter(10 * mask, sigma=3)
    mask_tmp = mask.copy()
    mask = (mask > 0.1).astype('int32')

    tmp = mask.copy() * 0 
    kernel = ball(radius)
    kI, kJ, kK = np.asarray(kernel.shape) // 2    

    I, J, K = tmp.shape

    volcov = np.prod(kernel.shape)

    perc = perc / 100.

    i_s = np.random.randint(radius, I - radius, size=samples)
    j_s = np.random.randint(radius, J - radius, size=samples)
    k_s = np.random.randint(radius, K - radius, size=samples)

    for i, j, k in zip(i_s, j_s, k_s):

        hit = (mask[i - kI:i + kI + 1, j - kJ:j + kJ + 1, k - kK : k + kK + 1] * kernel).sum()

        # if hit == 0:
        if hit < perc * volcov:
            tmp[i - kI:i + kI + 1, j - kJ:j + kJ + 1, k - kK : k + kK + 1] += kernel


    print('Duration ', time() - t)
    print()

    return tmp


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


def berserker2_brain_seg(data, affine, return_all=False, orig_est=100, m_est=3, theta_value=120.):

    data = np.ascontiguousarray(data, dtype='f8')

    print('Preparing data')
    X = prepare_data(data, affine, perc=(2, 98), mult=300., enh=3.)
    Ivol, Jvol, Kvol, W = X_to_many_3D(X, data.shape)

    mask_orig = W < orig_est
    W[mask_orig] = 0
    mask_orig = mask_orig.astype('uint8')

    print('Time to start throwing snow balls ..')
    ball_mask = ball_overlap(mask_orig, 15, 100000, perc=0.5)

    print('Calculating gradients')
    Wgrad = gaussian_gradient_magnitude(W, sigma=1)

    Xpg = X.copy()
    Xpg[:, 3] += 2 * Wgrad.ravel()
    Ivol, Jvol, Kvol, Wpg = X_to_many_3D(Xpg, data.shape)

    Xng = X.copy()
    Xng[:, 3] -= 2 * Wgrad.ravel()
    Ivol, Jvol, Kvol, Wng = X_to_many_3D(Xng, data.shape)

    print('Applying Berserker 2 on Xg (i, j, k, 3w + 2gradW)')
    t = time()
    data_labels_pg, centroids_pg = berserker2_to_3D_labels(Xpg, theta_value, data_dims=data.shape, verbose=False)
    print('Duration ', time() - t)
    print()

    print_centroids(centroids_pg)

    mask_pg = mask_lt(data_labels_pg, centroids_pg, bk_thr= orig_est * m_est)
    dt_pg = distance_transform_edt(1 - mask_pg)

    print('Calculating weights')
    weights = dt_pg.copy()
    weights[dt_pg > 5] = 5
    weights_tmp = weights.copy()
    weights = gaussian_filter(weights, sigma=1)
    weights_tmp2 = weights.copy()
    weights = weights * W

    print('Applying Berserker 2 on X (i, j, k, 3w)')
    t = time()
    data_labels, centroids = berserker2_to_3D_labels(X, theta_value, data_dims=data.shape, verbose=False)
    print('Duration ', time() - t)
    print()

    print_centroids(centroids)

    mask = mask_lt(data_labels, centroids, bk_thr= orig_est * m_est)
    dt = distance_transform_edt(1 - mask)
      
    Xw = X.copy()
    Xw[:, 3] = weights.ravel()
    print('Applying Berserker 2 on Xw (re-weighted)')
    t = time()
    data_labels_w, centroids_w = berserker2_to_3D_labels(Xw, theta_value * 5, data_dims=data.shape, verbose=False)
    print('Duration ', time() - t)
    print()

    print_centroids(centroids_w)

    # 5 works but 6 also in some cases such as HH-1211
    mask_w = mask_lt(data_labels_w, centroids_w, bk_thr= 5 * orig_est * m_est)
    imask_w = 1 - mask_w

    advanced_analysis = True

    if advanced_analysis: 
        print('Starting cluster analysis of dl_pg ..')
        total_pg, total_blur_pg = cluster_analysis(data_labels_pg, centroids_pg, 0.9)

        print('Starting cluster analysis of dl ..')
        total, total_blur = cluster_analysis(data_labels, centroids, 0.9)
        
    aff = np.eye(4)
    images = []
    images.append((mask_orig, aff, 'mask_orig'))
    images.append((ball_mask, aff, 'ball_mask'))

    if advanced_analysis:
        images.append((total_pg, aff, 'total_pg'))
        images.append((total_blur_pg, aff, 'total_blur_pg'))
        images.append((total, aff, 'total'))
        images.append((total_blur, aff, 'total_blur'))

    images.append((mask_pg, aff, 'mask_pg'))
    images.append((data_labels_pg, aff, 'dl_pg'))
    images.append((data_labels, aff, 'dl'))
    images.append((imask_w, aff, 'imask_w'))
    images.append((Wpg, aff, 'Wpg'))
    images.append((Wgrad, aff, 'Wgrad'))
    images.append((W, aff, 'W'))

    return images

    images = [(mask_pg, aff, 'mask_pg'),
              (data_labels_pg, aff, 'dl_pg'),
              (dl_process, aff, 'dl_process'),
              (imask_w, aff, 'imask_w'), 
              (Wgrad, aff, 'Wgrad'),
              (W, aff, 'W')]

    return images


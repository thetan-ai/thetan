from time import time
import numpy as np
from patchify import patchify
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


def analyze_labels(data_labels_focused, num_focused, verbose=False):

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
    return favorite


def mask_lt(data_labels, centroids, bk_thr=300):
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
    return mask


def mask_gt(data_labels, centroids, bk_thr=300):
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

def berserker2_brain_seg(data, affine, return_all=False):

    theta_value = 120.

    data = np.ascontiguousarray(data, dtype='f8')

    print('Preparing data')
    X = prepare_data(data, affine, perc=(2, 98), mult=300., enh=3.)
    Ivol, Jvol, Kvol, W = X_to_many_3D(X, data.shape)

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

    mask_pg = mask_lt(data_labels_pg, centroids_pg, bk_thr=300)
    # mask_pgo = binary_opening(1 - mask_pg, iterations=5).astype('i4')
    dt_pg = distance_transform_edt(1 - mask_pg)

    ignore = True
    if not ignore:
        print('Applying Berserker 2 on X (i, j, k, 3w)')
        t = time()
        data_labels, centroids = berserker2_to_3D_labels(X, theta_value, data_dims=data.shape, verbose=False)
        print('Duration ', time() - t)
        print()

        print_centroids(centroids)

        mask = mask_lt(data_labels, centroids, bk_thr=300)
        dt = distance_transform_edt(1 - mask)

        print('Applying Berserker 2 on X (i, j, k, 3w - 2gradW)')
        t = time()
        data_labels_ng, centroids_ng = berserker2_to_3D_labels(Xng, theta_value, data_dims=data.shape, verbose=False)
        print('Duration ', time() - t)
        print()

        print_centroids(centroids_ng)

        mask_ng = mask_lt(data_labels_ng, centroids_ng, bk_thr=300)
        mask_ngo = binary_opening(1 - mask_ng, iterations=5).astype('i4')

        dt_ng = distance_transform_edt(1 - mask_ng)

        dt_avg = 0.5 * dt_pg + 0.5 * dt
        dt_avg2 = gaussian_filter(dt_avg, sigma=1)
        dt_avg2 = dt_avg


    weights = dt_pg.copy()
    weights[dt_pg > 5] = 5
    weights_tmp = weights.copy()
    weights = gaussian_filter(weights, sigma=1)
    weights_tmp2 = weights.copy()
    weights = weights * W

    ignore = True
    if not ignore:
        # kernel = np.ones((3, 3, 3))
        kernel = generate_binary_structure(3, 1)
        data_labels, num_labels = label(dt_pg > 5, structure=kernel)
        favorite = analyze_labels(data_labels, num_labels)

        locus = np.zeros(data.shape)
        locus[data_labels == favorite] = 1

        print('Re-weighting')
        dist_from_focused = distance_transform_edt(1 - locus)
        dist_from_focused[dist_from_focused > 10] = 10.0

        dist_from_focused = (dist_from_focused.max() - dist_from_focused)
        dist_from_focused = dist_from_focused + 10 * (1 - mask_pg) # + 1. * (W / 90.)

        dff_ps = patch_sums(dist_from_focused, (5, 5, 5))

        from dipy.viz import horizon
        aff = np.eye(4)
        horizon(images=[(dt_pg, aff, 'dt_pg'),
                        (dt, aff, 'dt'),
                        (dt_ng, aff, 'dt_ng'),
                        (dt_avg, aff, 'dt_avg'),
                        (dist_from_focused, aff, 'dff'),
                        (locus, aff, 'locus'),
                        (Wgrad, aff, 'Wgrad'),
                        (dff_ps, aff, 'dff_ps'),
                        (1 - mask_pg, aff, '1 - mask_pg'),
                        (1 - mask, aff, '1 - mask'),
                        (1 - mask_ng, aff, '1 - mask_ng'),
                        (W * locus, aff, 'W * locus'),
                        (W, aff, 'W')])

        Xd = X.copy()
        Xd[:, 3] = 30 * dist_from_focused.ravel()

        print('Applying Berserker 2 on Xd (re-weighted)')
        t = time()
        data_labels_d, centroids_d = berserker2_to_3D_labels(Xd, theta_value, data_dims=data.shape, verbose=False)
        print('Duration ', time() - t)
        print()


        weights = (dist_from_focused.max() - dist_from_focused)
        weights_from_focused = (weights / weights.max()) ** 4
        weighted_image =  weights_from_focused * W

    Xw = X.copy()
    Xw[:, 3] = weights.ravel()

    print('Applying Berserker 2 on Xw (re-weighted)')
    t = time()
    data_labels_w, centroids_w = berserker2_to_3D_labels(Xw, theta_value * 5, data_dims=data.shape, verbose=False)
    print('Duration ', time() - t)
    print()

    print_centroids(centroids_w)

    # TODO: given that there are only a few labels here
    # it would be interesting if we could edt them individualy
    # and check amount of volume change

    # 5 works but 6 also in some cases such as HH-1211
    mask_w = mask_lt(data_labels_w, centroids_w, bk_thr= 5 * 300)
    imask_w = 1 - mask_w

    imask_w_tmp = imask_w.copy()

    # TODO: this part was useful only for HH-1211 but reduces quality of mask
    # ps_iw = patch_sums(imask_w, (5, 5, 5))
    dt_iw = distance_transform_edt(imask_w)

    imask_w2 = np.zeros(data.shape)
    # 110 used here due to 125
    # imask_w2[ps_iw > 110] = 1
    imask_w2[dt_iw > 2.5] = 1

    # TODO: create an edt based transition between imask_w2 and imask_w
    # or do the same directly from the gradient mag image
    
    kernel = generate_binary_structure(3, 1)
    # imask_wb = binary_opening(imask_w, structure=kernel, iterations=2).astype('i4')
    # imask_w = imask_wb    
    data_labels, num_labels = label(imask_w2, structure=kernel)
    favorite = analyze_labels(data_labels, num_labels)

    final_mask = np.zeros(data.shape)
    final_mask[data_labels == favorite] = 1

    aff = np.eye(4)
    images=[(final_mask, aff, 'final_mask'),
            # (ps_iw, aff, 'ps_iw'),
            (dt_iw, aff, 'dt_iw'),
            (imask_w, aff, 'imask_w'), 
            (imask_w2, aff, 'imask_w2'), 
            (data_labels_w, aff, 'dl_w'),
            (weights_tmp, aff, 'weights_tmp'),
            (weights_tmp2, aff, 'weights_tmp2'),
            (weights, aff, 'weights'),
            (dt_pg, aff, 'dt_pg'),
            (mask_pg, aff, 'dl_w'),
            (Wgrad, aff, 'Wgrad'),
            (3 * W + 2 * Wgrad, aff, '3W + 2Wgrad'),
            (W, aff, 'W')]

    from dipy.viz import horizon
    horizon(images=images)
    t = 1
    return images


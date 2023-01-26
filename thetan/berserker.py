from time import time
import numpy as np
from thetan.theta import theta_g1, reassign_uncertain_update_labels, centroids_from_labels
from thetan.theta import theta_means, candidate_labels_c, candidate_labels_outliers_c


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
              manage_outliers={'mode':'min_size', 'size_limit':60}):


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
    c_centroids, c_labels, _, _ = theta_g1(initial_centroids, theta=theta_value, metric=metric
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

        info_dict['Manage Outliers'] = t18 - t17

        print('Final Shape:', even_centroids_ex2.shape)


    print('Times')
    print(info_dict)
    t = 1

    if all_out:
        return even_centroids_ex2, updated_labels_ex2, initial_centroids, even_centroids, even_centroids_ex, p_centroids, p_labels, even_centroids2

    else:
        return even_centroids_ex2

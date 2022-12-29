from time import time
import numpy as np
from thetan.theta import theta_g1, candidate_labels, reassign_uncertain_update_labels, centroids_from_labels
from thetan.theta import theta_means


def check_module_load():

    print('yep')


def berserker(points, theta_value, all_out=True, verbose=True):

    np.random.shuffle(points)

    t1 = time()

    initial_centroids, initial_labels, _, _ = theta_g1(points, theta=theta_value,
                                                       all_out=True)

    t2 = time()

    print('ic', initial_centroids.shape)

    t3 = time()
    c_centroids, c_labels, _, _ = theta_g1(initial_centroids, theta=theta_value,
                                           all_out=True)

    t4 = time()

    t5 = time()

    nice, candidates, _clusters, updated_labels = candidate_labels(c_centroids, c_labels, initial_labels)

    t6 = time()

    even_centroids = np.delete(initial_centroids, candidates, axis=0)
    print('ec', even_centroids.shape)

    t7 = time()

    updated_labels2 = reassign_uncertain_update_labels(points, even_centroids,
                                                       updated_labels, nice,
                                                       metric='euclidean')

    t8 = time()

    even_centroids_ex = centroids_from_labels(points, updated_labels2)

    t9 = time()

    # This second shuffle resolves problems with centroids between clusters
    np.random.shuffle(points)

    t10 = time()

    p_centroids, p_labels, _, _ = theta_means(points, even_centroids_ex, theta=theta_value,
                                              all_out=True)

    t11 = time()
    print('p_centroids', p_centroids.shape)

    t12 = time()
    c_centroids2, c_labels2, _, _ = theta_g1(p_centroids, theta=theta_value, # + 0.1 * theta_value,
                                             all_out=True)

    t13 = time()

    nice2, candidates2, _clusters2, updated_labels2x = candidate_labels(c_centroids2, c_labels2, p_labels)

    t14 = time()

    even_centroids2 = np.delete(p_centroids, candidates2, axis=0)

    t15 = time()
    print('ec2 ', even_centroids2.shape)

    t16 = time()
    updated_labels_ex2 = reassign_uncertain_update_labels(points, even_centroids2,
                                                          updated_labels2x, nice2,
                                                          metric='euclidean')
    t17 = time()

    even_centroids_ex2 = centroids_from_labels(points, updated_labels_ex2)
    t18 = time()
    print('ec2x ', even_centroids_ex2.shape)


    # Peripheral
    # Solution 1: threshold on cluster sizes and reassign

    unique_labels_ex2, unique_counts_ex2 = np.unique(updated_labels_ex2,
                                                     return_counts=True)
    

    # Solution 2: find close clusters - merge smaller to larger
    # even_centroids_ex3, even_labels_ex3, _, _ = theta_g1(even_centroids_ex2, theta=theta_value + 0.3 * theta_value,
    #                                                      all_out=True)


    # Solution 3: study and plot the labels of the centroid indices. Any patterns?

    # Solution 4: after theta_means remove all new clusters. So shape at the end should be equal to even_centroids_ex


    # if even_centroids_ex2.shape[0] == 101:

    print('Times ', t2 - t1, t4 - t3, t6 - t5, t7 - t6, t8 - t7, t9 - t8, t10 - t9, t11 - t10, t13 - t12, t14 - t13, t15 - t14, t17 - t16, t18 - t17)

    if all_out:
        return even_centroids_ex2, updated_labels_ex2, initial_centroids, even_centroids, even_centroids_ex, p_centroids, p_labels, even_centroids2

    else:
        return even_centroids_ex2

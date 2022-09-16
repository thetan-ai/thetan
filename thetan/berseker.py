import numpy as np
from thetan.theta import reorder_labels, candidate_labels, theta_g1, theta_means, reassign_uncertain_update_labels


def centroids_from_labels(points, labels):

    new_centroids = []
    for i, l in enumerate(range(labels.max() + 1)):
        # print(i)
        avg = np.average(points[labels==l], axis=0)
        new_centroids.append(avg)

    t = 0
    return np.ascontiguousarray(new_centroids)


def berserker(points, theta_value, all_out=True, verbose=True):

    np.random.shuffle(points)

    t = 2

    initial_centroids, initial_labels, _, _ = theta_g1(points, theta=theta_value,
                                                       all_out=True)
    print('ic', initial_centroids.shape)
    c_centroids, c_labels, _, _ = theta_g1(initial_centroids, theta=theta_value,
                                           all_out=True)

    nice, candidates, _clusters, updated_labels = candidate_labels(c_centroids, c_labels, initial_labels)
    even_centroids = np.delete(initial_centroids, candidates, axis=0)
    print('ec', even_centroids.shape)

    t = 0

    updated_labels2 = reassign_uncertain_update_labels(points, even_centroids,
                                                       updated_labels, nice,
                                                       metric='euclidean')
    t = 1

    even_centroids_ex = centroids_from_labels(points, updated_labels2)

    t = 2

    # This second shuffle resolves problems with centroids between clusters
    np.random.shuffle(points)

    p_centroids, p_labels, _, _ = theta_means(points, even_centroids_ex, theta=theta_value,
                                              all_out=True)

    print('p_centroids', p_centroids.shape)

    c_centroids2, c_labels2, _, _ = theta_g1(p_centroids, theta=theta_value, # + 0.1 * theta_value,
                                             all_out=True)

    nice2, candidates2, _clusters2, updated_labels2x = candidate_labels(c_centroids2, c_labels2, p_labels)
    even_centroids2 = np.delete(p_centroids, candidates2, axis=0)
    print('ec2 ', even_centroids2.shape)

    updated_labels_ex2 = reassign_uncertain_update_labels(points, even_centroids2,
                                                          updated_labels2x, nice2,
                                                          metric='euclidean')

    even_centroids_ex2 = centroids_from_labels(points, updated_labels_ex2)
    print('ec2x ', even_centroids_ex2.shape)

    # if even_centroids_ex2.shape[0] == 101:



    t = 5

    if all_out:
        return
    even_centroids_ex2, updated_labels_ex2, initial_centroids, even_centroids, even_centroids_ex, p_centroids, p_labels, even_centroids2


    else:
        return even_centroids_ex2

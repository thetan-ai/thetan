import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import normalize


def accuracy_munkres(true_labels, pred_labels):
    """Score a clustering against true labels with matched F1 scores.

    Cluster numbers are arbitrary, so each true class is first paired with
    one predicted cluster. The pairing is the one-to-one matching with the
    highest total F1 score, found with the Hungarian (Munkres) algorithm.
    The score is the mean F1 of the matched pairs. It is not the fraction
    of points that were clustered correctly.

    Parameters
    ----------
    true_labels : array_like, shape (n_samples,)
        True class of each point.
    pred_labels : array_like, shape (n_samples,)
        Predicted cluster of each point.

    Returns
    -------
    f1_hungarian(f_matrix(pred_labels, true_labels)) : float
        Mean F1 score of the matched pairs, between 0 and 1. When there
        are more clusters than classes, or more classes than clusters, the
        ones left without a partner are not counted.
    """
    return f1_hungarian(f_matrix(pred_labels, true_labels))


def f_matrix(labels_true, labels_pred):
    """F1 score of every pair of true class and predicted cluster.

    Parameters
    ----------
    labels_true : array_like, shape (n_samples,)
        True class of each point.
    labels_pred : array_like, shape (n_samples,)
        Predicted cluster of each point.

    Returns
    -------
    f1 : ndarray, shape (n_classes, n_clusters)
        F1 score of each class and cluster pair, rounded to 3 decimals.
        Pairs with no points in common get 0.
    """
    cont_mat = contingency_matrix(labels_pred=labels_pred, labels_true=labels_true)
    precision = normalize(cont_mat, norm='l1', axis=0)
    recall = normalize(cont_mat, norm='l1', axis=1)
    som = precision + recall
    f1 = np.round(np.divide((2 * recall * precision), som, out=np.zeros_like(som), where=som!=0), 3)
    return f1


def f1_hungarian(f1):
    """Mean F1 score of the best one-to-one matching of rows to columns.

    Parameters
    ----------
    f1 : ndarray, shape (n_rows, n_cols)
        F1 matrix from f_matrix.

    Returns
    -------
    fscore : float
        Mean of the matched entries of f1. The matching has
        min(n_rows, n_cols) pairs and is the one with the highest total.
    """
    inverse = 1 - f1
    rows, cols = linear_sum_assignment(inverse)
    indices = list(zip(rows, cols))
    fscore = sum([f1[i] for i in indices])/len(indices)
    return fscore

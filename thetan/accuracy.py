import numpy as np
from sklearn import metrics
from munkres import Munkres

# def calculate_accuracy_munkres(y_true, y_pred):
#     """
#     Calculate clustering accuracy using the Munkres module.

#     Args:
#     - y_true: List or array of ground truth labels.
#     - y_pred: List or array of predicted labels.

#     Returns:
#     - accuracy: Accuracy of the clustering.
#     """
#     # Get unique labels
#     labels_true = np.unique(y_true)
#     labels_pred = np.unique(y_pred)

#     # Build contingency matrix
#     cost_matrix = np.zeros((len(labels_true), len(labels_pred)), dtype=int)
#     for i, true_label in enumerate(labels_true):
#         for j, pred_label in enumerate(labels_pred):
#             cost_matrix[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

#     # Convert the cost matrix for minimization (Munkres minimizes by default)
#     munkres = Munkres()
#     cost_matrix_neg = -cost_matrix.tolist()  # Negate for maximization
#     indexes = munkres.compute(cost_matrix_neg)  # Optimal assignments

#     # Calculate accuracy
#     total_correct = sum(cost_matrix[row][col] for row, col in indexes)
#     accuracy = total_correct / len(y_true)
#     return accuracy

# # Example Usage
# y_true = np.array([0, 0, 1, 1, 2, 2])  # Ground truth labels
# y_pred = np.array([1, 1, 0, 0, 2, 2])  # Predicted labels (cluster IDs)

# accuracy = calculate_accuracy_munkres(y_true, y_pred)
# print(f"Clustering Accuracy: {accuracy:.2f}")




from munkres import Munkres
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import normalize


def accuracy_munkres(true_labels, pred_labels):
    return f1_hungarian(f_matrix(pred_labels, true_labels))


def f_matrix(labels_true, labels_pred):
    # Calculate F1 matrix
    cont_mat = contingency_matrix(labels_pred=labels_pred, labels_true=labels_true)
    precision = normalize(cont_mat, norm='l1', axis=0)
    recall = normalize(cont_mat, norm='l1', axis=1)
    som = precision + recall
    f1 = np.round(np.divide((2 * recall * precision), som, out=np.zeros_like(som), where=som!=0), 3)
    return f1

def f1_hungarian(f1):
    m = Munkres()
    inverse = 1 - f1
    indices = m.compute(inverse.tolist())
    fscore = sum([f1[i] for i in indices])/len(indices)
    return fscore

# # Example usage
# pred = [0, 1, 2, 0, 1, 2, 0, 1, 2]
# true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
# f1_hungarian(f_matrix(pred, true))

# pred = [0, 0, 0, 2, 2, 1, 1]
# true = [0, 0, 0, 1, 1, 2, 2]
# res = f1_hungarian(f_matrix(pred, true))

# print(res)


# pred = [0, 0, 0, 2, 2, 1, 1, ]
# true = [0, 0, 0, 1, 1, 2, 3]
# res = f1_hungarian(f_matrix(pred, true))

# print(res)
# def cluster_metric(label, pred):
#     nmi = metrics.normalized_mutual_info_score(label, pred)
#     ari = metrics.adjusted_rand_score(label, pred)
#     pred_adjusted = get_y_preds(label, pred, len(set(label)))
#     acc = metrics.accuracy_score(pred_adjusted, label)
#     print(
#         "[Clustering Result]: ACC = {:.2f}, NMI = {:.2f}, ARI = {:.2f}".format(
#             acc * 100, nmi * 100, ari * 100
#         )
#     )

def accuracy_tac(label, pred):
    pred_adjusted = get_y_preds(label, pred, len(set(label)))
    acc = metrics.accuracy_score(pred_adjusted, label)
    return acc

def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    """
    confusion_matrix = metrics.confusion_matrix(
        y_true, cluster_assignments, labels=None
    )
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred


# pred = [0, 0, 0, 2, 2, 1, 1]
# true = [0, 0, 0, 1, 1, 2, 3]
# res = accuracy(pred, true)

# t = 1
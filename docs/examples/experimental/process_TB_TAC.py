import torch
import faiss
import warnings
import numpy as np
import torch.nn.functional as F
from thetan.berserker import berserker
from thetan.theta import assign
import numpy as np
from sklearn import metrics
from munkres import Munkres

warnings.simplefilter("ignore")

def cluster_metric(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    pred_adjusted = get_y_preds(label, pred, len(set(label)))
    acc = metrics.accuracy_score(pred_adjusted, label)
    print(
        "[Clustering Result]: ACC = {:.2f}, NMI = {:.2f}, ARI = {:.2f}".format(
            acc * 100, nmi * 100, ari * 100
        )
    )


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


def kmeans(X, cluster_num):
    print("Perform K-means clustering...")
    d = X.shape[1]
    kmeans = faiss.Kmeans(d, cluster_num, gpu=True, spherical=True, niter=300, nredo=20)
    X = X.astype(np.float32)
    kmeans.train(X)
    D, I = kmeans.index.search(X, 1)
    I = I.reshape(-1)
    print("K-means clustering done.")
    return I

def tb(X, cluster_num, theta_value, metric):
    print('Perform thetan clustering...')
    d = X.shape[1]
    X = np.ascontiguousarray(X, dtype='f8')
    centroids, data_labels = berserker(X, theta=theta_value, metric = metric)
    print('Number of clusters:', len(centroids))
    uniq, sizes = np.unique(data_labels, return_counts=True)
    niceK = cluster_num
    biggest_k = np.argsort(sizes)[-niceK:]
    hd_centroids = np.ascontiguousarray(centroids[biggest_k].copy())
    labels2 = assign(X, hd_centroids)
    return labels2

if __name__ == "__main__":
    dataset = "ImageNet"  # 　["CIFAR-10", "CIFAR-20", "STL-10", "ImageNet-10", "ImageNet-Dogs", "DTD", "UCF101", "ImageNet"]
    tau = 0.005
    if dataset == "CIFAR-10" or dataset == "STL-10" or dataset == "ImageNet-10":
        cluster_num = 10
    elif dataset == "CIFAR-20":
        cluster_num = 20
    elif dataset == "ImageNet-Dogs":
        cluster_num = 15
    elif dataset == "DTD":
        cluster_num = 47
    elif dataset == "UCF101":
        cluster_num = 101
    elif dataset == "ImageNet":
        cluster_num = 1000
    else:
        raise NotImplementedError

    images_embedding = np.load("./data/" + dataset + "_image_embedding_test.npy")
    images_embedding = images_embedding / np.linalg.norm(
        images_embedding, axis=1, keepdims=True
    )
    labels = np.loadtxt("./data/" + dataset + "_labels_test.txt")

    nouns_embedding = np.load("./data/" + dataset + "_filtered_nouns_embedding.npy")
    nouns_embedding = nouns_embedding / np.linalg.norm(
        nouns_embedding, axis=1, keepdims=True
    )

    nouns_embedding = torch.from_numpy(nouns_embedding).cuda().half()
    nouns_num = nouns_embedding.shape[0]
    images_embedding = torch.from_numpy(images_embedding).cuda().half()
    image_num = images_embedding.shape[0]

    retrieval_embeddings = []
    batch_size = 8192
    for i in range(image_num // batch_size + 1):
        start = i * batch_size
        end = start + batch_size
        if end > image_num:
            end = image_num
            images_batch = images_embedding[start:end]
        similarity = torch.matmul(images_embedding[start:end], nouns_embedding.T)
        similarity = torch.softmax(similarity / tau, dim=1)
        retrieval_embedding = (similarity @ nouns_embedding).cpu()
        retrieval_embeddings.append(retrieval_embedding)
        if i % 50 == 0:
            print(f"[Completed {i * batch_size}/{image_num}]")
    retrieval_embedding = torch.cat(retrieval_embeddings, dim=0).cuda().half()
    retrieval_embedding = F.normalize(retrieval_embedding, dim=1).cpu().numpy()
    images_embedding = images_embedding.cpu().numpy()

    concat_embedding = np.concatenate([images_embedding, retrieval_embedding], axis=1)
    # preds = kmeans(concat_embedding, cluster_num)
    # cluster_metric(labels, preds)

    # print('ClIP (kmeans)')
    # preds = kmeans(images_embedding, cluster_num)
    # cluster_metric(labels, preds)

    # print('TAC (TB)')
    # theta_value = 0.75
    # metric = 'euclidean'
    # preds = tb(concat_embedding, cluster_num, theta_value, 'euclidean')
    # cluster_metric(labels, preds)

    print('TAC with UMAP')
    theta_value = 0.22
    from umap import UMAP
    print('UMAP')
    print('theta:', theta_value)
    reducer = UMAP(n_components=30, random_state = 42)
    concat_embedding = reducer.fit_transform(concat_embedding)
    print('Finished Dim Red')
    print('concat embedding shape:', concat_embedding.shape)
    preds = tb(concat_embedding, cluster_num, theta_value, 'euclidean')
    cluster_metric(labels, preds)

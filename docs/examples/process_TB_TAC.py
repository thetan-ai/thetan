import torch
import faiss
import warnings
import numpy as np
import torch.nn.functional as F
from eval_utils import cluster_metric
from thetan.berserker import berserker2_clean
from thetan.theta import assign


warnings.simplefilter("ignore")


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
    centroids, data_labels = berserker2_clean(X, theta_value=theta_value, metric = metric)
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
    reducer = UMAP(n_components=700, random_state = 42)
    concat_embedding = reducer.fit_transform(concat_embedding)
    print('Finished Dim Red')
    print('concat embedding shape:', concat_embedding.shape)
    preds = tb(concat_embedding, cluster_num, theta_value, 'euclidean')
    cluster_metric(labels, preds)

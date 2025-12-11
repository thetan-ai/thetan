from thetan.tbknn import tbknn
from thetan import berserker
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from time import time
import matplotlib.pyplot as plt

def small_dataset():
    blob1 = np.random.rand(10,2)
    blob2 = np.random.rand(10,2) + 0.5
    X = np.vstack([blob1, blob2])
    query_point = np.array([0.5,0.5])
    return X, query_point

def large_dataset():
    X,_ = make_blobs(n_samples=50000, n_features=128, centers=50, random_state=42)
    query_index = np.random.randint(0, X.shape[0])
    query_point = X[query_index] + np.random.normal(0,0.01, X.shape[1])
    return X, query_point

def plot_knn_comparison(X, query_point, sk_indices, sk_dists, tb_indices, tb_dists, k, reduction_method='pca'):
    if X.shape[1] > 2:
        print(f"Applying {reduction_method}.")

        if reduction_method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            X_2d = reducer.fit_transform(X)
            query_2d = reducer.transform(query_point.reshape(1,-1))[0]
        
        elif reduction_method == 'tsne':
            from sklearn.manifold import TSNE
            X_with_query = np.vstack([X, query_point])
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            X_all_2d = reducer.fit_transform(X_with_query)
            X_2d = X_all_2d[:-1]
            query_2d = X_all_2d[-1]
        else:
            raise ValueError(f"Invalid reduction method: {reduction_method}")
        
        title_suffix = f"({reduction_method})"

    else:
        X_2d = X
        query_2d = query_point
        title_suffix = ""

    fig, axes = plt.subplots(1,2,figsize=(14,6))

    ax1 = axes[0]
    ax1.scatter(X_2d[:, 0], X_2d[:,1], c='lightgray', s=20, alpha=0.6, label='Data points')
    ax1.scatter(X_2d[sk_indices, 0], X_2d[sk_indices,1], c='blue', s=100,edgecolors='darkblue',linewidths=2,label=f'Top {k} neighbors')
    ax1.scatter(query_2d[0], query_2d[1], c='red', s=200, marker='*',edgecolors='darkred',linewidths=2,label='Query point')
    
    for idx in sk_indices:
        ax1.plot([query_2d[0], X_2d[idx,0]], [query_2d[1], X_2d[idx, 1]], 'b--',alpha=0.5,linewidth=1)
    
    ax1.set_title(f'Scikit-learn KNN{title_suffix}')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.scatter(X_2d[:, 0], X_2d[:, 1], c='lightgray', s=20, alpha=0.6, label='Data points')
    ax2.scatter(X_2d[tb_indices,0], X_2d[tb_indices,1], c='darkgreen', linewidths=2, label=f'Top {k} neighbors')
    ax2.scatter(query_2d[0], query_2d[1], c='red', s=200, marker='*', edgecolors='darkred', linewidths=2, label='Query point')

    for idx in tb_indices:
        ax2.plot([query_2d[0], X_2d[idx, 0]],[query_2d[1], X_2d[idx, 1]], 'g--', alpha=0.5, linewidth=1)
    
    ax2.set_title(f'Theta-KNN{title_suffix}')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('knn_comparison.png')
    plt.show()


def compare_results(verbose=False):
    X, query_point = small_dataset()
    # X, query_point = large_dataset()
    k=5
    print('Running scikit-learn kmeans nearest neighbors')
    t1 = time()
    sklearn_knn = NearestNeighbors(n_neighbors=k, algorithm='brute')
    sklearn_knn.fit(X)
    sk_dists, sk_indices = sklearn_knn.kneighbors([query_point])
    t2 = time()
    print('Time taken scikit-learn: ', t2-t1)

    sk_indices = sk_indices[0]
    sk_dists = sk_dists[0]

    print('Running theta-knn nearest neighbors')
    t1 = time()
    tb_indices, tb_dists, _ = tbknn(X, query_point, k=k, theta=0.5, nprobe=2) # For small dataset
    # tb_indices, tb_dists, _ = tbknn(X, query_point, k=k, theta=200.0, nprobe=1) # For big dataset
    t2 = time()
    print('Time taken theta-knn: ', t2-t1)

    print('Query point: ', query_point)
    print('Scikit-learn indices: ', sk_indices)
    print('TBKNN indices: ', tb_indices)
    print('Scikit-learn distances: ', sk_dists)
    print('TBKNN distances: ', tb_dists)

    np.testing.assert_equal(len(tb_indices), k)
    np.testing.assert_allclose(tb_dists, sk_dists)
    np.testing.assert_array_equal(tb_indices, sk_indices)

    if verbose:
        plot_knn_comparison(X, query_point, sk_indices, sk_dists, tb_indices, tb_dists, k)

if __name__ == '__main__':
    compare_results(verbose=True)
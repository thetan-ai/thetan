import numpy as np
from thetan.berserker import berserker2_clean as tb
from thetan.theta import assign


dpath = '/home/elef/Data/CLIP/'

fitrain = dpath + 'CIFAR-10_image_embedding_train.npy'

fitest = dpath + 'CIFAR-10_image_embedding_test.npy'

fconcat = dpath + 'concat_embedding.npy'

fgt_train = dpath + 'CIFAR-10_labels_train.txt'
fgt_test = dpath + 'CIFAR-10_labels_test.txt'

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import pairwise_distances

def distillation(X):
    # Simulated Data
    X = np.random.rand(500, 10)  # 500 samples, 10 features

    # Teacher Model (K-Means)
    teacher = KMeans(n_clusters=5, random_state=42)
    tf = teacher.fit(X)
    soft_labels = teacher.transform(X)  # Soft assignments (distance to centroids)

    # Student Model (Neural Network)
    student = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
    student.fit(X, np.argmax(-soft_labels, axis=1))  # Mimic teacher's clusters

    # Predictions
    student_clusters = student.predict(X)
    print("Student Clusters:", np.unique(student_clusters))


def normalization():
    pass

def keep_largest(X, centroids, labels, nicest=10):
    uniq, sizes = np.unique(labels, return_counts=True)
    niceK = nicest
    biggest_k = np.argsort(sizes)[-niceK:]
    hd_centroids = np.ascontiguousarray(centroids[biggest_k].copy())
    labels2 = assign(X, hd_centroids)
    return labels2, sizes



from sklearn import metrics
from munkres import Munkres


def evaluate(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    print(
        "[Clustering Result]: NMI = {:.2f}, ARI = {:.2f}".format(nmi * 100, ari * 100)
    )
    return nmi, ari


X = np.load(fitest).astype('f8')

# X = X / np.linalg.norm(X, axis=1, keepdims=True)

gt_labels = np.loadtxt(fgt_test)

theta = 0.44
metric = 'arccos'

centroids, labels = tb(X, theta, metric=metric)

evaluate(gt_labels, labels)
print('#Clusters', centroids.shape)

labels2, sizes = keep_largest(X, centroids, labels, 10)
evaluate(gt_labels, labels2)

uniq2, sizes2 = np.unique(labels2, return_counts=True)

labels3, sizes = keep_largest(X, centroids, labels, 18)
evaluate(gt_labels, labels3)

uniq3, sizes3 = np.unique(labels3, return_counts=True)

from pylab import figure, plot, show, legend

figure()
plot(sorted(sizes)[::-1], label='sizes')
plot(sorted(sizes2)[::-1], label='sizes2')
plot(sorted(sizes3)[::-1], label='sizes3')
legend()
show()

t = 1
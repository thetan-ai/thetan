import numpy as np
from keras.datasets import mnist
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from sklearn.metrics.cluster import adjusted_rand_score, normalized_mutual_info_score
from thetan.berserker import berserker2, tb_randomized, berserker2_clean
from thetan.theta import theta_g1, assign, centroids_from_labels, distance_matrix


def report_accuracy(y_true, y_pred):

    accs = accuracy_score(y_true, y_pred,
                          normalize=True,
                          sample_weight=None)
    return accs

def plot_centroid_images(centroids):
    num_centroids = centroids.shape[0]
    f, axarr = plt.subplots(1, num_centroids)

    for i in range(num_centroids):
        axarr[i].imshow(centroids[i].reshape(28, 28), cmap='gray')

    plt.show()


def plot_centroid_images_obo(centroids):
    for i in range(centroids.shape[0]):
        plt.figure()
        plt.title('Label ' + str(i))
        plt.imshow(centroids[i].reshape(28, 28))
        plt.show()


def classification_task():

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # x = np.concatenate((x_train, x_test))
    # y = np.concatenate((y_train, y_test))  ## labels (used for verification)
    # x = x.reshape((x.shape[0], -1))
    # x = np.divide(x, 255.)
    # n_clusters = len(np.unique(y))
    # print(n_clusters)


    # from sklearn import datasets, manifold

    # digits = datasets.load_digits()
    # X, y = digits.data, digits.target
    # n_samples, n_features = X.shape



    x_train_cp = np.ascontiguousarray(x_train.reshape(60000, 28*28), dtype='f8')
    x_test_cp = np.ascontiguousarray(x_test.reshape(10000, 28*28), dtype='f8')

    # centroids_train_avgs = centroids_from_labels(x_train_cp, x_test.astype('i8'))
    # centroids, labels = berserker2(x_train_cp, theta_value=1900., metric='euclidean', all_out=False, verbose=True)

    centroids_train = []
    corresponding_labels_train = []
    # L2
    # theta_value = 1800.
    # theta_value = 1700.
    # metric = 'euclidean'

    # Dot Product
    # theta_value = 100.
    # metric = 'dot_prod'

    # L1
    theta_value = 18000. # AC: Train 0.9476 Test 0.935
    theta_value = 15000. # AC: Train 0.9739 Test 0.943
    theta_value = 17000. # AC: Train 0.9554 Test 0.935
    metric = 'l1_norm'


    for y_t in range(10):
        x_tmp = np.ascontiguousarray(x_train_cp[y_train == y_t].copy())
        centroids, labels = berserker2(x_tmp, theta_value=theta_value, metric=metric, all_out=False, verbose=False)
        # centroids, labels, sot = tb_randomized(x_tmp, theta=theta_value, metric=metric, num_shuffles=10)
        centroids_train.append(centroids)
        corresponding_labels_train.extend(centroids.shape[0] * [y_t])
        print(y_t, centroids.shape)


    centroids_train = np.ascontiguousarray(np.concatenate(centroids_train))

    # dot product needs work - it creates NaNs for ber
    # centroids, labels = berserker2(x, theta_value=80, metric='dot_prod', all_out=False, verbose=True)

    # TS does not create NaN but not helping much
    # centroids, labels = theta_g1(x, theta=70, metric='dot_prod', all_out=False, verbose=False)
    print('Number of centroids found ', centroids_train.shape[0])

    t = 1

    # plot_centroid_images(centroids)


    t = 2

    labels_x_train = assign(x_train_cp, centroids_train, metric=metric)

    y_train_pred = []
    for i, l in enumerate(labels_x_train):
        y_train_pred.append(corresponding_labels_train[l])
    y_train_pred = np.ascontiguousarray(y_train_pred, dtype='i8')

    labels_x_test = assign(x_test_cp, centroids_train, metric=metric)

    y_test_pred = []
    for i, l in enumerate(labels_x_test):
        y_test_pred.append(corresponding_labels_train[l])
    y_test_pred = np.ascontiguousarray(y_test_pred, dtype='i8')


    ac_train = accuracy_score(y_train, y_train_pred)

    ac_test = accuracy_score(y_test, y_test_pred)

    print('Reporting scores ...')
    print('AC Train ', ac_train, ' AC Test ', ac_test)


def test_cosine_distance():

    # A = np.random.rand(20, 3) - 0.5
    # X = np.concatenate((A, A))
    A = np.array([[1.414213/2, 1.414213/2, 0]])
    B = np.array([[1., 0, 0]])

    D = distance_matrix(A, B, metric='cosine')
    print(D)

    A = np.array([[1.414213/2, 1.414213/2, 0]])
    B = np.array([[0., 1, 0]])

    D = distance_matrix(A, B, metric='cosine')
    print(D)

    A = np.array([[0., -1, 0]])
    B = np.array([[0., 1, 0]])

    D = distance_matrix(A, B, metric='cosine')
    print(D)


def test_arccos_distance():

    A = np.array([[1.414213/2, 1.414213/2, 0]])
    B = np.array([[1., 0, 0]])

    D = distance_matrix(A, B, metric='arccos')
    print(D, np.rad2deg(D))

    A = np.array([[1.414213/2, 1.414213/2, 0]])
    B = np.array([[0., 1, 0]])

    D = distance_matrix(A, B, metric='arccos')
    print(D, np.rad2deg(D))

    A = np.array([[0., -1, 0]])
    B = np.array([[0., 1, 0]])

    D = distance_matrix(A, B, metric='arccos')
    print(D, np.rad2deg(D))

    t = 1
    # centroids, labels = berserker2_clean(X, 4, metric='euclidean')
    # assert centroids.shape == (2, 3)

    # assert np.sum(labels==0) == 20
    # assert np.sum(labels==1) == 20

    # for theta_value in np.linspace(0, 2, 40):
    #     print('Cosine Distance ', theta_value)
    #     t0 = time()
    #     tb_centroids_cosd, tb_labels_cosd = berserker2_clean(X, theta_value=0.5, metric='cosine')
    #     print('Duration ', time() - t0)
    #     print(tb_centroids_cosd.shape)
    #     print()

    # for theta_value in np.linspace(0, np.pi, 40):
    #     print('Arccos Distance ', theta_value)
    #     t0 = time()
    #     tb_centroids_acosd, tb_labels_acosd = berserker2_clean(X, theta_value=np.pi/4, metric='arccos')
    #     print('Duration ', time() - t0)
    #     print(tb_centroids_acosd.shape)
    #     print()


    t = 1


def extensive_evaluation(x_train_cp, y_train, tb_centroids, tb_labels):

    tb_uniq, tb_sizes = np.unique(tb_labels, return_counts=True)
    tb_centroids2 = tb_centroids[np.argsort(tb_sizes)[::-1][:10]]

    tb_labels2 = assign(x_train_cp, tb_centroids2, metric=metric)

    ari = adjusted_rand_score(y_train, tb_labels)
    nmi = normalized_mutual_info_score(y_train, tb_labels)

    print('All: ARI ', ari, ' NMI ', nmi)

    ari2 = adjusted_rand_score(y_train, tb_labels2)
    nmi2 = normalized_mutual_info_score(y_train, tb_labels2)

    print('B10: ARI ', ari2, ' NMI ', nmi2)

    return ari, nmi, ari2, nmi2


if __name__ == '__main__':

    # print('Cosine')
    # test_cosine_distance()
    # print()
    # print('Arccos')
    # test_arccos_distance()
    # print()

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_cp = np.ascontiguousarray(x_train.reshape(60000, 28*28), dtype='f8')
    x_test_cp = np.ascontiguousarray(x_test.reshape(10000, 28*28), dtype='f8')

    # TODO remove asap
    x_train_cp = x_test_cp.copy()
    y_train = y_test.copy()

    print('Data shape ', x_train_cp.shape)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    x_train_cp = scaler.fit_transform(x_train_cp)
    # print(scaler.fit(data))
    # StandardScaler()
    # print(scaler.mean_)

    theta_value = 17000.
    metric = 'l1_norm'

    print('>>> L1 Norm ')
    t0 = time()
    tb_centroids, tb_labels = berserker2_clean(x_train_cp, theta_value=theta_value, metric=metric)
    print('Duration ', time() - t0)

    print(tb_centroids.shape)

    extensive_evaluation(x_train_cp, y_train, tb_centroids, tb_labels)


    # print('Cosine Distance ')
    # t0 = time()
    # tb_centroids_cosd, tb_labels_cosd = berserker2_clean(x_train_cp, theta_value=0.5, metric='cosine')
    # print('Duration ', time() - t0)

    print()
    theta_value = np.pi / 4.
    print('>>> Arccos Distance ')
    for theta_value in np.linspace(0.63, 1.2, 30):
    # for theta_value in np.linspace(0.98, 1.2, 20):

        print()
        print('theta ', theta_value)
        metric = 'arccos'

        print('Starting Berserker2 ... ')
        t0 = time()
        tb_centroids_acosd, tb_labels_acosd = berserker2_clean(x_train_cp, theta_value=theta_value, metric=metric)
        print('Duration ', time() - t0)

        print(tb_centroids_acosd.shape)

        print('Evaluation')
        extensive_evaluation(x_train_cp, y_train, tb_centroids_acosd, tb_labels_acosd)
        print()

    # tb_centroids3 = tb_centroids[np.argsort(tb_sizes)[::-1][:500]]

    # from thetan.theta import distance_matrix

    # D = distance_matrix(x_train_cp, tb_centroids3, metric=metric)

    # for theta_value in np.arange(50000, 100000, 5000):
    # # for theta_value in np.arange(10000, 20000, 500):
    #     print()
    #     print(theta_value)
    #     tb_centroids4, tb_labels4 = berserker2_clean(D, theta_value=theta_value, metric=metric)

    #     ari4 = adjusted_rand_score(y_train, tb_labels4)
    #     nmi4 = normalized_mutual_info_score(y_train, tb_labels4)

    #     print(tb_centroids4.shape, ari4, nmi4)

    #     tb_uniq4, tb_sizes4 = np.unique(tb_labels4, return_counts=True)
    #     tb_centroids5 = tb_centroids4[np.argsort(tb_sizes4)[::-1][:10]]

    #     tb_labels5 = assign(D, tb_centroids5, metric=metric)

    #     ari5 = adjusted_rand_score(y_train, tb_labels5)
    #     nmi5 = normalized_mutual_info_score(y_train, tb_labels5)

    #     print(tb_centroids5.shape, ari5, nmi5)


    t = 3
from datetime import timedelta
from robotikos.fgm import assign_to_centroids, fgn_shuffles
from robotikos.metric import (NormalizedSumAbsDiffMetric,
                              NormalizedSumSquaredDiffMetric)
from robotikos.metric import IdentityFeature
from robotikos.metricspeed import (assign_to_centroid_nl1,
                                   assign_to_centroid_nl2)
from robotikos.rsvd import bksvd
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from time import time


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def extract_hog(img, orientations=9, px_per_cell=(2, 2),
                cells_per_block=(1, 1), viz=False):
    res = hog(
        img, orientations=orientations, pixels_per_cell=px_per_cell,
        cells_per_block=cells_per_block, visualize=viz, transform_sqrt=True,
        multichannel=True)
    return res


def fsn_test_with_fgn_shuffles(x_test, centroids, metric):
    return y_pred_with_fgn_shuffles_fast(x_test, centroids, metric)


def fsn_training_with_fgn_shuffles(x_train, y_train, thresholds, metric,
                                   num_shuffles=1, mode='fgn_ma',
                                   verbose=False, level=None, num_labels=10):
    sep_data = separate_train_data(x_train, y_train, num_labels=num_labels)
    if verbose:
        print('#')
        print('FGN SHUFFLES version')
        print('Thresholds', thresholds)
    if level is None:
        level = len(thresholds)
    centroids = {}
    for s in sep_data:
        print('>> Processing label ', s)
        collect2 = fgn_shuffles(sep_data[s], thresholds, metric=metric,
                                num_shuffles=num_shuffles, mode=mode,
                                verbose=verbose)
        centroids[s] = collect2[level]['centroids']
    total_no_centroids = np.sum([len(centroids[c]) for c in centroids])
    if verbose:
        print('Total number of centroids', total_no_centroids)
    y_pred = y_pred_with_fgn_shuffles_fast(x_train, centroids, metric)
    return centroids, y_pred


def nn_feature(img, model, preprocessing=None):
    img = np.expand_dims(img, axis=0)
    # TODO: Try Keras preproc
    """
    Convert the images from RGB to BGR, then will zero-center each color 
    channel with respect to the ImageNet dataset, without scaling.
    """
    #img = preprocess_input(img)
    return model.predict(img)


def report_accuracy(y_true, y_pred):
    accs = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    return accs


def separate_train_data(x_train, y_train, num_labels=10):
    """ Separates training data to each category"""
    sep_data = {}
    for i in range(num_labels):
        y_arg = np.where(y_train == i)
        sep_data[i] = x_train[y_arg[0]]
    return sep_data


def viz_hog(img):
    fd, hog_img = extract_hog(img, viz=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(img)
    ax1.set_title('Input image')

    ax2.axis('off')
    ax2.imshow(hog_img, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()


def y_pred_with_fgn_shuffles_fast(x_features, centroids, metric):
    cnt = 0
    lengths = [len(centroids[key]) for key in centroids]
    total_centroids = np.sum(lengths)
    feature_shape = x_features[0].shape
    all_centroids = np.zeros((total_centroids,) + feature_shape)
    for key in centroids:
        for c in centroids[key]:
            all_centroids[cnt] = centroids[key][c]
            cnt += 1
    if isinstance(metric, NormalizedSumSquaredDiffMetric):
        assign_to_centroid_fast = assign_to_centroid_nl2
    elif isinstance(metric, NormalizedSumAbsDiffMetric):
        assign_to_centroid_fast = assign_to_centroid_nl1
    else:
        print('Unknown metric to use here - switching to slow version')
        assign_to_centroid_fast = assign_to_centroids
    labels = assign_to_centroid_fast(metric, x_features.astype('f4'),
                                     all_centroids.astype('f4'))
    tiled = np.repeat(np.arange(len(centroids.keys())), tuple(lengths))
    y_pred = [tiled[l] for l in labels]
    y_pred = np.array(y_pred)
    return y_pred


if __name__ == '__main__':
    dataset = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    print('Original training size: {}'.format(y_train.shape[0]))
    print('Original testing size: {}'.format(y_test.shape[0]))

    # Data normalization
    x_train_work = x_train / 255.

    """
    data_shape = x_train.shape[1:]
    # [N, 1, 3072=(32x32x3)]
    x_train_work = np.reshape(x_train, (train_sz, 1, np.prod(data_shape)))
    # [N, 3, 1024=(32x32)]
    """

    #viz_hog(x_train[0])

    """
    t = time()
    x_train_work = np.array([extract_hog(x_train_work[i])[np.newaxis, :]
                             for i in range(y_train.shape[0])])
    time_diff = str(timedelta(seconds=time() - t)).split('.')[0]
    print('Duration feature descriptor calculation on training dataset: '
          '{}'.format(time_diff))
    """

    base_model = VGG16(include_top=False, weights='imagenet',
                       input_shape=(32, 32, 3), classes=10)
    base_model.summary()
    model = Model(inputs=base_model.inputs,
                  outputs=base_model.get_layer('block5_pool').output)

    t = time()
    x_train_work = np.array([nn_feature(x_train_work[i], model)[0, 0]
                             for i in range(y_train.shape[0])])
    time_diff = str(timedelta(seconds=time() - t)).split('.')[0]
    print('Duration feature descriptor calculation on training dataset: '
          '{}'.format(time_diff))

    test_size = .2
    print('Splitting training dataset into training and validation({}/{})'
          '...'.format(int((1 - test_size) * 100), int(test_size * 100)))
    x_train_work, x_val_work, y_train, y_val = train_test_split(
        x_train_work, y_train, test_size=test_size, random_state=42,
        stratify=y_train)
    train_sz = y_train.shape[0]
    val_sz = y_val.shape[0]
    test_sz = y_test.shape[0]
    print('Split training size: {}'.format(train_sz))
    print('Split validation size: {}'.format(val_sz))

    metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())

    thresholds = [.1, .05]
    print('Thresholds: {}'.format(thresholds))

    num_shuffles = 5
    use_sobel = False
    use_bksvd = False
    k = 10
    verbose = True
    level = None

    # Low rank approximation
    x_train_work_ev = np.zeros((train_sz, 1, k))

    if use_bksvd:
        t = time()
        for i in range(train_sz):
            x_train_work_ev[i][0] = bksvd(x_train_work[i], k=k)[1]
        time_diff = str(timedelta(seconds=time() - t)).split('.')[0]
        print('Duration bksvd: {}'.format(time_diff))
        x_train_work = x_train_work_ev

    x_val_work_ev = np.zeros((val_sz, 1, k))

    if use_bksvd:
        for i in range(val_sz):
            x_val_work_ev[i][0] = bksvd(x_val_work[i], k=k)[1]
        x_val_work = x_val_work_ev

    y_train_work = y_train
    y_val_work = y_val

    feature_shape = x_train_work[0].shape

    t = time()
    centroids, y_train_pred = fsn_training_with_fgn_shuffles(
        x_train_work, y_train_work, thresholds, metric,
        num_shuffles=num_shuffles, mode='fgn_ma', verbose=verbose, level=level,
        num_labels=10)
    time_diff = str(timedelta(seconds=time() - t)).split('.')[0]
    print('Duration FGN shuffles training: {}'.format(time_diff))

    total_centroids = np.sum([len(centroids[c]) for c in centroids])
    print('Total number of centroids: {}'.format(total_centroids))

    t = time()
    y_val_pred = fsn_test_with_fgn_shuffles(x_val_work, centroids, metric)
    time_diff = str(timedelta(seconds=time() - t)).split('.')[0]
    print('Duration FGN shuffles validation: {}'.format(time_diff))

    print('Accuracy training set: {}'.format(
        report_accuracy(y_train_work, y_train_pred)))
    print('Accuracy validation set (min_dist): {}'.format(
        report_accuracy(y_val_work, y_val_pred)))

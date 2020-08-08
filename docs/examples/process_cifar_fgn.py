from datetime import timedelta
from robotikos.fgm import assign_to_centroids, fgn_shuffles
from robotikos.metric import (NormalizedSumAbsDiffMetric,
                              NormalizedSumSquaredDiffMetric)
from robotikos.metric import IdentityFeature
from robotikos.metricspeed import (assign_to_centroid_nl1,
                                   assign_to_centroid_nl2)
from skimage.feature import hog
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from time import time
from torchvision import transforms


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras.applications as tf_apps
import torch
import torchvision.models as pt_apps


LEVEL = None
VERBOSE = True


def feature_from_pt(img, trans, model):
    img = trans(img)
    img = torch.unsqueeze(img, 0)
    feature = model(img)
    feature = feature.data.numpy()
    feature = np.reshape(feature, (1, np.prod(feature.shape)))
    return feature


def feature_from_tf(img, model):
    img = np.expand_dims(img, axis=0)
    return model.predict(img)


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


def plot(x, num_centroids, train_accs, val_accs):
    fig = plt.figure(10)
    fig.clf()

    ax = fig.add_subplot(311)
    ax.plot(x, num_centroids)
    ax.set_ylabel('Centroids')

    ax = fig.add_subplot(312)
    ax.plot(x, train_accs)
    ax.set_ylabel('Training Accuracy')

    ax = fig.add_subplot(313)
    ax.plot(x, val_accs)
    ax.set_ylabel('Validation Accuracy')

    plt.show()


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


def shog(img, orientations=9, px_per_cell=(8, 8), cells_per_block=(2, 2),
         norm='L2-Hys', viz=False, sqrt=True):
    res = hog(
        img, orientations=orientations, pixels_per_cell=px_per_cell,
        cells_per_block=cells_per_block, block_norm=norm, visualize=viz,
        transform_sqrt=sqrt, multichannel=True)
    return res


def viz_hog(img):
    fd, hog_img = shog(img, viz=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(img)
    #ax1.set_title('Input image')

    ax2.axis('off')
    ax2.imshow(hog_img, cmap=plt.cm.gray)
    #ax2.set_title('Histogram of Oriented Gradients')
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
    data_shape = x_train_work.shape[1:]
    # [N, 1, 3072=(32x32x3)]
    x_train_work = np.reshape(x_train_work, (y_train.shape[0], 1, 
                                             np.prod(data_shape)))
    # [N, 3, 1024=(32x32)]
    """

    #viz_hog(x_train[0])

    t = time()
    x_train_work = np.array([shog(x_train_work[i])[np.newaxis, :]
                             for i in range(y_train.shape[0])])
    time_diff = str(timedelta(seconds=time() - t)).split('.')[0]
    print('Duration feature descriptor calculation on training dataset: '
          '{}'.format(time_diff))

    """
    base_model = tf_apps.VGG16(include_top=False, weights='imagenet',
                               input_shape=(32, 32, 3), classes=10)
    base_model.summary()
    model = Model(inputs=base_model.inputs,
                  outputs=base_model.get_layer('block5_pool').output)
    """

    """
    base_model = tf_apps.ResNet50(include_top=False, weights='imagenet',
                               input_shape=(32, 32, 3), classes=10)
    base_model.summary()
    model = Model(inputs=base_model.inputs,
                  outputs=base_model.get_layer('conv5_block3_out').output)
    """

    """
    t = time()
    x_train_work = np.array([feature_from_tf(x_train_work[i], model)[0, 0]
                             for i in range(y_train.shape[0])])
    time_diff = str(timedelta(seconds=time() - t)).split('.')[0]
    print('Duration feature descriptor calculation on training dataset: '
          '{}'.format(time_diff))
    """

    """
    base_model = pt_apps.alexnet(pretrained=True).features
    print(base_model)
    # Conv layers at: 0, 3, 6, 8, and 10
    model = base_model[:11]
    # Deactivate dropout layers
    model = model.eval()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    t = time()
    x_train_work = np.array([feature_from_pt(x_train[i], transform, model)
                             for i in range(y_train.shape[0])])
    time_diff = str(timedelta(seconds=time() - t)).split('.')[0]
    print('Duration feature descriptor calculation on training dataset: '
          '{}'.format(time_diff))
    """

    val_prop = .2
    print('Splitting training dataset into training and validation({}/{})'
          '...'.format(int((1 - val_prop) * 100), int(val_prop * 100)))
    x_train_work, x_val_work, y_train, y_val = train_test_split(
        x_train_work, y_train, test_size=val_prop, random_state=42,
        stratify=y_train)
    train_sz = y_train.shape[0]
    val_sz = y_val.shape[0]
    test_sz = y_test.shape[0]
    print('Split training size: {}'.format(train_sz))
    print('Split validation size: {}'.format(val_sz))

    y_train_work = y_train
    y_val_work = y_val

    feature_shape = x_train_work[0].shape

    # HOG(orientations=9, px_per_cell=(8, 8), cells_per_block=(2, 2),
    # norm='L2-Hys', sqrt=True)
    """
    shuffle = 2
    metric = NormalizedSumSquaredDiffMetric(feature=IdentityFeature())
    thr = .01
    min_thr = .001
    thr_step = .001
    # Thr=.006, Train Acc=.91835, Val Acc=.5006
    """
    """
    shuffle = 2
    metric = NormalizedSumAbsDiffMetric(feature=IdentityFeature())
    thr = .1
    min_thr = .01
    thr_step = .01
    # Thr=.06, Train Acc=.9559, Val Acc=.5264
    """
    metric = NormalizedSumAbsDiffMetric(feature=IdentityFeature())
    thr = .06
    shuffle = 10
    min_shuffle = 1
    shuffle_step = 1

    # ResNet50(output='conv5_block3_out')
    """
    num_shuffles = 2
    metric = NormalizedSumAbsDiffMetric(feature=IdentityFeature())
    thr = .01
    min_thr = .001
    thr_step = .001
    # Thr=.006, Train Acc=, Val Acc=.3
    """

    #thrs = []
    shuffles = []

    train_times = []
    num_centroids = []
    valpred_times = []
    train_accs = []
    val_accs = []

    #while thr >= min_thr:
    while shuffle >= min_shuffle:
        #thrs.append(thr)
        shuffles.append(shuffle)

        t = time()
        centroids, y_train_pred = fsn_training_with_fgn_shuffles(
            x_train_work, y_train_work, [thr], metric,
            num_shuffles=shuffle, mode='fgn_ma', verbose=VERBOSE,
            level=LEVEL, num_labels=10)
        time_diff = str(timedelta(seconds=time() - t)).split('.')[0]
        print('Duration FGN shuffles training: {}'.format(time_diff))
        train_times.append(time_diff)

        total_centroids = np.sum([len(centroids[c]) for c in centroids])
        print('Total number of centroids: {}'.format(total_centroids))
        num_centroids.append(total_centroids)

        t = time()
        y_val_pred = fsn_test_with_fgn_shuffles(x_val_work, centroids, metric)
        time_diff = str(timedelta(seconds=time() - t)).split('.')[0]
        print('Duration FGN shuffles validation: {}'.format(time_diff))
        valpred_times.append(time_diff)

        train_acc = report_accuracy(y_train_work, y_train_pred)
        print('Accuracy training set: {}'.format(train_acc))
        train_accs.append(train_acc)

        val_acc = report_accuracy(y_val_work, y_val_pred)
        print('Accuracy validation set (min_dist): {}'.format(val_acc))
        val_accs.append(val_acc)

        #plot(thrs, num_centroids, train_accs, val_accs)
        plot(shuffles, num_centroids, train_accs, val_accs)

        #thr -= thr_step
        shuffle -= shuffle_step

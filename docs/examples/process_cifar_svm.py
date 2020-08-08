from datetime import timedelta
from docs.examples.process_cifar_fgn import report_accuracy, shog, viz_hog
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from time import time


import numpy as np
import tensorflow as tf


if __name__ == '__main__':
    dataset = tf.keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    print('Original training size: {}'.format(y_train.shape[0]))
    print('Original testing size: {}'.format(y_test.shape[0]))

    x_train_work = x_train / 255.

    # viz_hog(x_train_work[0])

    t = time()
    x_train_work = np.array([shog(x_train_work[i])
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

    y_train_work = np.ravel(y_train)

    grid = GridSearchCV(LinearSVC(loss='hinge'), {
        'C': [100, 10, 8, 4, 2, 1.5, 1.2, 1, .1, .01, .001]}, n_jobs=-1)

    """
    pipeline = Pipeline([('kernel', Nystroem(kernel='linear')),
                         ('clf', LinearSVC())])

    param_grid = [
        {'kernel': [Nystroem(gamma=.01), Nystroem(gamma=.001),
                    Nystroem(gamma=.0001)],
         'clf__penalty': ['l1'], 'clf__loss': ['hinge'],
         'clf__C': [100, 10, 8, 4, 2, 1.5, 1.2, .1, .01, .001],
         'clf__dual': [False]}
    ]
    grid = GridSearchCV(pipeline, param_grid, n_jobs=-1)
    """

    t = time()
    grid.fit(x_train_work, y_train_work)
    time_diff = str(timedelta(seconds=time() - t)).split('.')[0]
    print('Duration hyperparams estimation: {}'.format(time_diff))

    model = grid.best_estimator_

    # HOG(orientations=9, px_per_cell=(8, 8), cells_per_block=(2, 2),
    # norm='L2-Hys', sqrt=True)
    #model = LinearSVC(dual=False)  # Train Acc=.562175, Val Acc=.5359
    #model = LinearSVC(C=1.2)  # Train Acc=.5625, Val Acc=.5357
    #model = LinearSVC(C=8, dual=False)  # Train Acc=.563, Val Acc=.533
    #model = LinearSVC(C=4, dual=False)  # Train Acc=.5628, Val Acc=.5333
    """
    model = Pipeline([('kernel', Nystroem(gamma=.01)),
                      ('clf', LinearSVC(penalty='l1', C=2, dual=False))])
    # Train Acc=.511275, Val Acc=.5004
    """
    """
    model = LinearSVC(loss='hinge', C=.001)  # Train Acc=.505525, Val Acc=.4952
    """

    t = time()
    model.fit(x_train_work, y_train_work)
    time_diff = str(timedelta(seconds=time() - t)).split('.')[0]
    print('Duration model fitting: {}'.format(time_diff))

    t = time()
    y_train_pred = model.predict(x_train_work)
    print('Duration training prediction: {}'.format(time_diff))

    t = time()
    y_val_pred = model.predict(x_val_work)
    time_diff = str(timedelta(seconds=time() - t)).split('.')[0]
    print('Duration validation prediction: {}'.format(time_diff))

    print('Accuracy training set: {}'.format(
        report_accuracy(y_train, y_train_pred)))
    print('Accuracy validation set: {}'.format(
        report_accuracy(y_val, y_val_pred)))

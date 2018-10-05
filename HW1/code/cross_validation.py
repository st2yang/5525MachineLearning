import numpy as np
from math import ceil


def n_fold_cross_val(X, y, classifier_, number_folds, number_splits):
    assert X.shape[0] == y.size
    print(number_folds, "fold cross validation with", number_splits, "splits for", classifier_.__name__)
    number_observations = y.size
    train_error_mat = np.zeros((number_splits, number_folds))
    test_error_mat = np.zeros((number_splits, number_folds))
    index = np.arange(number_observations)
    for rep in range(0, number_splits):
        np.random.shuffle(index)
        index = index[np.argsort(y[index], kind='mergesort')]
        for fold in range(0, number_folds):
            train_index = np.remainder(index, number_folds) != fold
            test_index = np.remainder(index, number_folds) == fold
            X_train = X[train_index, :]
            y_train = y[train_index]
            X_test = X[test_index, :]
            y_test = y[test_index]
            model = classifier_(X_train, y_train)
            train_error_mat[rep, fold] = model.validate(X_train, y_train)
            test_error = model.validate(X_test, y_test)
            test_error_mat[rep, fold] = test_error
            print('no.{} fold no.{} split, test error {}'.format(fold, rep, test_error))
    print("cross validation finished")
    return test_error_mat, train_error_mat


def percents_validation(X, y, Classifier_, number_splits, train_percent):
    print("percent validation with training percent", train_percent)
    print("with", number_splits, "random 80-20 train-test splits", "for", Classifier_.__name__)
    test_error_mat = np.zeros((number_splits, train_percent.size))
    for rep in range(0, number_splits):
        # split 80% as training and 20% as test
        train_index, test_index = train_test_index(y, .8)
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]
        for p in range(0, train_percent.size):
            percent = train_percent[p] / 100
            train_sub_index = train_test_index(y_train, percent)[0]
            X_train_sub = X_train[train_sub_index, :]
            y_train_sub = y_train[train_sub_index]
            model = Classifier_(X_train_sub, y_train_sub)
            test_error = model.validate(X_test, y_test)
            test_error_mat[rep, p] = test_error
            print('{}% percents no.{} split, test error {}'.format(train_percent[p], rep, test_error))
    print("percents validation finished")
    return test_error_mat


def train_test_index(y, train_percent):
    number_observations = y.size
    y_index = np.arange(number_observations)
    y_vals = np.unique(y)
    train_index = np.array([], dtype=int)
    for k in range(0, y_vals.size):
        y_sub_index = y_index[y == y_vals[k]]
        num_train_k = ceil(y_sub_index.size * train_percent)
        train_index_k = np.random.choice(y_sub_index, num_train_k, replace=False)
        train_index = np.concatenate((train_index, train_index_k))
    test_index = np.setdiff1d(y_index, train_index, assume_unique=True)
    return train_index, test_index

# This file contains functions for cross validation
import numpy as np
from math import ceil


def n_fold_cv(X, y, Classifier_, n_fold, n_rep):
    print("Running", n_fold, "fold cross validation with", n_rep, "replicates for", Classifier_.__name__)
    n_obs = y.size
    train_error_mat = np.zeros((n_rep, n_fold))
    test_error_mat = np.zeros((n_rep, n_fold))
    index = np.arange(n_obs)
    for rep in range(0, n_rep):
        print("Runnnig replicate No.", rep, end="\r")
        np.random.shuffle(index)
        index = index[np.argsort(y[index], kind='mergesort')]
        for fold in range(0, n_fold):
            train_index = np.remainder(index, n_fold) != fold
            test_index = np.remainder(index, n_fold) == fold
            X_train = X[train_index, :]
            y_train = y[train_index]
            X_test = X[test_index, :]
            y_test = y[test_index]
            model = Classifier_(X_train, y_train)
            train_error_mat[rep, fold] = model.validate(X_train, y_train)
            test_error_mat[rep, fold] = model.validate(X_test, y_test)
    print("Cross Validation Complete")
    return (test_error_mat, train_error_mat)


def fancy_cv(X, y, Classifier_, n_rep, train_percent):
    print("Running validation for training percent", train_percent)
    print("with", n_rep, "random 80-20 train-test splits", "for", Classifier_.__name__)
    # train_error_mat = np.zeros((n_rep, train_percent.size))
    test_error_mat = np.zeros((n_rep, train_percent.size))
    for rep in range(0, n_rep):
        print("Split No.", rep, end="\r")
        # here split the whole set as 80% train 20% test
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
            # train_error_mat[rep, p] = model.validate(X_train_sub, y_train_sub)
            test_error_mat[rep, p] = model.validate(X_test, y_test)
    print("Validation Complete")
    return test_error_mat


def train_test_index(y, train_percent):
    num_obs = y.size
    y_index = np.arange(num_obs)
    y_vals = np.unique(y)
    train_index = np.array([], dtype=int)
    for k in range(0, y_vals.size):
        y_sub_index = y_index[y == y_vals[k]]
        num_train_k = ceil(y_sub_index.size * train_percent)
        train_index_k = np.random.choice(y_sub_index, num_train_k, replace=False)
        train_index = np.concatenate((train_index, train_index_k))
    test_index = np.setdiff1d(y_index, train_index, assume_unique=True)
    return (train_index, test_index)
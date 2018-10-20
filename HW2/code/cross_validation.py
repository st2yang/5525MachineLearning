import numpy as np


def n_fold_cross_val(X, y, classifier_, regulator_C, number_folds):
    assert X.shape[0] == y.size
    print(number_folds, "fold cross validation for", classifier_.__name__)
    number_observations = y.size
    train_error_vec = np.zeros(number_folds)
    test_error_vec = np.zeros(number_folds)
    index = np.arange(number_observations)
    np.random.shuffle(index)
    index = index[np.argsort(y[index], kind='mergesort')]
    for fold in range(0, number_folds):
        train_index = np.remainder(index, number_folds) != fold
        test_index = np.remainder(index, number_folds) == fold
        X_train = X[train_index, :]
        y_train = y[train_index]
        X_test = X[test_index, :]
        y_test = y[test_index]
        model = classifier_(X_train, y_train, regulator_C)
        train_error_vec[fold] = model.validate(X_train, y_train)
        test_error = model.validate(X_test, y_test)
        test_error_vec[fold] = test_error
        print('no.{} fold, test error {}'.format(fold, test_error))
    print("cross validation finished")
    return test_error_vec, train_error_vec
# main file for question 4
# compare simple generative and discriminative classifiers on
# the learning curves using the HomeVal50 and Digits datasets

import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression
from naive_bayes import NaiveBayes
from cross_validation import percents_validation
from sklearn.datasets import load_digits
from sklearn.datasets import load_boston


def _logisticRegression(data, number_splits=10, train_percent=np.array([10, 25, 50, 75, 100])):
    X = data[:, 1:]
    y = data[:, 0].astype(int)
    test_error = percents_validation(X, y, LogisticRegression, number_splits, train_percent)
    test_error_mean = np.mean(test_error, axis=0)
    test_error_std = np.std(test_error, axis=0, ddof=1)
    return test_error_mean, test_error_std


def _naiveBayesGaussian(data, number_splits=10, train_percent=np.array([10, 25, 50, 75, 100])):
    X = data[:, 1:]
    y = data[:, 0].astype(int)
    test_error = percents_validation(X, y, NaiveBayes, number_splits, train_percent)
    test_error_mean = np.mean(test_error, axis=0)
    test_error_std = np.std(test_error, axis=0, ddof=1)
    return test_error_mean, test_error_std


def NgJordanExperiment(data, dataset_name, train_percent=np.array([10, 25, 50, 75, 100])):
    # the first column of the data is the target
    lr_error_mean, lr_error_std = _logisticRegression(data)
    nb_error_mean, nb_error_std = _naiveBayesGaussian(data)

    plt.xlim([9.0, 101])
    plt.ylim([0.0, 0.3])
    plt.errorbar(train_percent, lr_error_mean, yerr=lr_error_std, label='logistic regression', fmt='-o', capthick=2)
    plt.errorbar(train_percent, nb_error_mean, yerr=nb_error_std, label='naive bayes', fmt='--o', color='r', capthick=2)
    plt.legend()
    plt.ylabel('Test Error Rate')
    plt.xlabel('Training Percent')
    plt.title('Logistic Regression V.S. Naive Bayes with ' + dataset_name, fontsize=12, verticalalignment='bottom')
    plt.show()


def main():
    np.random.seed(0)
    # digits data
    digits = load_digits()
    digits_input = np.c_[digits.target, digits.data]
    NgJordanExperiment(digits_input, 'digits')
    # HomeVal50 data
    boston = load_boston()
    boston_y = boston.target
    number_observations = boston_y.size
    y_med = np.median(boston_y)
    HomeVal50 = np.zeros(number_observations)
    for i in range(number_observations):
        HomeVal50[i] = boston_y[i] >= y_med
    HomeVal50_input = np.c_[HomeVal50, boston.data]
    NgJordanExperiment(HomeVal50_input, 'HomeVal50')


if __name__ == '__main__':
    main()
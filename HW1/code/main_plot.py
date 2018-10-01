import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression
from naive_bayes import NaiveBayes
from cross_validation import fancy_cv
from random import seed
from sklearn.datasets import load_digits


def _logisticRegression(data, num_splits=100, train_percent=np.array([5, 10, 15, 20, 25, 30])):
    X = data[:, 1:]
    y = data[:, 0].astype(int)
    test_error = fancy_cv(X, y, LogisticRegression, num_splits, train_percent)
    test_error_mean = np.mean(test_error, axis=0)
    test_error_std = np.std(test_error, axis=0, ddof=1)
    return (test_error_mean, test_error_std)


def _naiveBayesGaussian(data, num_splits=100, train_percent=np.array([5, 10, 15, 20, 25, 30])):
    X = data[:, 1:]
    y = data[:, 0].astype(int)
    test_error = fancy_cv(X, y, NaiveBayes, num_splits, train_percent)
    test_error_mean = np.mean(test_error, axis=0)
    test_error_std = np.std(test_error, axis=0, ddof=1)
    return (test_error_mean, test_error_std)

seed(5525)
np.random.seed(5525)
digits = load_digits()
X = digits.data
y = digits.target
input = np.c_[y, X]
logreg_error_mean, logreg_error_std = _logisticRegression(input)
nb_error_mean, nb_error_std = _naiveBayesGaussian(input)
train_percent = np.array([5, 10, 15, 20, 25, 30])

plt.xlim([4.5, 30.5])
plt.ylim([0.05, 0.5])
plt.errorbar(train_percent, logreg_error_mean, yerr=logreg_error_std * 1.96, label='logistic regression', fmt='-o', capthick=2)
plt.errorbar(train_percent, nb_error_mean, yerr=nb_error_std * 1.96, label='naive bayes', fmt='--o', color='r', capthick=2)
plt.legend()
plt.ylabel('Test Error Rate')
plt.xlabel('Training Percent')
plt.title('Logistic Regression V.S. Naive Bayes', fontsize=18, verticalalignment='bottom')
plt.show()
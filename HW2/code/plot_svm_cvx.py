import numpy as np
from random import seed
import matplotlib.pyplot as plt
from svm_cvx import SVMCVX
from cross_validation import n_fold_cross_val


data_MNIST = np.genfromtxt('../data/MNIST-13.csv', delimiter=',')
X = data_MNIST[:, 1:]
y = data_MNIST[:, 0]


def plot_test_performance(C):
    number_folds = 5
    test_error_mean = []
    test_error_std = []
    for c in C:
        test_error_vec, _ = n_fold_cross_val(X, y, SVMCVX, c, number_folds)
        test_error_mean.append(np.mean(test_error_vec))
        test_error_std.append(np.std(test_error_vec))
    plt.errorbar(C, test_error_mean, yerr=test_error_std, fmt='-o', capthick=2)
    plt.ylabel('test error rate')
    plt.xlabel('C value')
    plt.xscale('log')
    plt.show()


def plot_margin_related(C):
    number_support_vectors = []
    margin = []
    for c in C:
        model = SVMCVX(X, y, c)
        number_support_vectors.append(model.number_support_vectors)
        margin.append(model.margin)
    plt.plot(C, number_support_vectors)
    plt.ylabel('number of support vectors')
    plt.xlabel('C value')
    plt.xscale('log')
    plt.show()
    plt.plot(C, margin)
    plt.ylabel('geometric margin')
    plt.xlabel('C value')
    plt.xscale('log')
    plt.show()


def plot_versus_C(C):
    plot_margin_related(C)
    plot_test_performance(C)


def main():
    seed(5525)
    np.random.seed(5525)
    C = np.array([0.01, 0.1, 1, 10, 100])
    plot_versus_C(C)


if __name__ == "__main__":
    main()

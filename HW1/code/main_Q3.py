# main file for question 3
# use Fishers linear discriminant analysis (LDA) 1D and 2D to project data
# and do a bivariate Gaussian generative modeling for 2D projected data

from sklearn.datasets import load_digits
from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
from cross_validation import n_fold_cross_val
from LDA2d_GaussGM import LDA2dGaussGM


def _LDA1dProjection(data):
    X = data[:, 1:]
    y = data[:, 0].astype(int)
    number_observations = y.size
    # process the data
    y_med = np.median(y)
    HomeVal50 = np.zeros(number_observations)
    for i in range(number_observations):
        HomeVal50[i] = y[i] >= y_med
    X_class0 = X[HomeVal50 == 0]
    X_class1 = X[HomeVal50 == 1]
    # LDA
    mean_class0 = np.mean(X_class0, axis=0)
    mean_class1 = np.mean(X_class1, axis=0)
    S_w = np.matmul(np.transpose(X_class1 - mean_class1), (X_class1 - mean_class1)) + \
          np.matmul(np.transpose(X_class0 - mean_class0), (X_class0 - mean_class0))
    w = np.dot(np.linalg.inv(S_w), (mean_class1 - mean_class0))
    # projection and plot
    X_projected = np.matmul(X, w)
    plt.hist(X_projected[HomeVal50 == 1], label='class1', bins=100)
    plt.hist(X_projected[HomeVal50 == 0], label='class0', bins=100)
    plt.legend()
    plt.ylabel('number of projected points')
    plt.xlabel('projected x value')
    plt.title('Histogram of LDA1dProjection with HomeVal50 data')
    plt.show()


def _LDA2dGaussGM(data, num_cross_val=10):
    X = data[:, 1:]
    y = data[:, 0].astype(int)
    test_error_mat, train_error_mat = n_fold_cross_val(X, y, LDA2dGaussGM, num_cross_val, 10)
    train_error = np.mean(train_error_mat, axis=0)
    test_error_mean = np.mean(test_error_mat, axis=0)
    test_error_std = np.std(test_error_mat, axis=0, ddof=1)
    print("train_error for LDA2dGaussGM with digits dataset", train_error)
    print("test_error for LDA2dGaussGM with digits dataset", test_error_mean)
    print("test_error_std for LDA2dGaussGM with digits dataset", test_error_std)

def main():
    # question 3a
    boston = load_boston()
    boston_input = np.c_[boston.target, boston.data]
    _LDA1dProjection(boston_input)
    # question 3c
    digits = load_digits()
    digits_input = np.c_[digits.target, digits.data]
    _LDA2dGaussGM(digits_input)


if __name__ == '__main__':
    main()

from sklearn.datasets import load_digits
import numpy as np
from cross_validation import n_fold_cv
from cross_validation import fancy_cv
from LDA2d_GaussGM import LDA2dGaussGM
from naive_bayes import NaiveBayes
from logistic_regression import LogisticRegression

digits = load_digits()
X = digits.data
y = digits.target
print("X.shape ", X.shape)
print("y.shape ", y.shape)

# LDA2dGaussGM
# test_error = n_fold_cv(X, y, LDA2dGaussGM, 10, 2)
# print("test_error", test_error)

# NaiveBayes
# train_percent=np.array([5, 10, 15, 20, 25, 30])
# test_error = fancy_cv(X, y, NaiveBayes, 2, train_percent)
# print("test_error", test_error)

# LogisticRegression
# train_percent=np.array([5, 10, 15, 20, 25, 30])
# test_error = fancy_cv(X, y, LogisticRegression, 2, train_percent)
# print("test_error", test_error)

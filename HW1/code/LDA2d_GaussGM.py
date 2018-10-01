import numpy as np
from numpy import mat
from math import log
from scipy.sparse.linalg import eigs
from scipy.stats import multivariate_normal
from classifier import Classifier
from data_preprocessor import DataPreprocessor


class LDA2dGaussGM(Classifier):
    def __init__(self, X, y):
        # remove zero covariance features and standardize
        self.data_preprocessor = DataPreprocessor(X)
        X = self.data_preprocessor.process_data(X)
        y = np.copy(y)
        self.number_features = X.shape[1]
        self.all_classes = np.unique(y)
        self.number_classes = self.all_classes.size
        self.W = self.calculate_weight_vector(X, y)
        self.prior, self.mean, self.covariance = \
            self.calculate_GaussGM_parameters(self.LDA_projection(X), y)

    def calculate_weight_vector(self, X, y):
        # hard coded for 2d projection
        k = 2
        X_kclass = {}
        for one_class in self.all_classes:
            X_kclass[one_class] = X[y == one_class]
        mean_all = np.mean(X, axis=0)
        S_T = np.matmul(np.transpose(X - mean_all), X - mean_all)
        S_W = np.zeros((self.number_features, self.number_features))
        for one_class in self.all_classes:
            mean_each = np.mean(X_kclass[one_class], axis=0)
            S_W += np.matmul(np.transpose(X_kclass[one_class] - mean_each),
                             X_kclass[one_class] - mean_each)
        S_B = S_T - S_W
        temp_mat = mat(np.linalg.inv(S_W)) * mat(S_B)
        _, eig_vecs = eigs(temp_mat, k=k)
        return eig_vecs.real

    def LDA_projection(self, X):
        assert X.shape[1] == self.W.shape[0]
        return X.dot(self.W)

    def calculate_GaussGM_parameters(self, X, y):
        number_features = X.shape[1]
        priors = [np.sum(y == one_class) / y.size for one_class in self.all_classes]
        means = np.zeros((self.number_classes, number_features))
        covariances = np.zeros((self.number_classes, number_features, number_features))
        for k in range(0, self.number_classes):
            index = y == self.all_classes[k]
            X_classk = X[index, :]
            means[k, :] = np.mean(X_classk, axis=0)
            covariances[k, :, :] = np.cov(X_classk, rowvar=False, bias=True)
        return priors, means, covariances

    def validate(self, X_test, y_test):
        X_test = self.data_preprocessor.process_data(X_test)
        assert X_test.shape[1] == self.number_features
        X_test = self.LDA_projection(X_test)
        predicted_scores = self.predict_score(X_test)
        predicted_class = self.predict_class(predicted_scores)
        test_error = self.calculate_predict_error(predicted_class, y_test)
        return test_error

    def calculate_predict_error(self, predicted_class, y):
        predicted_indicator = np.array([predicted_class[i] == y[i] for i in range(0, y.size)])
        return 1 - np.sum(predicted_indicator) / y.size

    def predict_class(self, score):
        max_indicator = np.argmax(score, axis=1)
        return np.array([self.all_classes[i] for i in max_indicator])

    def predict_score(self, X):
        N = X.shape[0]
        log_score = np.zeros((N, self.number_classes))
        for k in range(self.number_classes):
            mean_k = self.mean[k, :]
            cov_k = self.covariance[k, :, :]
            log_score[:, k] = multivariate_normal.logpdf(X, mean_k, cov_k)
        log_prior = [log(p) for p in self.prior]
        log_score += log_prior
        return log_score

# -----------
#  test
# from sklearn.datasets import load_digits
# digits = load_digits()
# X = digits.data
# y = digits.target
# model = LDA2dGaussGM(X, y)
# print(model.validate(X, y))

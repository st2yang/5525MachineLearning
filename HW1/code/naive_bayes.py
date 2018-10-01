import numpy as np
from math import log
from scipy.stats import norm
from classifier import Classifier
from data_preprocessor import DataPreprocessor


class NaiveBayes (Classifier):
    def __init__(self, X, y):
        # remove zero covariance features and standardize
        self.data_preprocessor = DataPreprocessor(X)
        X = self.data_preprocessor.process_data(X)
        y = np.copy(y)
        self.number_features = X.shape[1]
        self.all_classes = np.unique(y)
        self.number_classes = self.all_classes.size
        self.prior, self.mean_array, self.std_array = self.calculate_Gauss_parameters(X, y)

    def calculate_Gauss_parameters(self, X, y):
        prior = [np.sum(y == y_val) / y.size for y_val in self.all_classes]
        num_obs, _ = X.shape
        mean_array = np.zeros((self.number_classes, self.number_features))
        std_array = np.zeros((self.number_classes, self.number_features))
        for k in range(0, self.number_classes):
            index = y == self.all_classes[k]
            X_sub = X[index, :]
            mean_array[k, :] = np.mean(X_sub, axis=0)
            std_array[k, :] = np.std(X_sub, axis=0, ddof=1)
            std_array[std_array < 1e-03] = 1e-03
        return prior, mean_array, std_array

    def validate(self, X_test, y_test):
        X_test = self.data_preprocessor.process_data(X_test)
        assert X_test.shape[1] == self.number_features
        predicted_score = self.predict_score(X_test)
        predicted_class = self.predict_class(predicted_score)
        prediction_error = self.calculate_predict_error(predicted_class, y_test)
        return prediction_error

    def calculate_predict_error(self, predicted_class, y):
        predicted_indicator = np.array([predicted_class[i] == y[i] for i in range(0, y.size)])
        return 1 - np.sum(predicted_indicator) / y.size

    def predict_class(self, predicted_score):
        max_indicator = np.argmax(predicted_score, axis=1)
        return np.array([self.all_classes[i] for i in max_indicator])

    def predict_score(self, X):
        N = X.shape[0]
        log_score = np.zeros((N, self.number_classes))
        for k in range(0, self.number_classes):
            for j in range(0, self.number_features):
                log_score[:, k] += norm.logpdf(X[:, j], loc=self.mean_array[k, j], scale=self.std_array[k, j])
        log_prior = [log(p) for p in self.prior]
        log_score += log_prior
        return log_score


##### test
# from sklearn.datasets import load_digits
# digits = load_digits()
# X = digits.data
# y = digits.target
# model = NaiveBayes(X, y)
# print(model.validate(X, y))
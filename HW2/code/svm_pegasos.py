import numpy as np
from classifier import Classifier
from data_preprocessor import DataPreprocessor
from math import floor


class SVMPegasos(Classifier):
    def __init__(self, X, y, sgd_batch_size):
        self.data_preprocessor = DataPreprocessor(X)
        X = self.data_preprocessor.process_data(X)
        y = np.copy(y).astype(int)
        self.all_classes = np.unique(y)
        assert self.all_classes.size == 2
        self.target_value = np.array([1, -1]).astype(int)
        y[y == self.all_classes[0]] = self.target_value[0]
        y[y == self.all_classes[1]] = self.target_value[1]
        self.number_features = X.shape[1]
        # prepare for optimization
        self.loss_record = []
        penalty_lambda = 1
        w_init = np.random.normal(0, 0.001, self.number_features)
        self.svm_weight = self.pegas(X, y, penalty_lambda, w_init, sgd_batch_size)

    def pegas(self, X, y, lambda_, w0, k):
        X_train, y_train, number_samples = self.group_train_data(X, y, k)
        # optimization hyperparameters
        max_iterations = 1000
        max_ktot = 100 * X.shape[0]
        ktot = 0
        for i in range(1, max_iterations):
            ktot += k
            self.loss_record.append(self.compute_loss(X, y, lambda_, w0))
            X_batch, y_batch = self.select_batch(X_train, y_train, number_samples, w0)
            w1 = self.update_weight(X_batch, y_batch, lambda_, w0, k, i)
            w0 = w1
            if ktot >= max_ktot:
                break
        return w0

    def group_train_data(self, X, y, k):
        X_train = {}
        y_train = {}
        for one_class in self.target_value:
            index = y == one_class
            X_train[one_class] = X[index]
            y_train[one_class] = y[index]
        percent = k / len(y)
        number_class1 = floor(percent * len(y_train[self.target_value[0]]))
        number_class2 = k - number_class1
        assert number_class2 <= len(y_train[self.target_value[1]])
        number_samples = {self.target_value[0]: number_class1, self.target_value[1]: number_class2}
        return X_train, y_train, number_samples

    def select_batch(self, X_train, y_train, number_samples, w):
        X_batch = np.array([]).reshape(0, self.number_features)
        y_batch = np.array([])
        for one_class in self.target_value:
            n = len(y_train[one_class])
            k = number_samples[one_class]
            random_number = np.random.permutation(n)
            random_idx = random_number[np.arange(k)]
            X_batch = np.r_[X_batch, X_train[one_class][random_idx, :]]
            y_batch = np.r_[y_batch, y_train[one_class][random_idx]]
        index = (np.dot(X_batch, w) * y_batch) < 1
        return X_batch[index, :], y_batch[index]

    def update_weight(self, X, y, lambda_, w0, k, iter_t):
        # decay learning rate
        eta = 1 / (lambda_ * iter_t)
        w_half = (1 - eta * lambda_) * w0 + eta / k * np.dot(y, X)
        # for numerical stability
        if np.sum(w_half * w_half) < 1e-07:
            w_half = np.maximum(w_half, 1e-04)
        w1 = np.minimum(1, 1 / np.sqrt(lambda_) / np.sqrt(np.sum(w_half * w_half))) * w_half
        return w1

    def compute_loss(self, X, y, lambda_, w):
        n = X.shape[0]
        tmp_loss = 1 - y * np.dot(X, w)
        loss = np.sum(tmp_loss[tmp_loss > 0]) / n + lambda_ / 2 * np.sum(w * w)
        return loss

    def predict(self, X_new):
        X = self.data_preprocessor.process_data(X_new)
        predicted_score = self.predict_score(X)
        predicted_class = self.predict_class(predicted_score)
        return predicted_class

    def validate(self, X_test, y_test):
        X_test = self.data_preprocessor.process_data(X_test)
        assert X_test.shape[1] == self.number_features
        predicted_score = self.predict_score(X_test)
        predicted_class = self.predict_class(predicted_score)
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
        svm_score = np.zeros((N, 2))
        svm_score[:, 0] = X.dot(self.svm_weight)
        svm_score[:, 1] = -svm_score[:, 0]
        return svm_score


##### test
# mnist = np.genfromtxt('../data/MNIST-13.csv', delimiter=',')
# mnist_data = mnist[:, 1:]
# mnist_target = mnist[:, 0].astype(int)
# model = SVMPegasos(mnist_data, mnist_target, 5)
# print(model.validate(mnist_data, mnist_target))

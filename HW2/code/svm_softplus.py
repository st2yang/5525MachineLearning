import numpy as np
from classifier import Classifier
from data_preprocessor import DataPreprocessor
from numpy import genfromtxt

# training error rate 0.013, just acceptable, to tune the parameters


class SVMSoftplus(Classifier):
    def __init__(self, X, y, sgd_batch_percent):
        # hardcode a which should be very small
        self.softplus_a = 0.01
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
        self.svm_weight = self.optimization(X, y, penalty_lambda, w_init, sgd_batch_percent)

    def optimization(self, X, y, lambda_, w0, k):
        X_train = {}
        y_train = {}
        for one_class in self.target_value:
            index = y == one_class
            X_train[one_class] = X[index]
            y_train[one_class] = y[index]
        # optimization hyperparameters
        max_iterations = 1000
        learning_rate = 0.01
        stopping_epsilon = 1e-6
        for i in range(max_iterations):
            self.loss_record.append(self.compute_loss(X, y, lambda_, w0))
            X_batch, y_batch = self.select_batch(X_train, y_train, k)
            grad_w = self.compute_gradient(X_batch, y_batch, lambda_, w0)
            w1 = w0 - learning_rate * grad_w
            if np.sum((w1 - w0) ** 2) < stopping_epsilon:
                break
            else:
                w0 = w1
        return w0

    def select_batch(self, X_train, y_train, k):
        X_batch = np.array([]).reshape(0, self.number_features)
        y_batch = np.array([])
        for one_class in self.target_value:
            n = len(y_train[one_class])
            number_samples = np.floor(n * k / 100).astype(int)
            random_number = np.random.permutation(n)
            random_idx = random_number[np.arange(number_samples)]
            X_batch = np.r_[X_batch, X_train[one_class][random_idx, :]]
            y_batch = np.r_[y_batch, y_train[one_class][random_idx]]
        return X_batch, y_batch

    def compute_gradient(self, X, y, lambda_, w):
        n = X.shape[0]
        gradient_w = np.zeros(self.number_features)
        for i in range(n):
            temp = np.exp((1 - y[i] * w.dot(X[i, :])) / self.softplus_a)
            assert temp != 0
            gradient_w += 1 / (1 + temp) * temp * (-y[i] / self.softplus_a * X[i, :])
        gradient_w = gradient_w * self.softplus_a / n + 2 * lambda_ * w
        return gradient_w

    def compute_loss(self, X, y, lambda_, w):
        n = X.shape[0]
        loss = 0
        for i in range(n):
            loss += np.log(1 + np.exp((1 - y[i] * w.dot(X[i, :])) / self.softplus_a))
        loss = loss * self.softplus_a / n + lambda_ * np.sum(w * w)
        return loss

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
# mnist = genfromtxt('../data/MNIST-13.csv', delimiter=',')
# mnist_data = mnist[:, 1:]
# mnist_target = mnist[:, 0].astype(int)
# model = SVMSoftplus(mnist_data, mnist_target, 10)
# print(model.validate(mnist_data, mnist_target))

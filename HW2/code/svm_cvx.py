import numpy as np
from classifier import Classifier
from data_preprocessor import DataPreprocessor
from numpy import genfromtxt
from cvxopt import matrix
from cvxopt.solvers import qp

class SVMCVX(Classifier):
    def __init__(self, X, y, regulator):
        self.data_preprocessor = DataPreprocessor(X)
        X = self.data_preprocessor.process_data(X)
        y = np.copy(y).astype(int)
        self.all_classes = np.unique(y)
        assert self.all_classes.size == 2
        y[y == self.all_classes[0]] = 1
        y[y == self.all_classes[1]] = -1
        self.number_features = X.shape[1]
        alpha = self.solve_dual_problem(X, y, regulator)
        self.svm_weight, self.svm_bias = SVMCVX.compute_svm_parameters(alpha, X, y, regulator)

    def solve_dual_problem(self, X, y, c):
        # QP problem
        # min 0.5 * xTPx + qTx
        # st Gx <= h, Ax = b
        number_observations = X.shape[0]
        yX = np.reshape(y, (number_observations, 1)) * X
        P = matrix(yX.dot(yX.T))
        q = matrix(-np.ones(number_observations))
        A = matrix(np.reshape(y.astype(float), (1, number_observations)))
        b = matrix([0.0])
        I = np.identity(number_observations)
        G = matrix(np.concatenate((I, -I), axis=0))
        vector_c = c * np.ones(number_observations)
        vector_0 = np.zeros(number_observations)
        h = matrix(np.concatenate((vector_c, vector_0)))
        solution = qp(P, q, G, h, A, b)
        alpha = np.array(solution['x'])
        return alpha.reshape((-1,))

    def compute_svm_parameters(alpha, X, y, c):
        w = (alpha * y).dot(X)
        b = 0
        count = 0
        for i in range(alpha.shape[0]):
            if 0 < alpha[i] < c:
                count += 1
                b += y[i] - w.dot(X[i, :])
        assert count > 0
        b /= count
        return w, b

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
        svm_score[:, 0] = X.dot(self.svm_weight) + self.svm_bias
        svm_score[:, 1] = -svm_score[:, 0]
        return svm_score


##### test
# mnist = genfromtxt('../data/MNIST-13.csv', delimiter=',')
# mnist_data = mnist[:, 1:]
# mnist_target = mnist[:, 0].astype(int)
# model = SVMCVX(mnist_data, mnist_target, 0.01)
# print(model.validate(mnist_data, mnist_target))

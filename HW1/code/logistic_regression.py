import numpy as np
from math import log
from classifier import Classifier
from data_preprocessor import DataPreprocessor

class LogisticRegression (Classifier):
    def __init__(self, X, y):
        # Data_Preprocessor will copy X
        self.data_preprocessor = DataPreprocessor(X)
        X = self.data_preprocessor.process_data(X)
        y = np.copy(y)
        self.all_classes = np.unique(y)
        self.number_classes = self.all_classes.size
        self.number_observations, self.number_features = X.shape
        # row-wise concatenated weight vector
        W_init = np.random.normal(0, 0.001, self.number_classes * self.number_features)
        self.W = self.IRLS(W_init, X, y)

    def IRLS(self, W, X, y):
        # construct YT to compute gradients and hessian
        T = np.zeros((self.number_observations, self.number_classes))
        Y = np.zeros((self.number_observations, self.number_classes))
        # through iterations
        number_iterations = 30
        loss_record = np.zeros(number_iterations)
        for iter in range(number_iterations):
            W_mat = self.W_vector2matrix(W)
            for i in range(self.number_observations):
                T[i, y[i]] = 1
                Y[i, :] = LogisticRegression.softmax(W_mat, X[i, :])
            loss_record[iter] = LogisticRegression.cross_entropy_loss(Y, T)
            grad_W = self.compute_gradient(X, Y, T)
            hess_W = self.compute_hessian(X, Y)
            W += - 0.01 * np.matmul(np.linalg.inv(hess_W), grad_W)
            # W += - 0.01 * grad_W
        return self.W_vector2matrix(W)

    def compute_gradient(self, X, Y, T):
        grad_mat = np.zeros((self.number_classes, self.number_features))
        for i in range(self.number_classes):
            grad_mat[i, :] = (Y[:, i] - T[:, i]).dot(X)
        return grad_mat.reshape(self.number_classes * self.number_features)

    def cross_entropy_loss(Y, T):
        loss = 0
        N, K = Y.shape
        for n in range(N):
            for k in range(K):
                loss += -T[n, k] * log(Y[n, k])
        return loss

    def compute_hessian(self, X, Y):
        hess_mat = np.zeros((self.number_classes * self.number_features, self.number_classes * self.number_features))
        for j in range(self.number_classes):
            for k in range(self.number_classes):
                i_kj = 1 if (k == j) else 0
                dot_vec = Y[:, k] * (i_kj - Y[:, j])
                block_kj = np.matmul(np.matmul(X.T, np.diag(dot_vec)), X)
                hess_mat[j * self.number_features : (j + 1) * self.number_features, \
                k * self.number_features : (k + 1) * self.number_features] = block_kj
        # hessian may not be PSD due to numerical issue
        hess_mat = hess_mat + 0.1 * np.identity(self.number_classes * self.number_features)
        return hess_mat

    def W_vector2matrix(self, W_vec):
        assert(W_vec.size == self.number_classes * self.number_features)
        return W_vec.reshape((self.number_classes, self.number_features))

    def softmax(W, x):
        e = np.exp(W.dot(x))
        dist = e / np.sum(e)
        return dist

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
        softmax_score = np.zeros((N, self.number_classes))
        for i in range(N):
            softmax_score[i, :] = LogisticRegression.softmax(self.W, X[i, :])
        return softmax_score


##### test
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
y = digits.target
model = LogisticRegression(X, y)
print(model.validate(X, y))
import numpy as np


class BinaryThreshold(object):
    def __init__(self, X, feature_id, y, weight):
        self.x = X[:, feature_id]
        self.number_features = X.shape[1]
        self.feature_id = feature_id
        self.unique_x = np.unique(self.x)
        self.y = y
        self.n = len(y)
        self.w = weight

    def train(self):
        less_thold, less_score = self.train_less_equal()
        more_thold, more_score = self.train_more_than()
        if less_score < more_score:
            self.is_less = True
            self.thold = less_thold
            return less_score

        else:
            self.is_less = False
            self.thold = more_thold
            return more_score

    def train_less_equal(self):
        # classified as 1 when x <= threshold, otherwise -1
        best_thold = -1
        best_score = float('inf')

        for thold in self.unique_x:
            score = 0
            for i in range(self.n):
                classify_label = -1
                if self.x[i] <= thold:
                    classify_label = 1

                if classify_label * self.y[i] < 0:
                    score += self.w[i]

            if score < best_score:
                best_score = score
                best_thold = thold

        return best_thold, best_score

    def train_more_than(self):
        # classified as 1 when x > threshold, otherwise -1
        best_thold = -1
        best_score = float('inf')

        for thold in self.unique_x:
            score = 0
            for i in range(self.n):
                classify_label = -1
                if self.x[i] > thold:
                    classify_label = 1

                if classify_label * self.y[i] < 0:
                    score += self.w[i]

            if score < best_score:
                best_score = score
                best_thold = thold

        return best_thold, best_score

    def predict(self, X_new):
        X_new = np.reshape(X_new, (-1, self.number_features))
        x_new = X_new[:, self.feature_id]
        predicted_class = -np.ones(len(x_new))
        predicted_class[x_new <= self.thold] = 1
        if self.is_less:
            return predicted_class
        else:
            return -predicted_class


class AdaBoost(object):
    def __init__(self, X, y, number_learners):
        self.number_features = X.shape[1]
        self.n = len(y)
        self.alpha = np.zeros(number_learners)
        self.classifier = []
        self.train(X, y, number_learners)

    def train(self, X, y, M):
        w_m = 1 / self.n * np.ones(self.n)
        features = np.array([0, 2, 3, 1])
        for m in range(M):
            # fit a classifier
            feature_m = features[m % 4]
            classifier = BinaryThreshold(X, feature_m, y, w_m)
            self.classifier.append(classifier)
            error_m = classifier.train()
            # evaluate quantities
            epsilon_m = error_m / np.sum(w_m)
            self.alpha[m] = 0.5 * np.log((1 - epsilon_m) / epsilon_m)
            if not __debug__:
                print('stage: {}/{}'.format(m, M))
                print('weight: ', self.alpha[m])
                print('prediction result y*f(): ', y * self.predict(X))
            # update data weights
            w_m = w_m * np.exp(-self.alpha[m] * y * classifier.predict(X))
            w_m /= np.sum(w_m)

    def predict(self, X_new):
        X_new = np.reshape(X_new, (-1, self.number_features))
        predicted_score = np.zeros(X_new.shape[0])
        for m in range(len(self.classifier)):
            predicted_score += self.alpha[m] * self.classifier[m].predict(X_new)
        return np.sign(predicted_score)


def main():
    # training
    data = np.genfromtxt('data.csv', delimiter=',')
    X = data[:, :-1]
    y = data[:, 4]
    ensembles = AdaBoost(X, y, 4)
    # predict
    x = np.array([3, 0, 0, 1])
    if ensembles.predict(x):
        print('Doggie Joey is adoptable')
    else:
        print('Doggie Joey is not adoptable')


if __name__ == "__main__":
    main()

from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt

# load data
boston = load_boston()
X = boston.data
y = boston.target
data_length = len(y)

# process the data
y_med = np.median(y)
HomeVal50 = np.empty(data_length)
for i in range(data_length):
    HomeVal50[i] = y[i] >= y_med
X_class0 = X[HomeVal50 == 0]
X_class1 = X[HomeVal50 == 1]

# LDA
mean_class0 = np.mean(X_class0, axis = 0)
mean_class1 = np.mean(X_class1, axis = 0)
S_w = np.matmul(np.transpose(X_class1 - mean_class1), (X_class1 - mean_class1)) + \
    np.matmul(np.transpose(X_class0 - mean_class0), (X_class0 - mean_class0))
w = np.dot(np.linalg.inv(S_w), (mean_class1 - mean_class0))

# projection and plot
X_projected = np.matmul(X, w)
plt.hist(X_projected[HomeVal50 == 1])
plt.hist(X_projected[HomeVal50 == 0])
plt.show()
from plotBoundary import *
from svm_cvx import SVMCVX
# import your SVM training code

print('======Training======')
# load data from csv files
train = loadtxt('../data/MNIST-13.csv', delimiter=',')
# use deep copy here to make cvxopt happy
X = train[:, 1:].copy()
y = train[:, 0].copy()

# Carry out training, primal and/or dual

# Define the predictSVM(x) function, which uses trained parameters

# plot training results
plotDecisionBoundary(X, y, None, [-1, 0, 1], title='SVM Train')


print('======Validation======')
# load data from csv files
validate = loadtxt('../data/MNIST-13.csv', delimiter=',')
X = train[:, 1:].copy()
y = train[:, 0].copy()
# plot validation results
plotDecisionBoundary(X, y, None, [-1, 0, 1], title='SVM Validate')

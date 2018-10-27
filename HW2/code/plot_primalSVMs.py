import numpy as np
from random import seed
import matplotlib.pyplot as plt
from svm_pegasos import SVMPegasos
from svm_softplus import SVMSoftplus


data_MNIST = np.genfromtxt('../data/MNIST-13.csv', delimiter=',')
X = data_MNIST[:, 1:]
y = data_MNIST[:, 0]


def draw_loss(classifier_, axis, k):
    number_runs = 5
    for i in range(number_runs):
        print('k={}, run {}/{}'.format(k, i+1, number_runs))
        model = classifier_(X, y, k)
        axis.plot(model.loss_record)
        axis.set_xscale('log')
        axis.set_yscale('log')
        axis.set_ylim([0.08, 1])
        axis.set_xlim([1, 1e4])


def svm_test_batch_size(classifier_):
    f, axarr = plt.subplots(3, 2, figsize=(8, 8))
    axarr[0, 0].set_title('k = 1')
    draw_loss(classifier_, axarr[0, 0], 1)
    axarr[0, 1].set_title('k = 20')
    draw_loss(classifier_, axarr[0, 1], 20)
    axarr[1, 0].set_title('k = 200')
    draw_loss(classifier_, axarr[1, 0], 200)
    axarr[1, 1].set_title('k = 1000')
    draw_loss(classifier_, axarr[1, 1], 1000)
    axarr[2, 0].set_title('k = 2000')
    draw_loss(classifier_, axarr[2, 0], 2000)
    f.delaxes(axarr[2][1])
    plt.suptitle(classifier_.__name__, fontsize=12)
    plt.tight_layout()
    plt.show()


def main():
    seed(5525)
    np.random.seed(5525)
    svm_test_batch_size(SVMPegasos)
    svm_test_batch_size(SVMSoftplus)


if __name__ == "__main__":
    main()


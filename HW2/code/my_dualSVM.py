from svm_cvx import SVMCVX
import numpy as np
import sys
from os.path import isfile
from cross_validation import n_fold_cross_val


def read_file(filename):
    assert isfile(filename)
    data = np.genfromtxt(filename, delimiter=',')
    print('successfully loaded', filename)
    return data


def myDualSVM(filename, C):
    C = float(C)
    data = read_file(filename)
    X = data[:, 1:]
    y = data[:, 0]
    cross_val_k = 10
    test_error_vec, _ = n_fold_cross_val(X, y, SVMCVX, C, cross_val_k)
    test_error_mean = np.mean(test_error_vec)
    test_error_std = np.std(test_error_vec)
    print('average test error: ', test_error_mean)
    print('standard deviation of test error: ', test_error_std)


def main(argv=sys.argv):
    if len(argv) == 3:
        myDualSVM(*argv[1:])
    else:
        print(
            'Usage: python3 ./my_dualSVM.py /path/to/dataset.csv C', file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

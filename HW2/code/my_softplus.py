from svm_softplus import SVMSoftplus
import numpy as np
import sys
import time
from os.path import isfile

# run time too long...


def read_file(filename):
    assert isfile(filename)
    data = np.genfromtxt(filename, delimiter=',')
    print('successfully loaded', filename)
    return data


def mySoftplus(filename, k, numruns):
    numruns = int(numruns)
    k = int(k)
    assert(numruns > 0 and k > 0)
    data = read_file(filename)
    X = data[:, 1:]
    y = data[:, 0]
    time_array = np.zeros(numruns)
    loss_list = []

    print(numruns, 'runs on', k, 'data points')
    for i in range(numruns):
        print('run ', i + 1, "/", numruns, ", please wait...", end="\r")
        begin = time.time()
        model = SVMSoftplus(X, y, 10)
        loss_list.append(model.loss_record)
        end = time.time()
        time_array[i] = end - begin

    time_avg = np.mean(time_array)
    time_std = np.std(time_array, ddof=1)
    with open("./tmp.txt", "w") as f:
        for li in loss_list:
            for val in li:
                print(val, end=", ", file=f)
            print(file=f)
    print('------')
    print('average runtime for ', numruns, ' runs with minibatch size of ', k, ':', round(time_avg, 3), 'seconds')
    print('SD of run time for ', numruns, ' runs with minibatch size of ', k, ':', round(time_std, 3), 'seconds')
    print("plot data exported to ./tmp.txt")


def main(argv=sys.argv):
    if len(argv) == 4:
        mySoftplus(*argv[1:])
    else:
        print('Usage: python3 ./svm_softplus.py /path/to/dataset.csv k numruns', file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

# https://github.com/cympfh/ladder/blob/master/load_mnist.py
import math
import numpy as np
from sklearn.datasets import fetch_mldata


def load(train_n, test_n):
    mnist = fetch_mldata('MNIST original', data_home='.')
    mnist.data = mnist.data.astype(np.float32) / 256.0
    mnist.target = mnist.target.astype(np.int32)
    N = len(mnist.data)

    order = np.random.permutation(N)

    train = {i: [] for i in range(10)}
    test = {i: [] for i in range(10)}

    train_m = math.ceil(train_n / 10)
    train_sum = 0

    test_m = math.ceil(test_n / 10)
    test_sum = 0

    for i in range(N):
        x = mnist.data[order[i]]
        y = mnist.target[order[i]]

        if train_sum < train_n and len(train[y]) < train_m:
            train[y].append(x)
            train_sum += 1

        if test_sum < test_n and len(test[y]) < test_m:
            test[y].append(x)
            test_sum += 1

    return train, test

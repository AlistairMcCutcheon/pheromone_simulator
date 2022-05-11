import numpy as np


def bound(x, upper):
    return round(max(min(x, upper - 2), 1))


def softmax(arr):
    arr = np.exp(arr - np.max(arr))
    return arr / arr.sum()


def one_hot(arr):
    max_elem = np.max(arr)
    max_indexes = []
    for i, x in enumerate(arr):
        if np.isclose(x, max_elem):
            max_indexes.append(i)
    max_index = int(np.random.choice(max_indexes))
    arr = np.zeros_like(arr)
    arr[max_index] = 1
    return arr


def normal(vector):
    return np.array([-vector[1], vector[0]])


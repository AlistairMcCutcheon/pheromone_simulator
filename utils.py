import numpy as np


def bound(x, upper):
    return round(max(min(x, upper - 1), 0))


def softmax(arr):
    arr = np.exp(arr - np.max(arr))
    return arr / arr.sum()


def normal(vector):
    return np.array([-vector[1], vector[0]])


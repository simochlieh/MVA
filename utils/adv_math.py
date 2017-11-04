"""
Some useful math functions used in the homeworks and projects
"""
import math
import numpy as np


def sigmoid(x):
    return 1. / (1 + math.exp(-x))


SIGMOID_FUNC = np.vectorize(sigmoid)
LOG = np.vectorize(math.log)


def compute_gaussian(x, mu, sigma):
    """

    :param x: x is a np array of shape (d, 1)
    :param mu: the mean: np array of shape (d, 1)
    :param sigma: the co-variance matrix: np array of shape (d, d)
    :return: the gaussian density at the point x
    """
    d = x.shape[0]
    return 1. / math.sqrt((2. * math.pi)**d * np.linalg.det(sigma)) \
           * math.exp(-1. / 2. * (x - mu).T.dot(np.linalg.inv(sigma)).dot(x - mu))

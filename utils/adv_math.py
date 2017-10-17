"""
Some useful math functions used in the homeworks and projects
"""
import math
import numpy as np


def sigmoid(x):
    return 1. / (1 + math.exp(-x))

SIGMOID_FUNC = np.vectorize(sigmoid)
LOG = np.vectorize(math.log)

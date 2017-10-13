"""
This script implements a MLE to a generative model (X,Y)
where Y has a Bernoulli distribution and X|Y has a normal distribution
with different means for different classes (Y=0, Y=1) but with the same
covariance matrix
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import utils.data_processing as data


def get_mle_estimates(data_x, data_y, nb_points=None, nb_successes=None):
    """
    returns the MLE estimates of the sample points stored in data,
    assuming a generative model (linear discriminant analysis)
    :param data_x: numpy array of floats n*d
    :param data_y: numpy array of floats n*1 having 1. or 0.
    :param nb_points: int
    :param nb_successes: float
    """
    if not nb_points:
        nb_points = data_x.shape[0]

    if not nb_successes:
        nb_successes = np.sum(data_y)

    pi = nb_successes / nb_points
    mu_1 = np.dot(np.transpose(data_y), data_x) / nb_successes
    mu_0 = np.dot(np.transpose(1 - data_y), data_x) / (nb_points - nb_successes)
    return pi, [mu_0, mu_1]




if __name__ == '__main__':
    data_x, data_y = data.parse_data_with_labels(
        os.path.abspath("classification_data_HWK1/classification_data_HWK1/classificationA.train"),
        dimension=2,
        delimiter="\t")

    successes = data_x[data_y[:, 0] == 1.]
    losses = data_x[data_y[:, 0] == 0.]
    number_points = data_x.shape[0]
    number_successes = np.sum(data_y)
    pi, mu = get_mle_estimates(data_x, data_y, nb_points=number_points, nb_successes=number_successes)

    print "number of successes is %s" % number_successes
    print "the bernoulli parameter pi is %s" % pi
    print "the mean for the class {y=1} is %s" % mu[1]
    print "the mean for the class {y=0} is %s" % mu[0]

    plt.plot(successes[:, 0], successes[:, 1], "bs", losses[:, 0], losses[:, 1], "g^")
    plt.plot(mu[1][:, 0], mu[1][:, 1], "rs", ms=7)
    plt.plot(mu[0][:, 0], mu[0][:, 1], "r^", ms=7)
    plt.show()

"""
This script implements a MLE to a generative model (X,Y)
where Y has a Bernoulli distribution and X|Y has a normal distribution
with different means for different classes (Y=0, Y=1) but with the same
covariance matrix
"""
import numpy as np
import utils.data_processing


def get_mle_estimates(data_x, data_y):
    """
    returns the MLE estimates of the sample points stored in data,
    assuming a generative model (linear discriminant analysis)
    :param data_x: numpy array of floats n*d
    :param data_y: numpy array of floats n*1
    """



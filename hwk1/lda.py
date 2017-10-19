"""
This class implements a MLE to a generative model (LDA) (X,Y)
where Y has a Bernoulli distribution and X|Y has a normal distribution
with different means for different classes (Y=0, Y=1) but with the same
covariance matrix
"""
import math
import matplotlib.pyplot as plt
import numpy as np

from base_classification import BaseClassification
import utils.data_viz as data_viz


class LDA(BaseClassification):

    def __init__(self, data_x, data_y, data_x_test=None, data_y_test=None, dataset_name=None):
        super(LDA, self).__init__(data_x, data_y, data_x_test, data_y_test, dataset_name)

        # Pi the Bernoulli parameter
        self.pi = None

        # The means of classes y = 0 and y = 1
        self.mu_0 = None
        self.mu_1 = None

        # The covariance matrix
        self.sigma = None

        # The learnt parameter
        self.w = None
        self.b = None

        self.title_training_plot = "Linear discriminant analysis: Training"
        self.title_test_plot = "Linear discriminant analysis: Test"

    def train(self):
        """
        Affects the mle estimates to the instance attributes
        pi, mu_0, mu_1, sigma
        """

        # Some useful computation
        nb_points = self.data_x.shape[0]
        nb_successes = np.sum(self.data_y)
        successes = self.data_x[self.data_y[:, 0] == 1.]
        losses = self.data_x[self.data_y[:, 0] == 0.]
        successes_ones_vector = np.ones((nb_successes, 1))
        losses_ones_vector = np.ones((nb_points - nb_successes, 1))

        # MLE estimates
        self.pi = nb_successes / nb_points
        self.mu_0 = (1 - self.data_y).T.dot(self.data_x) / (nb_points - nb_successes)
        self.mu_1 = self.data_y.T.dot(self.data_x) / nb_successes
        self.sigma = 1. / nb_points * (
            (successes - successes_ones_vector.dot(self.mu_1)).T.dot(successes - successes_ones_vector.dot(self.mu_1)) +
            (losses - losses_ones_vector.dot(self.mu_0)).T.dot(losses - losses_ones_vector.dot(self.mu_0)))

        # Find the boundary parameters
        inv_sigma = np.linalg.inv(self.sigma)
        self.w = inv_sigma.dot((self.mu_1 - self.mu_0).T)
        self.b = (1. / 2. * (self.mu_0.dot(inv_sigma).dot(self.mu_0.T) - self.mu_1.dot(inv_sigma).dot(self.mu_1.T)) \
                 + math.log(self.pi / (1. - self.pi)))[0, 0]

    def plot(self, test_mode=False):
        super(LDA, self).plot(test_mode)
        # Adding the mean and the variance
        plt.plot(self.mu_1[:, 0], self.mu_1[:, 1], "rs", ms=6)
        plt.plot(self.mu_0[:, 0], self.mu_0[:, 1], "r^", ms=6)
        data_viz.plot_cov_ellipse(self.sigma, self.mu_0[0, :], color='g')
        data_viz.plot_cov_ellipse(self.sigma, self.mu_1[0, :], color='b')


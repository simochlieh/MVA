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


class QDA(BaseClassification):

    def __init__(self, data_x, data_y, data_x_test, data_y_test, dataset_name):
        super(QDA, self).__init__(data_x, data_y, data_x_test, data_y_test, dataset_name)

        # Pi the Bernoulli parameter
        self.pi = None

        # The means of classes y = 0 and y = 1
        self.mu_0 = None
        self.mu_1 = None

        # The covariance matrices
        self.sigma_0 = None
        self.sigma_1 = None

        # The learnt parameter
        self.a = None
        self.w = None
        self.b = None

        self.title_training_plot = "Quadratic discriminant analysis: Training"
        self.title_test_plot = "Quadratic discriminant analysis: Test"

    def train(self):
        """
        Affects the mle estimates to the instance attributes
        pi, mu_0, mu_1, sigma_0, sigma_1
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
        self.sigma_0 = 1. / (nb_points - nb_successes) * (
            (losses - losses_ones_vector.dot(self.mu_0)).T.dot(losses - losses_ones_vector.dot(self.mu_0))
        )
        self.sigma_1 = 1 / nb_successes * (
            (successes - successes_ones_vector.dot(self.mu_1)).T.dot(successes - successes_ones_vector.dot(self.mu_1))
        )

        # Find the boundary parameters
        inv_sigma_0 = np.linalg.inv(self.sigma_0)
        inv_sigma_1 = np.linalg.inv(self.sigma_1)
        self.a = 1. / 2. * (inv_sigma_0 - inv_sigma_1)
        self.w = inv_sigma_1.dot(self.mu_1.T) - inv_sigma_0.dot(self.mu_0.T)
        self.b = (1. / 2. * (self.mu_0.dot(inv_sigma_0).dot(self.mu_0.T) - self.mu_1.dot(inv_sigma_1).dot(self.mu_1.T)) \
                  + math.log(self.pi / (1. - self.pi)))[0, 0] + 1. / 2. * math.log(np.linalg.det(self.sigma_0) / np.linalg.det(self.sigma_1))

    def plot(self, test_mode=False):
        super(QDA, self).plot(test_mode)
        # Adding the mean and the variance
        plt.plot(self.mu_1[:, 0], self.mu_1[:, 1], "rs", ms=6)
        plt.plot(self.mu_0[:, 0], self.mu_0[:, 1], "r^", ms=6)
        data_viz.plot_cov_ellipse(self.sigma_0, self.mu_0[0, :], color='g')
        data_viz.plot_cov_ellipse(self.sigma_1, self.mu_1[0, :], color='b')

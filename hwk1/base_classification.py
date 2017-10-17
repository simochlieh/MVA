"""
Base class for classification models,
so far only for 2D models with 2 class labels
"""

import matplotlib.pyplot as plt
import numpy as np


class BaseClassification(object):

    def __init__(self, data_x, data_y):

        # 2D dataset points
        self.data_x = data_x

        # labels
        self.data_y = data_y

        # Learnt parameter for the boundary line
        self.w = None
        self.b = None

        # Plot titles
        self.title_training_plot = None
        self.title_test_plot = None

    def train(self):
        """
        Affects the parameters to the instance attributes
        """

    def compute_misclassification_err(self, data_x_test, data_y_test):
        """
        :param data_x_test: test dataset for X
        :param data_y_test: test dataset for Y
        :return:
            the misclassifiction error w.r.t the test dataset
        """
        int_vec = np.vectorize(int)
        data_y_model = int_vec(data_x_test.dot(self.w) + self.b > 0.)
        return (data_y_model - data_y_test).T.dot(data_y_model - data_y_test) / data_y_test.shape[0]

    def plot(self, data_x_test=None, data_y_test=None):
        """
        :param data_x_test: test dataset for X
        :param data_y_test: test dataset for Y
        :return:
        Nothing, only plots the figures
        """
        if data_x_test is None:
            successes_plt, losses_plt = self.plot_cloud_points_label(self.data_x, self.data_y)
            plt.autoscale(enable=False)

            boundary, = self.plot_affine_boundary()
            plt.legend([successes_plt, losses_plt, boundary], ["y = 1", "y = 0", "boundary"])
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.title(self.title_training_plot)

        if data_x_test is not None and data_y_test is not None:
            plt.figure()
            successes_plt, losses_plt = self.plot_cloud_points_label(data_x_test, data_y_test)
            plt.autoscale(enable=False)

            boundary, = self.plot_affine_boundary()
            plt.legend([successes_plt, losses_plt, boundary], ["y = 1", "y = 0", "boundary"])
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.title(self.title_test_plot)

    @staticmethod
    def plot_cloud_points_label(data_x, data_y, **kwargs):
        successes = data_x[data_y[:, 0] == 1.]
        losses = data_x[data_y[:, 0] == 0.]
        successes_plt, = plt.plot(successes[:, 0], successes[:, 1], "bs", color="b", ms=6, **kwargs)
        losses_plt, = plt.plot(losses[:, 0], losses[:, 1], "^", color='lawngreen', ms=6, **kwargs)
        return successes_plt, losses_plt

    def plot_affine_boundary(self, x_min=-10, x_max=10, **kwargs):
        slope = - self.w[0, 0] / self.w[1, 0]
        intercept = - self.b / self.w[1, 0]
        xx = np.linspace(x_min, x_max, 2)
        yy = slope * xx + intercept
        return plt.plot(xx, yy, ls='-', lw=2, color="gray", **kwargs)

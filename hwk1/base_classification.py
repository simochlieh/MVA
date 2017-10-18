"""
Base class for classification models,
so far only for 2D models with 2 class labels
"""

import matplotlib.pyplot as plt
import numpy as np


class BaseClassification(object):

    def __init__(self, data_x, data_y, data_x_test, data_y_test, dataset_name):

        # 2D dataset points
        self.data_x = data_x
        self.data_x_test = data_x_test

        # labels
        self.data_y = data_y
        self.data_y_test = data_y_test

        self.dataset_name = dataset_name

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

    def compute_misclassification_err(self,):
        """
        :param data_x_test: test dataset for X
        :param data_y_test: test dataset for Y
        :return:
            the misclassifiction error w.r.t the training dataset, the misclassifiction error w.r.t the test dataset
        """
        int_vec = np.vectorize(int)
        data_y_model_test = int_vec(self.data_x_test[:, 0:2].dot(self.w) + self.b > 0.)
        data_y_model_train = int_vec(self.data_x[:, 0:2].dot(self.w) + self.b > 0.)
        training_error = (data_y_model_train - self.data_y).T.dot(data_y_model_train - self.data_y) / self.data_y.shape[0]
        test_error = (data_y_model_test - self.data_y_test).T.dot(data_y_model_test - self.data_y_test) / self.data_y_test.shape[0]

        return training_error, test_error

    def plot(self, test_mode=False):
        """
        :param data_x_test: test dataset for X
        :param data_y_test: test dataset for Y
        :return:
        Nothing, only plots the figures
        """
        data_x = self.data_x if not test_mode else self.data_x_test
        data_y = self.data_y if not test_mode else self.data_y_test
        title = self.title_training_plot if not test_mode else self.title_test_plot
        successes_plt, losses_plt = self.plot_cloud_points_label(data_x, data_y)
        plt.autoscale(enable=False)
        boundary, = self.plot_affine_boundary()

        plt.legend([successes_plt, losses_plt, boundary], ["y = 1", "y = 0", "boundary"])
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(title + " dataset %s" % self.dataset_name)

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

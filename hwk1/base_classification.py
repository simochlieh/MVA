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

    def train(self):
        """
        :return:
        A list of parameters learnt from the dataset
        """
        return None

    def find_class(self, x):
        """
        :param x: point (1 row) from the dataset
        :return:
         The class which the point x belongs to
        """
        return int(x.dot(self.w) + self.b > 0)

    def plot(self):
        """
        :param title: string title
        :return:
        Plot the cloud of points and the boundary line that separates the classes
        """
        successes_plt, losses_plt = self.plot_cloud_points_label()
        plt.autoscale(enable=False)

        boundary, = self.plot_affine_boundary()
        plt.legend([successes_plt, losses_plt, boundary], ["y = 1", "y = 0", "boundary"])
        plt.xlabel("x1")
        plt.ylabel("x2")

    def plot_cloud_points_label(self, **kwargs):
        successes = self.data_x[self.data_y[:, 0] == 1.]
        losses = self.data_x[self.data_y[:, 0] == 0.]
        successes_plt, = plt.plot(successes[:, 0], successes[:, 1], "bs", color="b", ms=6, **kwargs)
        losses_plt, = plt.plot(losses[:, 0], losses[:, 1], "^", color='lawngreen', ms=6, **kwargs)
        return successes_plt, losses_plt

    def plot_affine_boundary(self, x_min=-10, x_max=10, **kwargs):
        slope = - self.w[0, 0] / self.w[1, 0]
        intercept = - self.b / self.w[1, 0]
        xx = np.linspace(x_min, x_max, 2)
        yy = slope * xx + intercept
        return plt.plot(xx, yy, ls='-', lw=2, color="gray", **kwargs)

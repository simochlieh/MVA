"""
This class implements a linear regression to a 2D model,
with points X and labels Y in {0, 1}. The boundary we will learn is
in the form f(x) = transpose(w) * x + b affine (with a constant term)
"""
import numpy as np

from base_classification import BaseClassification


class LinearRegression(BaseClassification):

    def __init__(self, data_x, data_y, data_x_test, data_y_test, dataset_name, lambda_val=0):
        super(LinearRegression, self).__init__(data_x, data_y, data_x_test, data_y_test, dataset_name)

        # constant to penalize w
        self.lambda_val = lambda_val

        # Learnt variance
        self.variance = None

        # adding an extra column to the matrix in order to include the constant term b in the model
        self.data_x = np.hstack((data_x, np.ones((data_x.shape[0], 1))))
        self.data_x_test = np.hstack((data_x_test, np.ones((data_x_test.shape[0], 1))))

        # Plot titles
        self.title_training_plot = "Linear Regression: Training"
        self.title_test_plot = "Linear Regression: Test"

    def train(self):
        nb_points = self.data_x.shape[0]
        w = np.linalg.solve(self.data_x.T.dot(self.data_x) + self.lambda_val * np.identity(self.data_x.shape[1]),
                            self.data_x.T.dot(self.data_y))

        # The boundary is defined as f(x) = transpose(w) * x + b = 0.5
        self.w = w[0:2, :]
        self.b = w[2, :] - 0.5
        self.variance = 1. / nb_points * (self.data_y - self.data_x.dot(w)).T.dot(self.data_y - self.data_x.dot(w))

    def plot(self, test_mode=False):
        super(LinearRegression, self).plot(test_mode)

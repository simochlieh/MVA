"""
This class implements a logistic regression to a 2D model,
with points X and labels Y in {0, 1}. The boundary we will learn is
in the form f(x) = transpose(w) * x + b affine (with a constant term)
"""
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

from base_classification import BaseClassification
import utils.adv_math as adv_math


class LogisticRegression(BaseClassification):

    def __init__(self, data_x, data_y, w0, nb_iterations=20, lambda_val=0):
        super(LogisticRegression, self). __init__(data_x, data_y)

        # Initial point of the iteration
        self.w0 = w0

        # Array of log-likelihoods at each step of the iteration
        self.log_likelihood = []

        self.nb_iterations = nb_iterations

        # Constant to penalize w
        self.lambda_val = lambda_val

        # adding a column of ones to include constant term
        self.data_x = np.hstack((data_x, np.ones((data_x.shape[0], 1))))


    def train(self):
        iteration = 0
        w = deepcopy(self.w0)
        log_likelihood = [self.compute_log_likelihood(self.w0)]

        while iteration < self.nb_iterations:
            try:
                search_dir = self.compute_newton_direction(w)
                w = w + search_dir
                log_likelihood.append(self.compute_log_likelihood(w))
                iteration += 1
                print "iteration %s" % iteration
            except OverflowError:
                print "Some computation gives back a double too large to be represented, most certainly in w"
                break
        self.w = w[0:2, :]
        self.b = w[2, 0]
        self.log_likelihood = log_likelihood

    def plot(self):
        super(LogisticRegression, self).plot()
        plt.title("Logistic Regression")

        plt.figure(2)
        plt.plot(xrange(len(self.log_likelihood)), self.log_likelihood, color='black')
        plt.xlabel("Number of iterations")
        plt.ylabel("Log-Likelihood")
        plt.title("Logistic Regression")

        plt.show()

    def compute_log_likelihood(self, w):
        return self.data_y.T.dot(adv_math.LOG(adv_math.SIGMOID_FUNC(self.data_x.dot(w))))[0, 0] + \
               (1. - self.data_y).T.dot(adv_math.LOG(adv_math.SIGMOID_FUNC(- self.data_x.dot(w))))[0, 0]

    def compute_newton_direction(self, w):

        # vectors of sigmoid applied to every data point
        sigmoid_vec = adv_math.SIGMOID_FUNC(self.data_x.dot(w))

        # the gradient vector
        gradient = self.data_x.T.dot(self.data_y - sigmoid_vec) - 2 * self.lambda_val * w

        # the hessian matrix
        diagonal = np.diag(sigmoid_vec[:, 0]).dot(np.diag(1 - sigmoid_vec[:, 0])) - np.diag(2 * self.lambda_val * w.T)
        hessian = self.data_x.T.dot(diagonal).dot(self.data_x)

        # compute search direction
        search_dir = np.linalg.inv(hessian).dot(gradient)

        return search_dir

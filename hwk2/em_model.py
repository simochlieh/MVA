from copy import deepcopy
import numpy as np
import math
import matplotlib.pyplot as plt

from utils import adv_math, data_viz


CHI_SQUARE_90 = 4.605
NB_POINTS_AREA_PER_UNIT = 5


class EmModel:

    def __init__(self, data, k, mu_0, sigma_0, pi_0, nb_iter, sigma_prop_identity=False):
        """

        :param data: np array of shape (nb_rows, dimension)
        :param k: the number of clusters
        :param mu_0: Initial value of the means for every cluster
                (a numpy array of shape (k, dimension))
        :param sigma_0: Initial value of the co-variance matrix for every cluster
                (a list of k numpy array of shape (2, 2))
        :param pi_0: Initial values for the parameters of the multinomial distribution of the clusters
        :param sigma_prop_identity: boolean, False by default, True if you want to suppose that sigma is proportional
                to identity
        """
        self.data = data
        self.n = data.shape[0]
        self.dim = data.shape[1]
        self.k = k
        self.mu_0 = mu_0
        self.sigma_0 = sigma_0
        self.pi_0 = pi_0
        self.nb_iter = nb_iter
        self.sigma_prop_identity = sigma_prop_identity
        self.log_likelihood = []

        # Initializing the parameters we want to learn
        self.mu = deepcopy(self.mu_0)
        self.sigma = deepcopy(self.sigma_0)
        self.pi = deepcopy(self.pi_0)
        self.q = np.zeros(shape=(self.n, self.k))

    def run(self):
        self.log_likelihood.append(self.compute_log_likelihood())
        for i in xrange(self.nb_iter):
            print "iteration %d" % i
            self.expectation_step()
            self.maximization_step()
            self.log_likelihood.append(self.compute_log_likelihood())

    def expectation_step(self):
        for i in xrange(self.n):
            # Computing the sum of gaussian densities
            sum_ = 0.
            for k in xrange(self.k):
                sum_ += self.pi[k] * adv_math.compute_gaussian(self.data[i, :].T, self.mu[k, :].T, self.sigma[k])

            for k in xrange(self.k):
                self.q[i, k] = self.pi[k] * adv_math.compute_gaussian(self.data[i, :].T, self.mu[k, :].T,
                                                                      self.sigma[k]) \
                               / sum_

    def maximization_step(self):
        sum_q = np.sum(self.q)
        for k in xrange(self.k):
            sum_q_k = np.sum(self.q[:, k])
            self.mu[k, :] = self.q[:, k].T.dot(self.data) / sum_q_k
            if self.sigma_prop_identity:
                # Computing a useful sum
                sum_ = 0.
                for i in xrange(self.n):
                    x_i = self.data[i, :].reshape((2, 1))
                    mu_k = self.mu[k, :].reshape((2, 1))
                    sum_ += (x_i.T - mu_k.T).dot(x_i - mu_k) * self.q[i, k]
                self.sigma[k] = 1. / 2. * sum_ / sum_q_k * np.identity(self.dim)
                # print k
                # print self.sigma[k]
            else:
                # Computing a useful sum
                sum_i = np.zeros(shape=(2, 2))
                for i in xrange(self.n):
                    x_i = self.data[i, :].reshape((2, 1))
                    mu_k = self.mu[k, :].reshape((2, 1))
                    sum_i = sum_i + (x_i - mu_k).dot(x_i.T - mu_k.T) * self.q[i, k]
                self.sigma[k] = sum_i / sum_q_k
            self.pi[k] = sum_q_k / sum_q

    def compute_log_likelihood(self, data_test=None):
        if data_test is None:
            data = self.data
        else:
            data = data_test
        sum_i = 0
        for i in xrange(self.n):
            sum_k = 0
            for k in xrange(self.k):
                sum_k += self.pi[k] * adv_math.compute_gaussian(data[i, :].T, self.mu[k, :].T, self.sigma[k])
            sum_i += math.log(sum_k)
        return sum_i

    def plot(self, title, data_test=None):
        if data_test is None:
            data = self.data
        else:
            data = data_test
        datapoints = []
        datapoints_label = []
        colors_points = ["b", "r", "g", "y"]
        colors_centers = ["cyan", "magenta", "lime", "gold"]
        centers = []
        centers_label = []

        plt.axis('equal')

        # First plotting the raw data
        for k in xrange(self.k):
            plt.plot(data[:, 0], data[:, 1], colors_points[k] + "s", ms=3)

        # Representing the areas of dominance for every cluster
        xmin, xmax = plt.xlim()
        ymin, ymax = plt.ylim()

        x = np.linspace(xmin, xmax, num=(xmax - xmin) * NB_POINTS_AREA_PER_UNIT)
        y = np.linspace(ymin, ymax, num=(ymax - ymin) * NB_POINTS_AREA_PER_UNIT)

        xx, yy = np.meshgrid(x, y)
        grid = np.c_[xx.ravel(), yy.ravel()]
        q = np.zeros(shape=(grid.shape[0], self.k))
        print "Calculating the area of dominance for every cluster..."
        for i in xrange(grid.shape[0]):
            # Computing the sum of gaussian densities
            sum_ = 0.
            for k in xrange(self.k):
                sum_ += self.pi[k] * adv_math.compute_gaussian(grid[i, :].T, self.mu[k, :].T, self.sigma[k])

            for k in xrange(self.k):
                q[i, k] = self.pi[k] * adv_math.compute_gaussian(grid[i, :].T, self.mu[k, :].T,
                                                                      self.sigma[k]) \
                               / sum_
        clusters_per_meshgrid_point = np.argmax(q, axis=1)

        # Plotting the latent variables learnt in the EM
        for k in xrange(self.k):
            datapoint, = plt.plot(grid[clusters_per_meshgrid_point == k, 0], grid[clusters_per_meshgrid_point == k, 1],
                                  's', color=colors_centers[k], ms=4, alpha=0.1)
            datapoints.append(datapoint)
            datapoints_label.append("Area of dominance of cluster %s" % k)

            center, = plt.plot(self.mu[k, 0], self.mu[k, 1], "^", color=colors_centers[k], ms=10)
            centers.append(center)

            data_viz.plot_cov_ellipse(self.sigma[k], self.mu[k, :], nstd=math.sqrt(CHI_SQUARE_90),
                                      color=colors_centers[k], lw=2)

            centers_label.append("center of  cluster %s" % k)

        plt.legend(datapoints + centers, datapoints_label + centers_label)
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(title)
        plt.show()

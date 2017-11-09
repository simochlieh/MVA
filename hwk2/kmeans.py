import numpy as np
import pandas as pnd
import matplotlib.pyplot as plt


class Kmeans:

    def __init__(self, data, k, max_nb_iter, verbose=False):
        """

        :param data: numpy array of shape (nb_rows, dimension)
        :param k: number of clusters
        :param max_nb_iter: max number of iterations
        """
        self.data = data
        self.n = self.data.shape[0]
        self.dim = self.data.shape[1]
        self.k = k
        self.max_nb_iter = max_nb_iter

        assert self.k < self.n, "Number of clusters should be less than the number of points"

        # The k centers we want to estimate of shape (d, 1)
        self.centers = None

        # The closest centers to every point
        self.closest_centers = pnd.DataFrame(np.hstack((np.zeros(shape=(self.n, 1)), self.data)))
        self.closest_centers.columns = ['cluster', 'x1', 'x2']

        self.distortion = None

        # The indicator matrix: z[i,k] = 1 if the point i is in cluster k, 0 otherwise
        self.z = np.zeros(shape=(self.n, self.k))

        self.verbose = verbose

    def initialize(self):
        centers_index = np.random.choice(self.n, self.k, replace=False)
        self.centers = pnd.DataFrame(self.data[centers_index, :]).reset_index()
        self.centers.columns = ['cluster', 'x1', 'x2']

    def assign_point_to_closest_center(self):
        distance_matrix = self.compute_distance_matrix()
        self.closest_centers.loc[:, 'cluster'] = distance_matrix.argmin(axis=1)

    def compute_distance_matrix(self):
        """
        Broadcasting the data in order to have the distance from 1 point to every center
        """
        return np.square(self.data[:, np.newaxis, :] - self.centers[['x1', 'x2']].as_matrix()).sum(axis=2)

    def recompute_centroids(self):
        """

        :return: updates self.centers if the new centers are different and returns is_done = False,
                    otherwise returns is_done = True
        """
        is_done = False
        new_centers = self.closest_centers.groupby('cluster').agg({'x1': 'mean', 'x2': 'mean'}).reset_index()
        if new_centers[['cluster', 'x1', 'x2']].equals(self.centers[['cluster', 'x1', 'x2']]):
            is_done = True
            return is_done
        self.centers = new_centers
        return is_done

    def run(self):
        self.initialize()
        iter_ = 0
        while iter_ < self.max_nb_iter:
            iter_ += 1
            self.assign_point_to_closest_center()
            is_done = self.recompute_centroids()

            if is_done:
                break
        self.compute_distortion()
        if self.verbose:
            print "After %d iterations, the distortion measured is %d" % (iter_, self.distortion)

    def plot(self):
        datapoints, = plt.plot(self.data[:, 0], self.data[:, 1], "bs", ms=3)
        center = None
        for k in xrange(self.k):
            center, = plt.plot(self.centers.loc[k, 'x1'], self.centers.loc[k, 'x2'], "r^", ms=6)

        plt.legend([datapoints, center], ["datapoints", "centers"])
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()

    def compute_distortion(self):
        distance_matrix = self.compute_distance_matrix()
        self.compute_indicator_matrix()
        self.distortion = np.sum(np.multiply(self.z, distance_matrix))

    def compute_indicator_matrix(self):
        for i in xrange(self.n):
            for k in xrange(self.k):
                if self.closest_centers.loc[i, 'cluster'] == k:
                    self.z[i, k] = 1

"""
Unittests for development purposes
"""
import numpy as np
import unittest

from hwk1.lda import LDA


class TestMleEstimates(unittest.TestCase):

    def test_mle_estimates(self):
        data_x = np.array([[1., 2.], [3., 4.], [2., 3.], [-1., 1.], [-3., 1.], [-3., 3.]])
        data_y = np.array([[1.], [1.], [1.], [0.], [0.], [0.]])
        lda = LDA(data_x, data_y)
        lda.train()
        self.assertEqual(lda.pi, 0.5, "Pi estimate is wrong. Found %s" % lda.pi)
        self.assertEqual(lda.mu_0.all(),
                         np.array([[7./3., 5./3.]]).all(),
                         "mu_0 estimate is wrong. Found %s" % lda.mu_0)
        self.assertEqual(lda.mu_1.all(),
                         np.array([[2., 3.]]).all(),
                         "mu_1 estimate is wrong. Found %s" % lda.mu_1)


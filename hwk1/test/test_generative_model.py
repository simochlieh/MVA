"""
Unittests for development purposes
"""
import numpy as np
import unittest

import hwk1.generative_model as gm


class TestMleEstimates(unittest.TestCase):

    def test_mle_estimates(self):
        data_x = np.array([[1., 2.], [3., 4.], [2., 3.], [-1., 1.], [-3., 1.], [-3., 3.]])
        data_y = np.array([[1.], [1.], [1.], [0.], [0.], [0.]])
        mle_estimates = gm.get_mle_estimates(data_x, data_y)
        self.assertEqual(mle_estimates[0], 0.5, "Pi estimate is wrong. Found %s" % mle_estimates[0])
        self.assertEqual(mle_estimates[1][0].all(),
                         np.array([[7./3., 5./3.]]).all(),
                         "mu_0 estimate is wrong. Found %s" % mle_estimates[1][0])
        print mle_estimates[2]
        self.assertEqual(mle_estimates[1][1].all(),
                         np.array([[2., 3.]]).all(),
                         "mu_1 estimate is wrong. Found %s" % mle_estimates[1][1])


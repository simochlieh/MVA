"""
Main point of entry to simulate the different classification models
"""
import os
import numpy as np

import utils.data_processing as data
from lda import LDA
from logistic_regression import LogisticRegression
from linear_regression import LinearRegression


def main(dataset):
    path = "classification_data_HWK1/classification_data_HWK1/classification%s.train" % dataset
    data_x, data_y = data.parse_data_with_labels(os.path.abspath(path), dimension=2, delimiter="\t")

    # LDA
    lda_model = LDA(data_x, data_y)
    lda_model.train()

    print "The bernoulli parameter pi is \n%s\n" % lda_model.pi
    print "The mean for the class {y=0} is \n%s\n" % lda_model.mu_0
    print "The mean for the class {y=1} is \n%s\n" % lda_model.mu_1
    print "Sigma is: \n%s" % lda_model.sigma

    lda_model.plot()

    # Logistic Regression
    # Adding an extra column to the matrix in order to include the constant term b in the model
    w0 = np.array([[0, 0, 1]]).T
    logistic_reg = LogisticRegression(data_x, data_y, w0, nb_iterations=20, lambda_val=0.1)
    logistic_reg.train()
    logistic_reg.plot()

    print "\nThe learnt parameter w is: \n%s\n" % logistic_reg.w

    # Linear Regression
    lin_reg = LinearRegression(data_x, data_y, lambda_val=0)
    lin_reg.train()

    print "\nThe learnt parameter w is: \n%s\n" % lin_reg.w
    print "The variance of Y computed for the latter w is: \n%s\n" % lin_reg.variance

    lin_reg.plot()


if __name__ == '__main__':
    dataset = raw_input("Choose Dataset between A, B, C: ")
    main(dataset)

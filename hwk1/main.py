"""
Main point of entry to simulate the different classification models
"""
import os
import sys
import numpy as np

sys.path.append("../")

import utils.data_processing as data
from lda import LDA
from logistic_regression import LogisticRegression
from linear_regression import LinearRegression


def main(dataset):
    path_train = "classification_data_HWK1/classification_data_HWK1/classification%s.train" % dataset
    path_test = "classification_data_HWK1/classification_data_HWK1/classification%s.test" % dataset
    data_x_train, data_y_train = data.parse_data_with_labels(os.path.abspath(path_train), dimension=2, delimiter="\t")
    data_x_test, data_y_test = data.parse_data_with_labels(os.path.abspath(path_test), dimension=2, delimiter="\t")

    # LDA
    lda_model = LDA(data_x_train, data_y_train)
    lda_model.train()

    print "\nLDA:\n"
    print "The bernoulli parameter pi is \n%s\n" % lda_model.pi
    print "The mean for the class {y=0} is \n%s\n" % lda_model.mu_0
    print "The mean for the class {y=1} is \n%s\n" % lda_model.mu_1
    print "Sigma is: \n%s\n" % lda_model.sigma

    lda_model.plot()

    print "Misclassification error is: %.2f %%\n" % (lda_model.compute_misclassification_err(data_x_test, data_y_test) * 100)
    lda_model.plot(data_x_test, data_y_test)

    # Logistic Regression
    print "\nLogistic Regression:\n"
    # Adding an extra column to the matrix in order to include the constant term b in the model
    w0 = np.array([[0, 0, 1]]).T
    logistic_reg = LogisticRegression(data_x_train, data_y_train, w0, nb_iterations=20, lambda_val=0.1)
    logistic_reg.train()

    print "\nThe learnt parameter w is: \n%s\n" % logistic_reg.w
    logistic_reg.plot()

    print "Misclassification error is: %.2f %%\n" % (logistic_reg.compute_misclassification_err(data_x_test, data_y_test) * 100.)
    logistic_reg.plot(data_x_test, data_y_test)

    # Linear Regression
    print("\nLinear Regression\n")
    lin_reg = LinearRegression(data_x_train, data_y_train, lambda_val=0)
    lin_reg.train()

    print "The learnt parameter w is: \n%s\n" % lin_reg.w
    print "The variance of Y computed for the latter w is: \n%s\n" % lin_reg.variance

    lin_reg.plot()

    print "Misclassification error is: %.2f %%\n" % (lin_reg.compute_misclassification_err(data_x_test, data_y_test) * 100.)
    lin_reg.plot(data_x_test, data_y_test)


if __name__ == '__main__':
    dataset_prompt = raw_input("Choose Dataset between A, B, C: ")
    main(dataset_prompt)

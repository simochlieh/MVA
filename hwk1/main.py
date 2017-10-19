"""
Main point of entry to simulate the different classification models
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")

import utils.data_processing as data
from lda import LDA
from qda import QDA
from logistic_regression import LogisticRegression
from linear_regression import LinearRegression


def main():
    datasets = ['A', 'B', 'C']
    lda_models = {}
    qda_models = {}
    logistic_reg_models = {}
    lin_reg_models = {}
    for dataset in datasets:
        path_train = "classification_data_HWK1/classification_data_HWK1/classification%s.train" % dataset
        path_test = "classification_data_HWK1/classification_data_HWK1/classification%s.test" % dataset
        data_x_train, data_y_train = data.parse_data_with_labels(os.path.abspath(path_train), dimension=2, delimiter="\t")
        data_x_test, data_y_test = data.parse_data_with_labels(os.path.abspath(path_test), dimension=2, delimiter="\t")

        # LDA
        lda_models[dataset] = LDA(data_x_train, data_y_train, data_x_test, data_y_test, dataset_name=dataset)
        lda_models[dataset].train()

        print "\nLDA_Dataset_%s:\n" % dataset
        print "The bernoulli parameter pi is \n%s\n" % lda_models[dataset].pi
        print "The mean for the class {y=0} is \n%s\n" % lda_models[dataset].mu_0
        print "The mean for the class {y=1} is \n%s\n" % lda_models[dataset].mu_1
        print "Sigma is: \n%s\n" % lda_models[dataset].sigma
        print "Training misclassification error is: %.2f %%\n" % (lda_models[dataset].compute_misclassification_err()[0] * 100)
        print "Test misclassification error is: %.2f %%\n" % (lda_models[dataset].compute_misclassification_err()[1] * 100)

        # Logistic Regression
        print "\nLogistic_Regression_Dataset_%s:\n" % dataset
        w0 = np.array([[0, 0, 1]]).T
        logistic_reg_models[dataset] = LogisticRegression(data_x_train, data_y_train, w0, data_x_test, data_y_test,
                                                          dataset_name=dataset, nb_iterations=20, lambda_val=0.01)
        logistic_reg_models[dataset].train()

        print "\nThe learnt parameter w is: \n%s\n" % logistic_reg_models[dataset].w
        print "\nThe learnt parameter b is: \n%s\n" % logistic_reg_models[dataset].b
        print "Training misclassification error is: %.2f %%\n" % (logistic_reg_models[dataset].compute_misclassification_err()[0] * 100.)
        print "Test misclassification error is: %.2f %%\n" % (logistic_reg_models[dataset].compute_misclassification_err()[1] * 100.)

        # Linear Regression
        print "\nLinear_Regression_Dataset_%s\n" % dataset
        lin_reg_models[dataset] = LinearRegression(data_x_train, data_y_train, data_x_test, data_y_test,
                                          dataset_name=dataset, lambda_val=0)
        lin_reg_models[dataset].train()

        print "The learnt parameter w is: \n%s\n" % lin_reg_models[dataset].w
        print "\nThe learnt parameter b is: \n%s\n" % logistic_reg_models[dataset].b
        print "The variance of Y computed for the latter w is: \n%s\n" % lin_reg_models[dataset].variance

        print "Training misclassification error is: %.2f %%\n" % (lin_reg_models[dataset].compute_misclassification_err()[0] * 100.)
        print "Test misclassification error is: %.2f %%\n" % (lin_reg_models[dataset].compute_misclassification_err()[1] * 100.)

        # QDA
        qda_models[dataset] = QDA(data_x_train, data_y_train, data_x_test, data_y_test, dataset_name=dataset)
        qda_models[dataset].train()

        print "\nQDA_Dataset_%s:\n" % dataset
        print "The bernoulli parameter pi is \n%s\n" % qda_models[dataset].pi
        print "The mean for the class {y=0} is \n%s\n" % qda_models[dataset].mu_0
        print "The mean for the class {y=1} is \n%s\n" % qda_models[dataset].mu_1
        print "Sigma for the class {y=0} is: \n%s\n" % qda_models[dataset].sigma_0
        print "Sigma for the class {y=1} is: \n%s\n" % qda_models[dataset].sigma_1
        print "Training misclassification error is: %.2f %%\n" % (qda_models[dataset].compute_misclassification_err()[0] * 100)
        print "Test misclassification error is: %.2f %%\n" % (qda_models[dataset].compute_misclassification_err()[1] * 100)

    for model in [lda_models, logistic_reg_models, lin_reg_models, qda_models]:
        plt.subplot(221)
        model['A'].plot()
        plt.subplot(222)
        model['B'].plot()
        plt.subplot(212)
        model['C'].plot()

        plt.show()

        if model == logistic_reg_models:
            plt.subplot(221)
            model['A'].plot_convergence_func()
            plt.subplot(222)
            model['B'].plot_convergence_func()
            plt.subplot(212)
            model['C'].plot_convergence_func()

            plt.show()

        plt.subplot(221)
        model['A'].plot(test_mode=True)
        plt.subplot(222)
        model['B'].plot(test_mode=True)
        plt.subplot(212)
        model['C'].plot(test_mode=True)

        plt.show()

if __name__ == '__main__':
    main()

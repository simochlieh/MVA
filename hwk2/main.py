import sys
sys.path.append("../")
import numpy as np
import matplotlib.pyplot as plt

from utils import data_processing as dp
from kmeans import Kmeans
from em_model import EmModel

NB_CLUSTERS = 4
MAX_K_MEAN_ITER = 50
NB_INITIALIZATION_RETRIES = 20
NB_ITER = {True: 60,
           False: 15}


def run_em_model(data, data_test, best_kmean_model, sigma_prop_identity):

    centers, sigma_0, pi_0 = init_param_values(data, best_kmean_model, sigma_prop_identity)
    # Creating the EM model
    em_model = EmModel(data, data_test, 4, mu_0=centers,
                         sigma_0=sigma_0, pi_0=pi_0, nb_iter=NB_ITER[sigma_prop_identity],
                         sigma_prop_identity=sigma_prop_identity)

    # Training the model
    title = "Case where covariance matrix is proportional to identity" if sigma_prop_identity \
        else "General Case"
    print '\n' + title + ':\n'
    em_model.run()

    # Plotting log-likelihood
    plt.plot(range(1, len(em_model.log_likelihood) + 1), em_model.log_likelihood)
    plt.xlabel("Iteration number")
    plt.ylabel("Log-Likelihood")
    plt.title("Log-Likelihood: " + title)
    plt.show()

    # Plotting the results and the learnt parameters
    em_model.plot(title="Results: " + title + " (training)")
    # And for the test data
    em_model.plot("Results: " + title + " (test)", test=True)

    # Printing log-likelihood for test data
    print "The Log-Likelihood for the test data is %.2f" % em_model.compute_log_likelihood(test=True)


def init_param_values(data, best_kmean_model, sigma_prop_identity):
    # computing a matrix of centers from the dataframe
    centers = best_kmean_model.centers[['x1', 'x2']].as_matrix()

    n = data.shape[0]
    d = data.shape[1]
    # Computing the initial values sigma_0, mu_0, pi_0
    sigma_0 = []
    pi_0 = []

    # The indicator matrix: z[i,k] = 1 if the point i is in cluster k, 0 otherwise
    sum_z = np.sum(best_kmean_model.z)
    for k in xrange(NB_CLUSTERS):
        sum_z_k = np.sum(best_kmean_model.z[:, k])
        if sigma_prop_identity:
            sum_i = 0.
            for i in xrange(n):
                sum_i += (data[i, :] - centers[k, :]).dot(data[i, :] - centers[k, :]) * best_kmean_model.z[i, k]
            sigma_0.append(1. / 2. * sum_i / sum_z_k * np.identity(d))
        else:
            # Computing a useful sum
            sum_i = np.zeros(shape=(d, d))
            for i in xrange(n):
                x_i = data[i, :].reshape((d, 1))
                mu_k = centers[k, :].reshape((d, 1))
                sum_i = sum_i + (x_i - mu_k).dot(x_i.T - mu_k.T) * best_kmean_model.z[i, k]
            sigma_0.append(sum_i / sum_z_k)
        pi_0.append(sum_z_k / sum_z)

    return centers, sigma_0, pi_0


def main():
    # Reading the training data
    path_train = './data/EMGaussian.data'
    path_test = './data/EMGaussian.test'
    data = dp.parse_data_wo_labels(path_train, 2, delimiter=' ')
    data_test = dp.parse_data_wo_labels(path_test, 2, delimiter=' ')

    # Initialization with K-means
    best_kmean_model = None
    min_distortion = float("inf")
    distortions = []
    for i in xrange(NB_INITIALIZATION_RETRIES):
        kmean_model = Kmeans(data, NB_CLUSTERS, MAX_K_MEAN_ITER)
        kmean_model.run()
        distortions.append(kmean_model.distortion)

        if kmean_model.distortion < min_distortion:
            best_kmean_model = kmean_model
            min_distortion = kmean_model.distortion
    # Showing the distortions
    plt.plot(range(1, NB_INITIALIZATION_RETRIES + 1), distortions)
    plt.xlabel("Initialization number")
    plt.ylabel("Distortion")
    plt.title("Running few Kmeans and measuring the distortion for each")
    plt.show()

    # Plotting the result
    best_kmean_model.plot()

    # Case where the covariance matrix is proportional to identity
    run_em_model(data, data_test, best_kmean_model, sigma_prop_identity=True)

    # General Case
    run_em_model(data, data_test, best_kmean_model, sigma_prop_identity=False)


if __name__ == "__main__":
    main()

# Principle Component Analysis

from cmath import exp, pi, sqrt
from ctypes import sizeof
from random import random, randint
import numpy as np
import matplotlib.pyplot as plt

# randomly generate a p x n data matrix
# each row is a data point
# since this is an unsupervised learning problem, we don't need labels for data points
def generate_data(n, p):
    data = np.zeros((n, p))
    for i in range (n):
        for j in range (p):
            data[i, j] = randint(-400, 400) * random()
    return data

# standardlized data = data - mean
def standardlized_data (data):
    res = np.copy(data)
    mean = np.mean(data, axis = 0)
    return res - mean

# find eigenvectors of standardlized data
# input data in this function is standardlized
def eigen_vectors_of_covariance_matrix (std_data):
    # compute covariance matrix of standardlized data
    cov = np.cov(std_data.T) 

    # find eigenvalues and eigenvectors of standardlized data's covariance matrix 
    eig_val, eig_vec = np.linalg.eig(cov)

    # sorting:
    #    sort eigenvalues in descending order
    #    sort eigenvectors in descending eigenvalues orders
    order = np.flip(np.argsort(eig_val))
    for i in range (len(eig_vec[0])):
        eig_vec[:,i] = eig_vec[:,order[i]]
    eig_val = np.flip(np.sort(eig_val))

    return eig_vec

# multiply data matrix with K eigenvectors matrix
# This function turns (n x p) matrix to (n x K) matrix
def dimension_reduction (data, eig_vec, K):
    transform_matrix = eig_vec[:,0:K]
    res = data @ transform_matrix
    return res

# the main function
def Principal_Component_Analysis (data, K):
    standardlized_datas = standardlized_data(data)
    eig_vec = eigen_vectors_of_covariance_matrix(standardlized_datas)
    res = dimension_reduction(standardlized_datas, eig_vec, K)
    return res

# calculate proportion of variance explained
def proportion_of_variance_explained (data, reduced_data):
    # data in here is standardlized data
    truth_variance = np.sum(data ** 2)
    PCA_variance = np.sum(reduced_data ** 2)
    return 100.0 * PCA_variance / truth_variance

# demo
# you can adjust n,p, K and max_iteration anyway you want
if __name__ == '__main__':
    n = 400; p = 10; K = 2
    data = generate_data(n, p)
    res = Principal_Component_Analysis(data, K)
    PVE = proportion_of_variance_explained(data, res)
    print("Input data:\n", data)
    print("Output data:\n", res)
    print("PVE:", PVE, "%")

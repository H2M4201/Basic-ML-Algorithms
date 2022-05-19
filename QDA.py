# Quadratic Discriminant Analysis

from cmath import exp, pi, sqrt
from ctypes import sizeof
from random import random, randint
import numpy as np
import matplotlib.pyplot as plt

# randomly generate a n x (p+1) data matrix
# each column is a data point
# the (p+1)-th row is the label row
# K is the number of classes to classify
def generate_data(n, p, K):
    data = np.zeros((p + 1, n))
    for i in range (n):
        for j in range (p):
            data[j, i] = randint(-300, 300) * random()
            data[p, i] = randint(1, K)
    return data

# count classes' frequencies
def counting_class_frequency(data, K):
    n = len(data[0])
    p = len(data) - 1
    count = np.zeros(K, int)
    for i in range (n):
        count[int(data[p, i]) - 1] +=1
    return count

# calculate P(Y): probability of a data point belonging to each class
def estimate_prior_prob(data, K):
    count = counting_class_frequency(data, K)
    n = len(data[0])
    tmp = np.zeros(K)
    for i in range (K):
        tmp[i] = count[i] / n
    return tmp

# calculate mean value of i-th class data points
def mean_by_class(data, K):
    count = counting_class_frequency(data, K) 
    n = len(data[0])
    p = len(data) - 1
    mean = np.zeros((p, K))
    for i in range (p):
        for j in range (n):
            mean[i, int(data[p, j]) - 1] += data[i, j]
    
    for i in range (p):
        for j in range (K):
            if (count[j] > 0):
                mean[i, j] /= count[j]

    return mean

# calculate covariance matrix of i-th class elements
def covariance_by_class (data, K):
    n = len(data[0])
    p = len(data) - 1
    mean = mean_by_class(data, K)
    res = np.zeros((K, p, p))
    for i in range (n):
        k = int(data[p, i])
        x = data[0:p,i] - mean[:, k -  1]
        for j1 in range (p):
            for j2 in range (p):
                res[k - 1, j1, j2] += x[j1] * x[j2]
    
    size = counting_class_frequency(data, K)
    for i in range (K):
        res[i,:,:] /= size[i]
    
    return res

# main function
def quadratic_discriminant_analysis (data, K):
    n = len(data[0])
    p = len(data) - 1
    res = np.zeros(n)

    prob = estimate_prior_prob(data, K)
    mean = mean_by_class(data, K)
    cov = covariance_by_class(data, K)
    # inverse covariance matrix
    inverse_cov = np.zeros((K, p, p))
    for i in range (K):
        inverse_cov[i,:,:] = np.linalg.inv(cov[i,:,:])
    for i in range (n):
        QDA_i = np.zeros(K)
        for j in range (K):
            # Quadratic Discriminant
            QDA_i[j] = -0.5 * (data[0:p,i] - mean[:,j]).T @ inverse_cov[j,:,:] @ \
                (data[0:p,i] - mean[:,j]) \
                -0.5 *np.log(np.linalg.det(inverse_cov[j,:,:])) + np.log(prob[j])

        # assign data point to the class that has highest Quadratic Discriminant value
        res[i] = np.argmax(QDA_i) + 1

    return res

# calculate accuracy percentage
def accuracy (ground_truth, predict):
    # ground_truth: data's truth labels
    n = len(predict)
    s = 0.0
    for i in range (n):
        if ground_truth[i] == predict[i]:
            s += 1
    
    return 100.0 * s / n

# demo
# you can adjust n,p, K anyway you want
if __name__ =="__main__":
    K = 4; n = 100; p = 4
    data = generate_data(n, p, K)
    res = quadratic_discriminant_analysis(data, K)
    print("Input Data:\n", data)
    print("Truth Label:\n", data[p,:])
    print("Predicted label:\n", res)
    print("Accuracy:", accuracy(data[p,:], res), "%")
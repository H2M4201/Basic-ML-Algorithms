from cmath import exp, pi, sqrt
from ctypes import sizeof
from random import random, randint
import numpy as np
import matplotlib.pyplot as plt

def generate_data(n, p, K):
    data = np.zeros((p + 1, n))
    for i in range (n):
        for j in range (p):
            data[j, i] = randint(-300, 300) * random()
            data[p, i] = randint(1, K)
    return data

def counting_class_size(data, K):
    n = len(data[0])
    p = len(data) - 1
    count = np.zeros(K, int)
    for i in range (n):
        count[int(data[p, i]) - 1] +=1
    return count

def estimate_prior_prob(data, K):
    count = counting_class_size(data, K)
    n = len(data[0])
    tmp = np.zeros(K)
    for i in range (K):
        tmp[i] = count[i] / n
    return tmp

def mean_by_class(data, K):
    count = counting_class_size(data, K) 
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

def covariance_matrix (data, K):
    n = len(data[0])
    p = len(data) - 1
    mean = mean_by_class(data, K)
    res = np.zeros((p, p))
    for i in range (n):
        k = int(data[p, i])
        x = data[0:p,i] - mean[:, k -  1]
        for j1 in range (p):
            for j2 in range (p):
                res[j1, j2] += x[j1] * x[j2]
    
    return res / (n - K)

def linear_discriminant_analysis (data, K):
    n = len(data[0])
    p = len(data) - 1
    res = np.zeros(n)
    prob = estimate_prior_prob(data, K)
    mean = mean_by_class(data, K)
    cov = covariance_matrix(data, K)
    inverse_cov = np.linalg.inv(cov)
    print(cov)
    print(inverse_cov)
    for i in range (n):
        LDA_i = np.zeros(K)
        for j in range (K):
            LDA_i[j] = data[0:p,i].T @ inverse_cov @ mean[:,j] \
                - 0.5 * mean[:, j].T @ inverse_cov @ mean[:, j]  \
                    + np.log(np.float64(prob[j]))
        res[i] = np.argmin(LDA_i) + 1

    return res

def accuracy (ground_truth, predict):
    n = len(predict)
    s = 0.0
    for i in range (n):
        if ground_truth[i] == predict[i]:
            s += 1
    
    return s / n

if __name__ =="__main__":
    K = 4; n = 100; p = 4
    data = generate_data(n, p, K)
    res = linear_discriminant_analysis(data, K)
    print(data[p,:])
    print(res)
    print(accuracy(data[p, :], res))


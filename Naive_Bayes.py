# Naive Bayes
"""
Note: In this code, I assume that the i-th class, j-th features
has Gaussian distribution for ease of implementation; so the accuracy
might be quite low.
You could replace Gaussian distribution function with more precise 
probabilistic density functions to improve accuracy
"""

from math import exp, pi, sqrt
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
def counting_class_size(data, K):
    n = len(data[0])
    p = len(data) - 1
    count = np.zeros(K, int)
    for i in range (n):
        count[int(data[p, i]) - 1] +=1
    return count

# calculate P(Y): probability of a data point belonging to each class
def estimate_prior_prob(data, count, K):
    n = len(data[0])
    tmp = np.zeros(K)
    for i in range (K):
        tmp[i] = count[i] / n
    return tmp

# calculate mean value of i-th class, j-th data features
def mean_by_class_and_predictor(data, count, K):
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

# calculate variance value of i-th class, j-th data features
def variance_by_class_and_predictor(data, mean, K):
    n = len(data[0])
    p = len(data) - 1
    variance = np.zeros((p, K))
    count = np.zeros(K)
    for i in range (p):
        for j in range (n):
            variance[i, int(data[p, j]) - 1] += (data[i, j] - mean[i, int(data[p, j]) - 1]) ** 2
            if i == 0:
                count[int(data[p, j]) - 1] += 1
    
    for i in range (p):
        for j in range (K):
            if (count[j] > 0):
                variance[i, j] /= count[j]

    return variance

def Gaussian_probability_density (x, mean, sd):
    if (sd == 0):
        sd = 0.0000001
    a = -(x - mean) * (x - mean)
    b = 2 * sd * sd
    return np.float64(exp(a / b) / (sd * sqrt(2 * pi)))

# main function
def naive_bayes_classification (data, K):
    n = len(data[0])
    p = len(data) - 1
    count = counting_class_size(data, K)
    prior_prob = estimate_prior_prob(data, count, K)
    mean = mean_by_class_and_predictor(data, count, K)
    v = variance_by_class_and_predictor(data, mean, K)
    res = np.zeros(n)
    for i in range (n):
        classify_result = np.zeros(K)
        s = 0.0
        for j in range (K):
            tmp = 1.0
            for k in range (p):
                sd = sqrt(v[k, j])
                if (sd == 0):
                    sd = 0.0001
                tmp *= Gaussian_probability_density( \
                    data[k, i], mean[k ,j], sd)
            s += prior_prob[j] * tmp
            classify_result[j] = tmp
        
        for j in range (K):
            classify_result[j] /= s

        res[i] = np.argmax(classify_result) + 1

    return res

# calculate accuracy percentage
def accuracy (ground_truth, predict):
    n = len(predict)
    s = 0.0
    for i in range (n):
        if ground_truth[i] == predict[i]:
            s += 1
    
    return 100.0* s / n

# demo
# you can adjust n,p, K anyway you want
if __name__ == "__main__":
    K = 15
    n = 400
    p = 10
    data = generate_data(n, p, K)
    res = naive_bayes_classification(data, K)
    print("Input Data:\n", data)
    print("Truth Label:\n", data[p,:])
    print("Predicted label:\n", res)
    print("Accuracy:", accuracy(data[p,:], res), "%")
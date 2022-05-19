# An PCS-based algorithm to fill up missing values of a matrix (dataset)

from cmath import exp, isnan, nan, pi, sqrt
from ctypes import sizeof
from random import random, randint
import numpy as np
import matplotlib.pyplot as plt
import PCA as pca

# randomly generate a p x n data matrix
# each row is a data point
# since this is an unsupervised learning problem, we don't need labels for data points
# after data is generated, we randomly delete parts of data set
def generate_data(n, p):
    data = np.zeros((n, p))
    for i in range (n):
        for j in range (p):
            data[i, j] = randint(-400, 400) * random()

    # delete rate: 20% - 30%
    # You can fix this rate anyway you want
    count = randint(int(0.2*n*p), int(0.3*n*p))
    is_deleted = np.zeros((n, p))
    for loops in range (count):
        i = randint(0, n - 1)
        j = randint(0, p - 1)
        if is_deleted[i, j] == 0:
            data[i, j] = None
            is_deleted[i, j] = 1
        else:
            loops -= 1

    percent = 100.0 * count / (n*p)
    print("Empty Percentage:", percent, "%")

    return data

# Mark down original empty positions 
def empty_marking_matrix (data):
    n = len(data)
    p = len(data[0])
    res = np.zeros((n, p))
    for i in range (n):
        for j in range (p):
            if isnan(data[i, j]) == True:
                res[i, j] = 1
    return res

# calculate mean of values that are non-empty in a column
def calculate_non_empty_mean (data):
    n = len(data)
    p = len(data[0])
    res = np.zeros(p)
    for j in range (p):
        count = 0; s = 0
        for i in range (n):
            if isnan(data[i, j]) == False:
                count += 1
                s += data[i, j]
        res[j] = 1.0 * s / count

    return res

# fill up empty values with its column's mean value
def fill_empty_values_by_mean (data):
    mean = calculate_non_empty_mean(data)
    n = len(data)
    p = len(data[0])
    for i in range (n):
        for j in range (p):
            if isnan(data[i, j]) == True:
                data[i, j] = mean[j]

    return data

# fill up empty values with dot product of its PCA-value and eigenvector
def assign_empty_values_by_PCA (data, mark_matrix, PCA_data, eig_vec):
    n = len(data)
    p = len(data[0])
    for i in range (n):
        for j in range (p):
            if mark_matrix[i, j] == 1:
                data[i,j] = np.dot(PCA_data[i,:], eig_vec[j,:])
            else:
                continue
    return data

# loss function: total residual of original non-empty values and their
#  dot product of PCA-value and eigenvector
def compute_total_residual (data, mark_matrix, PCA_data, eig_vec):
    res = 0.0
    n = len(data)
    p = len(data[0])
    for i in range (n):
        for j in range (p):
            if mark_matrix[i, j] == 0:
                res += (data[i, j] - np.dot(PCA_data[i,:], eig_vec[j,:])) ** 2

    return res

# main function
# input data is a matrix with empty cells
# max_iteration is added to avoid infinite loop
# stop condition: max_iteration is reached or total residual can't decrease anymore
def matrix_completion (data, K, max_iteration):
    mark_matrix = empty_marking_matrix(data)
    data = fill_empty_values_by_mean(data)
    prev_total_residual = 0.0
    total_residual = 0.0
    i = 0
    for i in range (max_iteration):
        std_data = pca.standardlized_data(data)
        eig_vec = pca.eigen_vectors_of_covariance_matrix(std_data)[:,0:K]
        reduced = pca.dimension_reduction(std_data, eig_vec, K)
        copy_data = assign_empty_values_by_PCA(data, mark_matrix, reduced, eig_vec)
        total_residual = compute_total_residual(copy_data, mark_matrix, reduced, eig_vec)
        if i > 0 and total_residual > prev_total_residual:
            break
        else:
            data = np.copy(copy_data)
            prev_total_residual = total_residual

    print("Number of iteration:", i + 1)
            
    return data

# demo
# you can adjust n,p, K and max_iteration anyway you want
# K: number of principle components
if __name__ == "__main__":
    n = 1619; p = 12; K = 3
    max_iteration = 100
    data = generate_data(n, p)
    print("Input matrix:\n", data)
    data = matrix_completion(data, K, max_iteration)
    print("Filled data:\n", data)

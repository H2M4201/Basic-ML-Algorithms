# K-mean clustering

from cmath import sqrt
from random import random, randint
import numpy as np
import matplotlib.pyplot as plt

# randomly generate a p x n data matrix
# each row is a data point
# since this is an unsupervised learning problem, we don't need label column
def generate_data(n, p):
    data = np.zeros((n, p))
    for i in range (n):
        for j in range (p):
            data[i, j] = randint(-400, 400) * random()
    return data

# randomly generate clusters' centroids at first 
def initialize_centroids (p, K):
    res = np.zeros((K, p))
    for i in range (K):
        for j in range (p):
            res[i, j] = randint(-400, 400) * random()
    
    return res

# assign data points to the appropriate cluster centroid
def assign_data_point_to_clusters (data, centers):
    n = len(data); K = len(centers)
    distance = np.zeros((n, K))
    res = np.zeros(n)
    for i in range (n):
        for j in range (K):
            distance[i, j] = np.linalg.norm(data[i, :] - centers[j])
        res[i] = np.argmin(distance[i,:])

    return res

# new centroids = mean of all data points in a cluster
def calculate_new_centroids (data, K, cluster_division):
    n =len(data); p = len(data[0])
    new_centroids = np.zeros((K, p))
    count = np.zeros(K)
    for i in range (n):
        cluster_index = int(cluster_division[i])
        new_centroids[cluster_index, :] += data[i, :]
        count[cluster_index] += 1

    for i in range (K):
        if count[i] > 0:
            new_centroids[i, :] /= count[i]

    return new_centroids

# the main function
# max_iteration is added to avoid infinite loop
def K_means_clustering (data, K, max_iteration):
    n =len(data); p = len(data[0]) 
    res = np.zeros(n)
    centroids = initialize_centroids(p, K)
    for loops in range (max_iteration):
        cluster_division = assign_data_point_to_clusters(data, centroids)
        new_centroids = calculate_new_centroids(data, K, cluster_division)
        if (centroids == new_centroids).all() == True:
            break
        else:
            centroids = np.copy(new_centroids)
            res = np.copy(cluster_division)

    return centroids, res

# demo
# you can adjust n,p, K and max_iteration anyway you want
if __name__ == "__main__":
    n = 200; p = 15; K = 6; max_iteration = 1000
    data = generate_data(n, p)
    print("Input data:\n", data)
    centroids, clustering = K_means_clustering(data, K, max_iteration)
    print("Cluster centers:\n", centroids)
    print("Clustering result:\n", clustering)

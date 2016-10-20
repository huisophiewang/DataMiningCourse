import random
import math
from pprint import pprint

# installed packages
import numpy as np
import matplotlib.pyplot as plt

# utilities functions
from utilities import *


def random_initalize_clusters(data, k):
    clusters = []
    for i in range(data.shape[0]):
        c = random.randint(0, k-1)
        clusters.append(c)
    return clusters

def find_centers(data, clusters, k):
    n = data.shape[0]
    m = data.shape[1]
    
    centers = np.zeros([k, m-1])   
    nums = np.zeros(k)
    
    for i, c in enumerate(clusters):
        centers[c] += data[i][:-1]
        nums[c] += 1
        
    normalized_mutual_information(data, clusters, k)
    
    for i in range(k):
        centers[i] /= nums[i]

    return centers

def assign_new_clusters(data, centers, k):
    n = data.shape[0]
    m = data.shape[1]
    
    new_clusters = []
    
    for i in range(n):
        dists = []
        for j in range(k):
            dist = distance(data[i], centers[j], m-1)
            dists.append((dist, j))
        #print dists
        dists = sorted(dists, key=lambda x: x[0])
        new_clusters.append(dists[0][1])
        
    return new_clusters

def plot(data, clusters, k):
    x = data[:, 0]
    y = data[:, 1]
    colors = []
    for i in range(len(clusters)):
        if clusters[i] == 0:
            colors.append('r')
        elif clusters[i] == 1:
            colors.append('g')
        elif clusters[i] == 2:
            colors.append('b')
    
    plt.scatter(x,y, c=colors)
    plt.show()

if __name__ == '__main__':
    
    data = np.loadtxt(r"data\hw3\dataset1.txt")
    k = 3
    max_iter = 100
    
    clusters = random_initalize_clusters(data, k)
    plot(data, clusters, k)
    
    for iter in range(max_iter):
        print iter
        
        centers = find_centers(data, clusters, k)
    
        new_clusters = assign_new_clusters(data, centers, k)
        print new_clusters
        plot(data, new_clusters, k)
        
           
        if new_clusters == clusters:
            break
        else:
            clusters = new_clusters

    purity = purity(data, new_clusters, k)
    print "Purity: " + str(purity)
     
    nmi = normalized_mutual_information(data, clusters, k)
    print "Normalized Mutual Information: " + str(nmi)

    plot(data, new_clusters, k)
    
    
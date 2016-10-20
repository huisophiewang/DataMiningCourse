import math
import random
from pprint import pprint

# installed packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


###############################################################################################
# randomly choose k data points as cluster center
# assign each data point to its closed cluster center
# note: randomly assign data to a cluster with equal prob doesn't work, lead to identical clusters
#       randomly assign data to a cluster with prob 1 works, but always converge to a local optimal
def initialize(data, k):
    
    n = data.shape[0]
    p = data.shape[1]
    
    weights = np.zeros([data.shape[0], k])
    
    center_idx = random.sample(range(n), k) 
    #print center_idx
    
    for i in range(n):
        dists = np.zeros(k)
        for j in range(k):
            dists[j] = distance(data[i], data[center_idx][j], p-1)
        closest = np.argsort(dists)[0]
        weights[i][closest] = 1.0
   
    return weights
        
# Euclidean distance of x1 and x2, k dimensions
def distance(x1, x2, k):
    dist = 0.0
    for i in range(k):
        dist += math.pow((x1[i]-x2[i]), 2)
    dist = math.sqrt(dist)
    return dist        

###############################################################################
# find new Gaussian clusters
# estimate the parameters according to the maximization of weighted likelihood
def M_step(data, k, weights):
    # data n x m
    n = data.shape[0]
    m = data.shape[1]
    
    cluster_weight = np.sum(weights, axis=0)
    #print cluster_weight

    # mean k x (m-1)
    mean = np.zeros([k, m-1])
    for i in range(n):
        for j in range(k):
            mean[j] += weights[i][j] * data[i][:-1]
    
    for j in range(k):
        mean[j] /= cluster_weight[j]
    print mean
    
    # covariance matrix (m-1) x (m-1) for each k
    covari = []
    for j in range(k):
        covari.append(np.zeros([m-1, m-1]))
         
    for i in range(n):
        for j in range(k):
            for a in range(m-1):
                for b in range(m-1):
                    if a == b:
                        covari[j][a][b] += weights[i][j] * math.pow((data[i][a] - mean[j][a]), 2)
                    else:
                        covari[j][a][b] += weights[i][j] * (data[i][a] - mean[j][a]) * (data[i][b] - mean[j][b])

    for j in range(k): 
        covari[j] /= cluster_weight[j]     
        #print covari[j]

    
    # normalize cluster weight
    cluster_weight /= sum(cluster_weight)
    print cluster_weight
    
    return mean, covari, cluster_weight

#########################################################################################
# calculate expected likelihood given current estimate of parameters
# assign objects to new clusters
# weights are actually posterior prob, (given a point, the prob it belongs to a cluster)
def E_step(data, k, mean, covari, cluster_weight):
    n = data.shape[0]
    new_weights = np.zeros([n, k])
    likelihood = 0.0
    
    for i in range(n):
        for j in range(k):
            p = multivariate_normal.pdf(data[i][:-1], mean[j], covari[j])
            new_weights[i][j] = cluster_weight[j] * p
            
        likelihood += math.log(np.sum(new_weights[i]))
        new_weights[i] /= np.sum(new_weights[i])
        #print new_weights[i]
    return new_weights, likelihood
        
def plot(data, k, weights):
    n = data.shape[0]

    x = data[:, 0]
    y = data[:, 1]
    colors = []
    for i in range(n):

        index = sorted(range(k), key=lambda x: weights[i][x], reverse=True)[0]

        if index == 0:
            colors.append('r')
        elif index == 1:
            colors.append('g')
        elif index == 2:
            colors.append('b')
    
    plt.scatter(x,y, c=colors)
    plt.show()

if __name__ == '__main__':
    data = np.loadtxt(r"data\hw4\dataset2.txt")
    k = 3
    max_iter = 1000
    likelihood = float('-inf')
    
    weights = initialize(data, k)

    for iter in range(max_iter):
        print "Iteration: " + str(iter)

        mean, covari, cluster_weight = M_step(data, k, weights)
        new_weights, new_likelihood = E_step(data, k, mean, covari, cluster_weight)
        
        print "log likelihood: " + str(new_likelihood)
        
        # stop criterion, log likelihood doesn't change
        if new_likelihood > likelihood:  
            weights = new_weights
            likelihood = new_likelihood
        else:
            break
        
    
    for j in range(k):
        print 'cluster ' + str(j)
        print 'mean: ' + str(mean[j])
        print 'covariance: ' + str(covari[j])
    
    plot(data, k, new_weights)
        
    
    
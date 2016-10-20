import random

# installed packages
import numpy as np
import matplotlib.pyplot as plt

# utilities functions
from utilities import *

def get_neighbors(data, p, eps):
    n = data.shape[0]
    m = data.shape[1]
    neighbors = set()
    for idx in range(n):
        dist = distance(data[p], data[idx], m-1)
        if idx == p:
            continue
        if dist <= eps:
            neighbors.add(idx)
    
    return neighbors

def dbscan(data, eps, min_pts):

    n = data.shape[0]
    m = data.shape[1]
    
    unvisited = set(range(n))
    clusters = [None] * n
    cluster_id = 0

    while len(unvisited) != 0:
        # randomly choose unvisited point as p
        p = random.choice(tuple(unvisited))
        
        # mark as visited
        unvisited.remove(p)
        
        # neighbors of p
        neighbors = get_neighbors(data, p, eps)
     
        # p is core point, otherwise p is noise
        if len(neighbors) >= min_pts:
            # new cluster
            clusters[p] = cluster_id
            
            # stop until all the density-reachable points of p are visited
            while len(neighbors.intersection(unvisited)) != 0:
                # visit each neighbor of p
                for nb in neighbors:
                    
                    # skip visited ones
                    if not nb in unvisited:
                        continue
                    
                    # mark as visited
                    unvisited.remove(nb)
                    
                    # add to the same cluster as p, if not already in a cluster
                    if clusters[nb] == None:
                        clusters[nb] = cluster_id
                    
                    nb_neighbors = get_neighbors(data, nb, eps)
                    # this neighbor is a core point
                    if len(nb_neighbors) >= min_pts:
                        # core point's neighbors could be core points, which lead to more neighbors, update
                        neighbors = neighbors.union(nb_neighbors).copy()
            
            # done for this cluster
            cluster_id += 1
    
    return clusters
    
def count_clusters(clusters):
    count = set()
    for c in clusters:
        if c == None:
            continue
        if not c in count:
            count.add(c)
    return len(count)
        
def plot(data, clusters):
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
        elif clusters[i] == 3:
            colors.append('y')
        else:
            colors.append('w')
    
    plt.scatter(x,y, c=colors)
    plt.show()
    
if __name__ == '__main__':
    
    # data1, 0.7, 5
    # data2, 0.926, 3
    # data3, 0.3, 3
    data = np.loadtxt(r"data\hw3\dataset3.txt")
    eps = 0.3
    min_pts = 3
    
    
    clusters = dbscan(data, eps, min_pts)
    print 'Clusters: ' + str(clusters)
    
    k = count_clusters(clusters)
    
    purity = purity(data, clusters, k)
    print "Purity: " + str(purity)
    
    nmi = normalized_mutual_information(data, clusters, k)
    print "Normalized Mutual Information: " + str(nmi)
    
    plot(data, clusters)
    

        
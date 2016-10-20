import math
import numpy as np

# Euclidean distance of x1 and x2, k dimensions
def distance(x1, x2, k):
    dist = 0.0
    for i in range(k):
        dist += math.pow((x1[i]-x2[i]), 2)
    dist = math.sqrt(dist)
    return dist

def entropy(input):
    entropy = 0.0
    total = np.sum(input)
    for i in input:
        if i == 0:
            continue
        p = i/float(total)
        entropy += (-p)*math.log(p)
    return entropy

def purity(data, clusters, k):
    n = data.shape[0]
    m = data.shape[1]
    
    # get number of classes
    data_classes = set()
    for i in range(n):
        if not data[i][m-1] in data_classes:
            data_classes.add(data[i][m-1])
    num_c = len(data_classes)
    
    # put instance index to each class
    classes = []
    for i in range(num_c):
        classes.append([])
    for i in range(n):
        c = int(data[i][m-1])-1
        classes[c].append(i)
    
    # get max cluster in each class
    total_correct = 0
    for i, cls in enumerate(classes):
        counts = [0] * k
        for idx in cls:
            if clusters[idx] == None:
                continue
            counts[clusters[idx]] += 1
            
        total_correct += max(counts)
    
    purity = total_correct / float(n)
    return purity

def normalized_mutual_information(data, clusters, k):
    n = data.shape[0]
    m = data.shape[1]
    
    # get number of classes
    data_classes = set()
    for i in range(n):
        if not data[i][m-1] in data_classes:
            data_classes.add(data[i][m-1])
    num_c = len(data_classes)
        
    # put instance index to each class
    classes = []
    for i in range(num_c):
        classes.append([])
    for i in range(n):
        c = int(data[i][m-1])-1
        classes[c].append(i)
    
    # get class x cluster table
    table = np.zeros([num_c, k])
    for i, cls in enumerate(classes):
        counts = [0] * k
        for idx in cls:
            if clusters[idx] == None:
                continue
            counts[clusters[idx]] += 1
        for j in range(k):
            table[i][j] = counts[j]
    
    print table
    
    table_rows = np.sum(table, axis = 1)
    print table_rows
    table_cols = np.sum(table, axis = 0)
    print table_cols
    
    # mutual information
    mutual_info = 0.0
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            pij = table[i][j] / float(n)
            if pij == 0:
                continue
            pi = table_rows[i] / float(n)
            pj = table_cols[j] / float(n)
            mutual_info += pij*math.log(pij/(pi*pj))
    
    # entropy
    class_entropy = entropy(table_rows)
    cluster_entropy = entropy(table_cols)
    
    # normalized
    normalized = mutual_info / math.sqrt(class_entropy * cluster_entropy)

    return normalized
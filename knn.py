import math
from pprint import pprint
import numpy as np


# Euclidean distance of x1 and x2, k dimenstion 
def distance(x1, x2, k):
    dist = 0.0
    for i in range(k):
        dist += math.pow((x1[i]-x2[i]), 2)
    dist = math.sqrt(dist)
    return dist

def knn(train_data, test_data, k):
    # num of training instances
    n = train_data.shape[0]
    # num of featrues + 1
    p = train_data.shape[1]
    # num of test instances
    m = test_data.shape[0]
     
    # predict test data
    predict = np.empty((m,1))
    # distance from test instance to all training instances
    dist_mat = np.zeros((m,n))
     
    for i, x1 in enumerate(test_data):
        for j, x2 in enumerate(train_data):
            dist_mat[i][j] = distance(x1, x2, p-1)
    
    correct_num = 0
    for i in range(m):
        pos_num = 0
        neg_num = 0
        indices = np.argsort(dist_mat[i])
        
        # choose k neighbors
        for nb in indices[1:k+1]:
            if train_data[nb][p-1] == 1.0:
                pos_num += 1
            else:
                neg_num += 1
      
        if pos_num > neg_num:
            predict[i][0] = 1.0
        else:
            predict[i][0] = 0.0      
              
        if predict[i][0] == test_data[i][p-1]:
            correct_num += 1
      
    acc = float(correct_num)/float(m)  
    return acc
    
     


if __name__ == '__main__':
    data = np.loadtxt(r"data\knn\data.txt")

    k_accs = []
    fold = 5
    for k in range(1, 50, 2):
        #print k
        avg_acc = 0.0
        for j in range(fold):

            train_data = []
            test_data = []
            
            for i, x in enumerate(data):
                if i%fold == j:
                    test_data.append(data[i].tolist())
                else:
                    train_data.append(data[i].tolist())
                 
            acc = knn(np.array(train_data), np.array(test_data), k)
            avg_acc += acc
        avg_acc /= float(fold)
        #print avg_acc
        k_accs.append((k, avg_acc))
        
    k_accs = sorted(k_accs, key=lambda item:item[1], reverse=True)
    #pprint(k_accs)
    best = k_accs[0]
    print "The optimal value of k is: " + str(best[0])
    print "The highest accuracy is: " + str(best[1])
            

            

    


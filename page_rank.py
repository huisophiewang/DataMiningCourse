import numpy as np

def get_rank(m, n, beta):

    #r = np.full((n, 1), 1.0/n)
    r = np.zeros((n, 1))
    r[0] = 1
    
    print r
    m = beta*m + (1-beta)*np.full((n, n), 1.0/n)
    print m
    iter = 100

    for i in range(iter):
        print i
        
        
        r_new = np.dot(m, r)
        print r_new
        if np.array_equal(r_new, r):
            break
        else:
            r = r_new

def get_rank_rearrange(m, n, beta):

    r0 = np.full((n, 1), 1.0/n)
    
#     r0 = np.zeros((n, 1))
#     r0[0] = 1
    
    r = r0
    m = beta*m
    print m
    iter = 100

    for i in range(iter):
        print i
        
        
        r_new = np.dot(m, r) + (1-beta)*r0
        
        print r_new

        r = r_new     

            
if __name__ == '__main__':
    m = np.array([[0, 0, 0, 0.5, 0, 0],
              [0, 0, 0.25, 0, 0, 1.0],
              [0.5, 1, 0, 0, 0, 0],
              [0.5, 0, 0.25, 0.5, 0, 0],
              [0, 0, 0.25, 0, 0, 0],
              [0, 0, 0.25, 0, 1, 0]])
    
#     m = np.array([[0.5, 0.5, 0],
#                  [0.5, 0, 0.0],
#                  [0, 0.5, 1.0]])
    print m
    get_rank_rearrange(m, 6, 0.8)
    
    
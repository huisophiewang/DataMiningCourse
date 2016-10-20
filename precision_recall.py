truth = [3,3,1,1,1,4,3,3,4,2,4,2,1,2,3,2,1,2,4,4]
output = [2,2,3,3,3,4,2,2,3,1,4,1,3,1,2,1,2,1,4,1]

pair = []
n = 20
tp = 0
tn = 0
fp = 0
fn = 0
for i in range(n):
    for j in range(i+1, n):
        #print i, j, truth[i], truth[j]
        
        if output[i] == output[j]:
            same_cluster = True
        else:
            same_cluster = False

        if truth[i] == truth[j]:
            same_class = True
        else:
            same_class = False
        
        if same_class and same_cluster:
            tp += 1
        if not same_class and same_cluster:
            fp += 1
        if same_class and not same_cluster:
            fn += 1
        if not same_class and not same_cluster:
            tn += 1 
        


print tp, fp, fn, tn

print 2*0.6744*0.725/(0.6744+0.725)
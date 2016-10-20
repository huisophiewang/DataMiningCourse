import math

# p1 = float(4.0/14)
# p2 = float(6.0/14)
# p3 = float(4.0/14)
# 
# print p1
# 
# entropy = (-p1)*math.log(p1,2)+(-p2)*math.log(p2,2)+(-p3)*math.log(p3,2)
# 
# print entropy

def entropy(input):
    result = 0
    sum = 0
    for i in input:
        sum += i

    for i in input:
        p = i/float(sum)
        result += (-p)*math.log(p)
    

    return result

def result(item):
    n = 150.0
    p12 = item[0]/n
    p1 = item[1]/n
    p2 = item[2]/n
    result = p12* math.log(p12/(p1*p2))
    #print result
    return result


if __name__ == '__main__':  
    data = [(50, 50, 50)]
    sum = 0
    for line in data:
        sum += result(line)
    sum = sum*3
    
    print sum
    print entropy([5,5,5])
    
    print sum / math.sqrt(entropy([5,5,5])*entropy([5,5,5]))
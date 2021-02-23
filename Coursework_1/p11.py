import numpy as np
import time as time
import matplotlib.pyplot as plt
import math

def newSort(X,k=0):

    n = len(X)
    if n==1:
        return X
    elif n<=k:
        for i in range(n-1):
            ind_min = i
            for j in range(i+1,n):
                if X[j]<X[ind_min]:
                    ind_min = j
            X[i],X[ind_min] = X[ind_min],X[i]
        return X
    else:
        L = newSort(X[:n//2],k)
        R = newSort(X[n//2:],k)
        return merge(L,R)

def merge(L,R):

    M = [] #Merged list, initially empty
    indL,indR = 0,0 #start indices
    nL,nR = len(L),len(R)

    #Add one element to M per iteration until an entire sublist
    #has been added
    for i in range(nL+nR):
        if L[indL]<R[indR]:
            M.append(L[indL])
            indL = indL + 1
            if indL>=nL:
                M.extend(R[indR:])
                break
        else:
            M.append(R[indR])
            indR = indR + 1
            if indR>=nR:
                M.extend(L[indL:])
                break
    return M

def time_newSort():
 #Here I iterate through 10 evenly spaced N values with k values some of which are constant and others are
 #dependent upon n. The time function is used here in order to work out the computational time. 
 times=np.zeros(10)
 for i in range(1,11):
    start_time=time.time()
    newSort(np.random.randint(10,size=i*100),k=0)
    end_time=time.time()
    times[i-1]=end_time-start_time
 plt.plot(range(1,11), times,label='k=0')
 times=np.zeros(10)
 for i in range(1,11):
    start_time=time.time()
    newSort(np.random.randint(10,size=i*100),k=i*100)
    end_time=time.time()
    times[i-1]=end_time-start_time
 plt.plot(range(1,11), times,label='k=n')
 times=np.zeros(10)
 for i in range(1,11):
    start_time=time.time()
    newSort(np.random.randint(10,size=i*100),k=i*50)
    end_time=time.time()
    times[i-1]=end_time-start_time
 plt.plot(range(1,11), times,label='k=n/2')
 times=np.zeros(10)
 for i in range(1,11):
    start_time=time.time()
    newSort(np.random.randint(10,size=i*100),k=i*25)
    end_time=time.time()
    times[i-1]=end_time-start_time
 plt.plot(range(1,11), times,label='k=n/4')
 times=np.zeros(10)
 for i in range(1,11):
    start_time=time.time()
    newSort(np.random.randint(10,size=i*100),k=100)
    end_time=time.time()
    times[i-1]=end_time-start_time
 plt.plot(range(1,11), times,label='k=100')
 for i in range(1,11):
    start_time=time.time()
    newSort(np.random.randint(10,size=i*100),k=500)
    end_time=time.time()
    times[i-1]=end_time-start_time
 plt.plot(range(1,11), times,label='k=500')
 plt.legend()
 plt.xlabel("N 00's of iterations")
 plt.ylabel("Time(ms)")
 plt.title("Number of iterations vs Time taken to execute")
 plt.savefig("Figure_1.png")
 
#Here, you just need to input L and L0 as the same list. L0 is inputted so that the original list is preserved
#for when the index needs to be returned.
def findTrough(L,L0):
    n=len(L)
    #If the list is empty, there exists no minimal element and therefore there exists no peak.
    if n==0:
        return -(len(L)+1)
    #When the list has been whittled down to one element, this now gives us a trough.
    if n==1:
        return L0.index(L[n-1])
    #Here, recursion is used: if the middle element of the list is less than its right neighbour
    #then we know a trough exists in the first half of the list. Otherwise, the trough exists in the 
    #second half and we repeat this until one element is outputted. This gives us an algorithm of O(log(n))       
    if L[math.ceil(n/2)-1]>=L[math.ceil(n/2)]:
        return findTrough(L[math.ceil(n/2):],L0)
    if L[math.ceil(n/2)-1]<L[math.ceil(n/2)]:
        return findTrough(L[:math.ceil(n/2)],L0)
    
if __name__=='__main__':
    inputs=None
    outputs=time_newSort()
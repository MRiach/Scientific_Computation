"""Scientific Computation Project 3, part 1
Your CID here: 01349928
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
data1=np.load("data1.npy")
data2=np.load("data2.npy")
data3=np.load("data3.npy")
r=np.load("r.npy")
theta=np.load("theta.npy")

def hfield(r,th,h,levels=50):
    """Displays height field stored in 2D array, h,
    using polar grid data stored in 1D arrays r and th.
    Modify as needed.
    """
    thg,rg = np.meshgrid(th,r)
    xg = rg*np.cos(thg)
    yg = rg*np.sin(thg)
    plt.figure()
    plt.contourf(xg,yg,h,levels)
    plt.axis('equal')
    plt.title('Visual representation of Wave Heights')
    plt.xlabel('Radial distance')
    plt.ylabel('Radial distance')
    return None

def repair1(R,p,l=1.0,niter=10):
    """
    Question 1.1: Repair corrupted data stored in input
    array, R.
    Input:
        R: 2-D data array (should be loaded from data1.npy)
        p: dimension parameter
        l: l2-regularization parameter
        niter: maximum number of iterations during optimization
        inputs: can be used to provide other input as needed
    Output:
        A,B: a x p and p x b numpy arrays set during optimization
    """
    #problem setup
    R0 = R.copy() #Copying this is irrelevant to the task and adds extra layers of computation
    a,b = R.shape
    iK,jK = np.where(R0 != -1000) #indices for valid data
    aK,bK = np.where(R0 == -1000) #indices for missing data (seems unnecessary as this is never used )

    S = set() #S is never used in the rest of the code
    for i,j in zip(iK,jK):
            S.add((i,j))

    #Set initial A,B
    A = np.ones((a,p))  
    B = np.ones((p,b))

    #Create lists of indices used during optimization
    mlist = [[] for i in range(a)]
    nlist = [[] for j in range(b)]

    for i,j in zip(iK,jK):
        mlist[i].append(j) 
        nlist[j].append(i)

    dA = np.zeros(niter)
    dB = np.zeros(niter)

    for z in range(niter):
        Aold = A.copy()
        Bold = B.copy()

        #Loop through elements of A and B in different
        #order each optimization step
        for m in np.random.permutation(a): #ideally all random permutations would be vectorised, but this doesn't appear to be 
            for n in np.random.permutation(b): #vectorisable
                if n < p: #Update A[m,n]
                    Bfac = 0.0
                    Asum = 0

                    for j in mlist[m]:  #this should be vectorised to speed up the code 
                        Bfac += B[n,j]**2 #inefficient
                        Rsum = 0
                        for k in range(p):
                            if k != n: Rsum += A[m,k]*B[k,j] #this should be a dot product to be faster
                        Asum += (R[m,j] - Rsum)*B[n,j] #inefficient

                    A[m,n] = Asum/(Bfac+l) #New A[m,n]
                if m < p: 
                    Afac = 0.0
                    Bsum = 0
                    
                    for i in nlist[n]:  #this should be vectorised to speed up the code 
                        Afac += A[i,m]**2 #inefficient
                        Rsum = 0
                        for k in range(p):
                            if k != m: Rsum += A[i,k]*B[k,n]  #this should be a dot product to be faster
                        Bsum += (R[i,n] - Rsum)*A[i,m] #inefficient
                    
                    B[m,n]=Bsum/(Afac+l) #New B[m,n]
        dA[z] = np.sum(np.abs(A-Aold))
        dB[z] = np.sum(np.abs(B-Bold))
        if z%10==0: print("z,dA,dB=",z,dA[z],dB[z])  #stopping condition could be added for sufficiently small dA and dB


    return A,B

def repair2(R,p,l=1.0,niter=10):
    """
    Question 1.1: Repair corrupted data stored in input
    array, R. Efficient and complete version of repair1.
    Input:
        R: 2-D data array (should be loaded from data1.npy)
        p: dimension parameter
        l: l2-regularization parameter
        niter: maximum number of iterations during optimization
        inputs: can be used to provide other input as needed
    Output:
        A,B: a x p and p x b numpy arrays set during optimization
    """
    a,b = R.shape
    iK,jK = np.where(R != -1000) #indices for valid data

    #Set initial A,B
    A = np.ones((a,p))
    B = np.ones((p,b))

    #Create lists of indices used during optimization
    mlist = [[] for i in range(a)]
    nlist = [[] for j in range(b)]

    for i,j in zip(iK,jK):
        mlist[i].append(j) 
        nlist[j].append(i)

    dA = np.zeros(niter)
    dB = np.zeros(niter)
    
    arandompermutations = [[] for i in range(niter)] #precompute random permutations to expedite running of code 
    brandompermutations = [[] for j in range(niter)] #Not perfectly random, but cyclical with a varied component and faster
    
    for w in range(niter):
        if w%3==0:
            arandompermutations[w] = np.random.RandomState(seed=0).permutation(a)
            brandompermutations[w] = np.random.RandomState(seed=0).permutation(b)
        elif w%3==1:
            arandompermutations[w] = np.random.RandomState(seed=1).permutation(a)
            brandompermutations[w] = np.random.RandomState(seed=1).permutation(b)
        else:
            arandompermutations[w] = np.random.RandomState(seed=2).permutation(a)
            brandompermutations[w] = np.random.RandomState(seed=2).permutation(b)        

    for z in range(niter):
        Aold = A.copy()
        Bold = B.copy()
        #Loop through elements of A and B in different
        #order each optimization step
        for m in arandompermutations[z]:  #compute random permutations before and then call from them
            for n in brandompermutations[z]: 
                if n < p: #Update A[m,n]
                    Bfac = 0.0
                    Asum = 0
                    Bfac = np.sum(B[n,mlist[m]]**2)
                    Rsum = 0
                    Rsum = np.dot(A[m,:],B[:,mlist[m]])-A[m,n]*B[n,mlist[m]]
                    Asum = np.sum((R[m,mlist[m]] - Rsum)*B[n,mlist[m]]) #all computations are now vectorised and faster
                    A[m,n] = Asum/(Bfac+l) #New A[m,n]
                if m < p: 
                    Afac = 0.0
                    Bsum = 0
                    Afac = np.sum(A[nlist[n],m]**2)
                    Rsum = 0
                    Rsum = np.dot(A[nlist[n],:],B[:,n])-A[nlist[n],m]*B[m,n]
                    Bsum = np.sum((R[nlist[n],n] - Rsum)*A[nlist[n],m]) #all computations are now vectorised and faster                
                    B[m,n]=Bsum/(Afac+l) #New B[m,n]
        
        dA[z] = np.sum(np.abs(A-Aold))
        dB[z] = np.sum(np.abs(B-Bold))
        if dA[z] < 1e-05 and dB[z] < 1e-05:   #terminate if sufficiently small
            return A,B
    return A,B

def outwave(r0):
    """
    Question 1.2i)
    Calculate outgoing wave solution at r=r0
    See code/comments below for futher details
        Input: r0, location at which to compute solution
        Output: B, wave equation solution at r=r0
        
        Comment: RHS - Laplace's equation in polar co-ordinates, LHS second derivative wrt time, wave equation- sep of vars

    """
    from scipy.special import hankel1
    x = range(10)
    Hankel1 = lambda t: hankel1(t,1*8)
    Hankelr0 = lambda t: hankel1(t,r0*8) #omega value is 8 (see write up)
    vfunc1 = np.vectorize(Hankel1)
    vfuncr0 = np.vectorize(Hankelr0)
    Hankel1sum=np.sum(vfunc1(x))
    Hankelr0sum=np.sum(vfuncr0(x)) #vectorise the sum of the first 10 solutions to Bessel's equation 

    f=data2[0, :, :] # extract solution at r=1 from the dataset
    f=f/Hankel1sum.real #divide by solution at r=1 (which can be done due to separation of variables)
    B=f*Hankelr0sum.real #multiply by solution to Bessel's equation at r=r0 (which can be done due to separation of variables)
   
    return B

def analyze1():
    """
    Question 1.2ii)
    Add input/output as needed

    """
    X1 = data3[:,36,:] #pi/4
    X2 = data3[:,108,:] #3pi/4
    X3 = data3[:,180,:] #5pi/4
    plt.figure(0)
    plt.plot(X1[0,:],label='theta = pi/4') #time series at r=1
    plt.plot(X2[0,:],label='theta = 3pi/4') #time series at r=1
    plt.plot(X3[0,:],label='theta = 5pi/4') #time series at r=1
    plt.title("Time series of waves' heights at r=1")
    plt.xlabel("Time")
    plt.ylabel("Height")
    plt.legend()
    plt.show()
    plt.figure(1)
    plt.plot(X1[299,:],label='theta = pi/4') #time series at r=5
    plt.plot(X2[299,:],label='theta = 3pi/4') #time series at r=5
    plt.plot(X3[299,:],label='theta = 5pi/4') #time series at r=5
    plt.title("Time series of waves' heights at r=5")
    plt.xlabel("Time")
    plt.ylabel("Height")
    plt.legend()
    plt.show()
    plt.figure(2)
    plt.ylim((-1e-15,1e-15))
    plt.plot(np.linspace(1,5,300),np.mean(X1[:,:],axis=1),label='theta = pi/4')
    plt.plot(np.linspace(1,5,300),np.mean(X2[:,:],axis=1) ,label='theta = 3pi/4')
    plt.plot(np.linspace(1,5,300),np.mean(X3[:,:],axis=1) ,label='theta = 5pi/4') 
    plt.title("Mean of Heights against Radius")
    plt.xlabel("Radius")
    plt.ylabel("Avergage Height")
    plt.legend()
    plt.show()
    plt.figure(3)
    plt.plot(np.linspace(1,5,300),np.var(X1[:,:],axis=1),label='theta = pi/4')
    plt.plot(np.linspace(1,5,300),np.var(X2[:,:],axis=1) ,label='theta = 3pi/4')
    plt.plot(np.linspace(1,5,300),np.var(X3[:,:],axis=1) ,label='theta = 5pi/4') 
    plt.title("Variance of Heights against Radius")
    plt.xlabel("Radius")
    plt.ylabel("Variance")
    plt.legend()
    plt.show()
    plt.figure(4)
    w1,Pxx1 = welch(X1[0,:],nperseg=119) #welch's method is used due to aperiodicity
    w2,Pxx2 = welch(X2[0,:],nperseg=119) #Time period is 1, so can plot the direct result without scaling w
    w3,Pxx3 = welch(X3[0,:],nperseg=119)
    plt.plot(w1,Pxx1,label='theta = pi/4')
    plt.plot(w2,Pxx2,label='theta = 3pi/4')
    plt.plot(w3,Pxx3,label='theta = 5pi/4')
    plt.title("Autospectral Densities at different theta values at r=1")
    plt.xlabel("Frequency")
    plt.ylabel("Power Spectral Density")
    plt.legend()
    plt.show()
    plt.figure(5)
    w1,Pxx1 = welch(X1[299,:],nperseg=119)
    w2,Pxx2 = welch(X2[299,:],nperseg=119)
    w3,Pxx3 = welch(X3[299,:],nperseg=119)
    plt.plot(w1,Pxx1,label='theta = pi/4')
    plt.plot(w2,Pxx2,label='theta = 3pi/4')
    plt.plot(w3,Pxx3,label='theta = 5pi/4')
    plt.title("Autospectral Densities at different theta values at r=5")
    plt.xlabel("Frequency")
    plt.ylabel("Power Spectral Density")
    plt.legend()
    plt.show()
    return None 

def reduce(H = data3):
    """
    Question 1.3: Construct one or more arrays from H
    that can be used by reconstruct
    Input:
        H: 3-D data array
        inputs: can be used to provide other input as needed
    Output:
        arrays: a tuple containing the arrays produced from H
    """

    arrays0 = [data3[:,:,i] for i in range(119)] #slice across time dimension
    arrays = [[[],[]] for i in range(119)]
    for i in range(119):
     U,S,VT = np.linalg.svd(arrays0[i])
     F = U.T
     G = np.dot(F,arrays0[i]) #compute G matrix
     arrays[i][0] = U[:,range(40)] #store reduced data
     arrays[i][1] = G[range(40),:] #store reduced data

    return arrays

def reconstruct(arrays):
    """
    Question 1.3: Generate matrix with same shape as H (see reduce above)
    that has some meaningful correspondence to H
    Input:
        arrays: tuple generated by reduce
        inputs: can be used to provide other input as needed
    Output:
        Hnew: a numpy array with the same shape as H
    """

    reconstructedarrays= [[] for i in range(119)]
    
    for i in range(119):
        sum = 0
        for j in range(40):
            sum+=np.outer(arrays[i][0][:,j],arrays[i][1][j,:]) #compute outer products and sum to reconstruct
        reconstructedarrays[i]=sum

    Hnew = np.dstack(reconstructedarrays)  #this function makes a list of m*n arrays into m*n*p array 

    return Hnew


if __name__=='__main__':    
    U,S,VT = np.linalg.svd(data1)
    plt.plot(np.log(S),'bo',markersize = 3)
    plt.title("Log of Singular Values of data1")
    plt.xlabel("Nth singular value (in descending order)")
    plt.ylabel("Natural Logarithm of singular values")
    plt.show()
    plt.plot(data2[0,0,:])
    plt.title("Time series of heights at r=1 and theta = 0")
    plt.xlabel("Time increments (pi/80)")
    plt.ylabel("Height")
    analyze1()
    U,S,VT = np.linalg.svd(data3[:,:,0])
    plt.plot(np.cumsum(S),'bo',markersize = 3)
    plt.title("Cumulative sum of singular values of data3 at first time slot")
    plt.xlabel("Nth singular value")
    plt.ylabel("Cumulative sum of singular values")
    plt.axvline(x=39)
    plt.show()
    A,B=repair2(data1,5,l=0)
    C = np.matmul(A,B)
    hfield(r,theta,C,levels=50)

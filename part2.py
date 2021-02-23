"""Scientific Computation Project 3, part 2
Your CID here: 01349928
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.spatial.distance import pdist
data3=np.load("data3.npy")

def microbes(phi,kappa,mu,L = 1024,Nx=1024,Nt=1201,T=600,display=False):
    """
    Question 2.2
    Simulate microbe competition model

    Input:
    phi,kappa,mu: model parameters
    Nx: Number of grid points in x
    Nt: Number of time steps
    T: Timespan for simulation is [0,T]
    Display: Function creates contour plot of f when true

    Output:
    f,g: Nt x Nx arrays containing solution
    """

    #generate grid
    L = 1024
    x = np.linspace(0,L,Nx)
    dx = x[1]-x[0]
    dx2inv = 1/dx**2

    def RHS(y,t,k,r,phi,dx2inv):
        #RHS of model equations used by odeint

        n = y.size//2

        f = y[:n]
        g = y[n:]

        #Compute 2nd derivatives
        d2f = (f[2:]-2*f[1:-1]+f[:-2])*dx2inv
        d2g = (g[2:]-2*g[1:-1]+g[:-2])*dx2inv

        #Construct RHS
        R = f/(f+phi)
        dfdt = d2f + f[1:-1]*(1-f[1:-1])- R[1:-1]*g[1:-1]
        dgdt = d2g - r*k*g[1:-1] + k*R[1:-1]*g[1:-1]
        dy = np.zeros(2*n)
        dy[1:n-1] = dfdt
        dy[n+1:-1] = dgdt

        #Enforce boundary conditions
        a1,a2 = -4/3,-1/3
        dy[0] = a1*dy[1]+a2*dy[2]
        dy[n-1] = a1*dy[n-2]+a2*dy[n-3]
        dy[n] = a1*dy[n+1]+a2*dy[n+2]
        dy[-1] = a1*dy[-2]+a2*dy[-3]

        return dy


    #Steady states
    rho = mu/kappa
    F = rho*phi/(1-rho)
    G = (1-F)*(F+phi)
    y0 = np.zeros(2*Nx) #initialize signal
    y0[:Nx] = F
    y0[Nx:] = G + 0.01*np.cos(10*np.pi/L*x) + 0.01*np.cos(20*np.pi/L*x)

    t = np.linspace(0,T,Nt)

    #compute solution
    y = odeint(RHS,y0,t,args=(kappa,rho,phi,dx2inv),rtol=1e-6,atol=1e-6)
    f = y[:,:Nx]
    g = y[:,Nx:]
    if display:
        plt.figure()
        plt.contour(x,t,f)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Contours of f')
        plt.figure()
        plt.contour(x,t,g)
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Contours of g')



    return f,g


def newdiff(f,h):
    """
    Question 2.1 i)
    Input:
        f: array whose 2nd derivative will be computed
        h: grid spacing
    Output:
        d2f: second derivative of f computed with compact fd scheme
    """
    from scipy.linalg import solve_banded

    N = len(f)

    # Coefficients for compact fd scheme
    alpha = 9/38
    a = (696-1191*alpha)/428
    b = (2454*alpha-294)/535
    c = (1179*alpha-344)/2140

    d = np.array([145/12, -76/3, 29/2, -4/3, 1/12]) #initialise data which is used to construct RHS
    e = np.array([1/12, -4/3, 29/2, -76/3, 145/12])
    g = np.array([c/9, b/4, a, -2*c/9-2*b/4-2*a, a, b/4, c/9]) #group together f values

    # Construct banded matrix ab as in scipy.linalg.solve_banded documentation
    ab = np.zeros((3, N)) #initialise matrix
    ab[0, 0] = 0
    ab[0, 1] = 10
    ab[0, 2:] = alpha
    ab[1,:] = 1
    ab[2, :-2] = alpha
    ab[2, -2] = 10
    ab[2, -1] = 0

    # Construct RHS b
    b = np.zeros(N) #initialise right hand side
    b[0] = np.sum(d*f[:5])
    b[1] = np.sum(g[2:]*f[:5]) + np.sum(g[:2]*f[-3:-1])
    b[2] = np.sum(g[1:]*f[:6]) + g[0]*f[-2]
    b[3:-3] = g[0]*f[:-6] + g[1]*f[1:-5] + g[2]*f[2:-4] + g[3]*f[3:-3] + g[4]*f[4:-2] + g[5]*f[5:-1] + g[6]*f[6:] #vectorise
    b[-3] = np.sum(g[:-1]*f[-6:]) + g[-1]*f[1]
    b[-2] = np.sum(g[:-2]*f[-5:]) + np.sum(g[-2:]*f[1:3])
    b[-1] = np.sum(e*f[-5:])

    b = b/(h**2)  #divide everything by h^2 at the end so as to do this efficiently

    d2f = solve_banded((1, 1), ab, b) #l=u=1, as in solve_banded documentation
    return d2f

def analyzefd():
    """
    Question 2.1 ii)
    Add input/output as needed

    """
    alpha = 9/38
    a = (696-1191*alpha)/428
    b = (2454*alpha-294)/535
    c = (1179*alpha-344)/2140
    
    x = np.linspace(0, np.pi, 1000)
    y1 = x**2 
    y2 = 2-2*np.cos(x)
    y3 = (c/9)*(np.cos(3*x)-1)+(b/4)*(np.cos(2*x)-1)+a*(np.cos(x)-1)
    y3 = -2*y3
    y3 = y3/(2*alpha*np.cos(x)+1)
    plt.figure(1)
    plt.plot(x,y1,label='Exact') 
    plt.plot(x,y2,label='2nd order FD') 
    plt.plot(x,y3,label='Compact FD') 
    plt.title("Wavenumber analysis of two different schemes")
    plt.xlabel("kh")
    plt.ylabel("kh'")
    plt.legend()
    plt.show()
    
    
    return None 


def dynamics():
    """
    Question 2.2
    Add input/output as needed

    """
    #plots of behaviour at different values of kappa
    microbes(0.3,1.5,0.4*1.5,L = 1024,Nx=1024,Nt=1201,T=10000,display=True) 
    microbes(0.3,1.5,0.4*1.5,L = 1024,Nx=1024,Nt=2402,T=10000,display=True)
    microbes(0.3,1.7,0.4*1.7,L = 1024,Nx=1024,Nt=1201,T=10000,display=True)
    microbes(0.3,2,0.4*2,L = 1024,Nx=1024,Nt=1201,T=10000,display=True)
    
    #plots of correlation sum vs epsilon to determine fractal dimension
    
    f0,g0 = microbes(0.3,1.5,0.4*1.5,L = 1024,Nx=1024,Nt=1201,T=10000)
    f1,g1 = microbes(0.3,1.7,0.4*1.7,L = 1024,Nx=1024,Nt=1201,T=10000)
    f2,g2 = microbes(0.3,2,0.4*2,L = 1024,Nx=1024,Nt=1201,T=10000)
    
    y0=f0[600:,]   #discard influence of initial condition
    y1=f1[600:,]
    y2=f2[600:,]
    
    eps = np.logspace(1, 0.8, 100) #determine range of epsilon
    
    D0 = pdist(y0) #compute all n choose 2 distances
    D1 = pdist(y1)
    D2 = pdist(y2)
    C0,C1,C2 = np.zeros(100),np.zeros(100),np.zeros(100)
    
    for i in range(100):
        D0 = D0[D0<eps[i]]
        D1 = D1[D1<eps[i]]
        D2 = D2[D2<eps[i]]
        C0[i] = D0.size
        C1[i] = D1.size
        C2[i] = D2.size
        
    gradient0,intercept0 = np.polyfit(np.log(eps[0:20]),np.log(C0[0:20]),deg=1) #determine line of best fit
    
    plt.figure()
    plt.plot(eps,C0,label='C($\epsilon$)')
    plt.plot(eps,np.exp(intercept0)*np.power(eps,gradient0),'k--',label='least squares fit, slope = 11.60')
    plt.title('Correlation sum plot, k=1.5')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('$\epsilon$')
    plt.ylabel('C($\epsilon$))')
    plt.legend()
    plt.show()
    
    gradient1,intercept1 = np.polyfit(np.log(eps[0:20]),np.log(C1[0:20]),deg=1) #determine line of best fit
    
    plt.figure()
    plt.plot(eps,C1,label='C($\epsilon$)')
    plt.plot(eps,np.exp(intercept1)*np.power(eps,gradient1),'k--',label='least squares fit, slope = 10.31')
    plt.title('Correlation sum plot, k=1.7')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('$\epsilon$')
    plt.ylabel('C($\epsilon$))')
    plt.legend()
    plt.show()
    
    gradient2,intercept2 = np.polyfit(np.log(eps[0:20]),np.log(C2[0:20]),deg=1) #determine line of best fit
    
    plt.figure()
    plt.plot(eps,C2,label='C($\epsilon$)')
    plt.plot(eps,np.exp(intercept2)*np.power(eps,gradient2),'k--',label='least squares fit, slope = 7.79')
    plt.title('Correlation sum plot, k=2')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('$\epsilon$')
    plt.ylabel('C($\epsilon$))')
    plt.legend()
    plt.show()
    
    #Now to look at dynamics of data3 at theta = 3pi/4 and r=1 and r=5
    
    X = data3[:,108,:] #data at theta = 3pi/4    
    X1 = X[0,:] # r=1
    X2=X[299,:] # r=5
    y3 = [[0] for i in range(119)]
    y4 = [[0] for i in range(119)]
    y3 = np.array(y3)
    y4 = np.array(y4)
    for i in range(119):
        y3[i,0] = X1[i]
        y4[i,0] = X2[i]
    
    
    eps = np.logspace(1, 0, 100) #determine range of epsilon
    
    D3 = pdist(y3) #compute all n choose 2 distances
    D4 = pdist(y4)
    C3,C4= np.zeros(100),np.zeros(100)
    
    for i in range(100):
        D3 = D3[D3<eps[i]]
        D4 = D4[D4<eps[i]]
        C3[i] = D3.size
        C4[i] = D4.size
        
    gradient3,intercept3 = np.polyfit(np.log(eps[0:100]),np.log(C3[0:100]),deg=1) #determine line of best fit
    
    plt.figure()
    plt.plot(eps,C3,label='C($\epsilon$)')
    plt.plot(eps,np.exp(intercept3)*np.power(eps,gradient3),'k--',label='least squares fit, slope = 0.56')
    plt.title('Correlation sum plot, r=1, theta = 3pi/4')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('$\epsilon$')
    plt.ylabel('C($\epsilon$))')
    plt.legend()
    plt.show()
    
    gradient4,intercept4 = np.polyfit(np.log(eps[30:90]),np.log(C1[30:90]),deg=1) #determine line of best fit
    
    plt.figure()
    plt.plot(eps,C4,label='C($\epsilon$)')
    plt.plot(eps,np.exp(intercept4)*np.power(eps,gradient4),'k--',label='least squares fit, slope = 3.2 ')
    plt.title('Correlation sum plot, r=5, theta = 3pi/4')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel('$\epsilon$')
    plt.ylabel('C($\epsilon$))')
    plt.legend()
    plt.show()

if __name__=='__main__':
    analyzefd()
    dynamics()
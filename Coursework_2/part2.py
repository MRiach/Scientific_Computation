"""Scientific Computation Project 2, part 2
Your CID here: 01349928
"""
import numpy as np
import networkx as nx
from scipy.integrate import odeint
import matplotlib.pyplot as plt
Barabasi2000=nx.barabasi_albert_graph(2000, 4, seed=1)
Barabasi100=nx.barabasi_albert_graph(100, 5, seed=1)

def rwgraph(G,i0,M,Nt):
    """ Question 2.1
    Simulate M Nt-step random walks on input graph, G, with all
    walkers starting at node i0
    Input:
        G: An undirected, unweighted NetworkX graph
        i0: intial node for all walks
        M: Number of walks
        Nt: Number of steps per walk
    Output: X: M x Nt+1 array containing the simulated trajectories
    """
    X = np.zeros((M,Nt+1)) #matrix of paths
    X=X.astype(int) #convert to integers 
    Y = np.random.rand(M,Nt) #create matrix of numbers between zero and one (more efficient this way)
    V = [nbrdict for n, nbrdict in G.adjacency()]
    W= [x.keys() for x in V]
    Alist = [list(x) for x in W]  #extract Alist from networkx package and convert to correct data type
    Alist = np.array(Alist) #Construct Alist
    X[:,0] = i0 #First column is source node
    degrees = dict(Barabasi2000.degree)
    degrees = list(degrees.values())
    degrees = np.array(degrees) #store degrees so I don't have to repeatedly compute them
    for i in range(Nt):
     Alistindices=Alist[X[:,i]] #indices I am interested in are previous column of matrix to the column I am working with
     Y1=(np.multiply(degrees[X[:,i]],Y[:,i])) #Multiply degrees of each node by the random number generated
     Y1=Y1.tolist()
     Y1 = np.floor(Y1) #Calculate the floor of the degree of each node when multiplied by the random number generated
     Y2=Y1.tolist()
     Y2=np.array(Y2)
     Y2=Y2.astype(int)
     Y2=Y2.tolist() #Above lines involve converting to the appropriate data type
     newcolumn=np.zeros(M)
     for j in range(M): 
         newcolumn[j]=Alistindices[j][Y2[j]] #Go to the corresponding node depending on random number generated above
     X[:,i+1]=newcolumn #replace column of zeros with the next steps in the path for M simulations
   
    return X


def rwgraph_analyze1(G,M,Nt):
    """Analyze simulated random walks on
    Barabasi-Albert graphs.
    Modify input and output as needed.
    """
    index = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][0] #index of node with largest degree (with lowest value)
    X = rwgraph(G,index, M, Nt)    
    Y = sorted(Barabasi2000.degree, key=lambda x: x[1], reverse=True) #degree of each node
    plt.figure(0)
    plt.hist(X[:,Nt], bins=len(G),range=[0,len(G)], normed=True)
    plt.title("Histogram of destinations of M paths after Nt steps") 
    plt.xlabel("Node Number")
    plt.ylabel("Frequency of destination (density = 1)")
    plt.show()
    plt.figure(1)
    plt.vlines([a[0] for a in Y], ymin=0, ymax=np.divide([a[1] for a in Y],2*G.number_of_edges()))
    plt.xlabel("Node Number")
    plt.ylabel("Probability of ending up at node after infinite time")
    plt.title("Probability mass function of all nodes")
    plt.show()
    
    return None 


def rwgraph_analyze2(G,Nt,tfinal):
    """Analyze similarities and differences
    between simulated random walks and linear diffusion on
    Barabasi-Albert graphs.
    Modify input and output as needed.
    """
    times = np.linspace(0,tfinal,Nt) #values of t passed through differential equation
    N = len(G)
    index = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][0] #Find start node in the same way as in the random walk
    I = np.identity(N)
    A = nx.adjacency_matrix(G) #adjacency matrix
    Q = A.toarray().sum(axis=1) #matrix of degrees of nodes along diagonal
    f0 = np.zeros(N) #state of all nodes at time t = 0
    f0[index] = 1 #starting node has density = 1, and others zero
    Ls = np.subtract(I,np.dot(np.linalg.inv(np.diag(Q)),A.toarray())) # I-inv(Q)*A
    LsT = Ls.copy()
    LsT = LsT.T #Transpose of Ls
    Lseigenvalues,Lseigenvectors=np.linalg.eig(Ls) #eigenvalues extracted
    LsTeigenvalues,LsTeigenvectors=np.linalg.eig(LsT) #eigenvalues extracted
    def RHSLs(y,t):
     dydt = np.squeeze(np.array(-Ls.dot(y))) #differential equation set up
     return dydt
    def RHSLsT(y,t):
     dydt = np.squeeze(np.array(-LsT.dot(y)))  #differential equation set up
     return dydt 
    fLs=odeint(RHSLs,f0,times) #matrix of densities at my N nodes at Nt different times with final time being tfinal
    fLsT=odeint(RHSLsT,f0,times)#matrix of densities at my N nodes at Nt different times with final time being tfinal
    plt.figure(0) #plot of figures below
    plt.vlines(range(2000), ymin=0, ymax=fLs[Nt-1,:])
    plt.title("Densities of nodes against Nodes for scaled Laplacian")
    plt.xlabel("Node Number")
    plt.ylabel("Frequency of destination (density = 1)")
    plt.show()
    plt.figure(1)
    plt.vlines(range(2000), ymin=0, ymax=fLsT[Nt-1,:])
    plt.title("Densities of nodes against Nodes for transpose of scaled Laplacian")
    plt.xlabel("Node Number")
    plt.ylabel("Frequency of destination (density = 1)")
    plt.show()
    plt.figure(2)
    plt.scatter(range(2000), np.sort(Lseigenvalues))
    plt.title("Scaled Laplacian's eigenvalues in ascending order")
    plt.xlabel("Xth smallest eigenvalue")
    plt.ylabel("Value of eigenvalue")
    plt.show()
    plt.figure(3)
    plt.scatter(range(2000), np.sort(LsTeigenvalues))
    plt.title("Tranpose of scaled Laplacian's eigenvalues in ascending order")
    plt.xlabel("Xth smallest eigenvalue")
    plt.ylabel("Value of eigenvalue")
    plt.show()
    return None 



def modelA(G,x,i0,beta,gamma,tf,Nt):
    """
    Question 2.2
    Simulate model A

    Input:
    G: Networkx graph
    x: node which is initially infected with i_x=i0
    i0: magnitude of initial condition
    beta,gamma: model parameters
    tf,Nt: Solutions are computed at Nt time steps from t=0 to t=tf (see code below)

    Output:
    iarray: N x Nt+1 Array containing i across network nodes at
                each time step.
    """
    A = nx.adjacency_matrix(G)
    N = G.number_of_nodes()
    tarray = np.linspace(0,tf,Nt+1)
    initialstates = np.zeros(N)
    initialstates[x] = i0


    def RHS(y,t):
        """Compute RHS of modelA at time t
        input: y should be a size N array
        output: dy, also a size N array corresponding to dy/dt
        Discussion: I rely on the dot function in python in order to make use of vectorising
        """
        
        dy = -beta*y+gamma*(1-y)*(A.dot(y))  #use dot function to vectorise code and make running more efficient 
        return dy
    
    iarray=odeint(RHS,initialstates,tarray)
    
    return iarray

def modelB(G,i0,alpha,tf,Nt):
    N = G.number_of_nodes()
    L = nx.laplacian_matrix(G)
    tarray = np.linspace(0,tf,Nt+1)
    index = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][0]
    initialstates = np.zeros(2*N) #set this to be length 2N. First half accounts for s, second half for i
    initialstates[index + N] = i0
    dy = np.zeros(2*N)
    def RHS(y,t):
        s,i = np.array(y[0:N]),np.array(y[N:2*N]) 
        dy[:N] = i
        dy[N:2*N] = alpha*(L.dot(s))
        return dy #Everything is differentiated with respect to t so I amalgamate the two vectors into one
    modelBsolution = odeint(RHS,initialstates,tarray)
    return modelBsolution[:,N:2*N] #I am only interested in the values of the i's 


def transport(G,tf=100,Nt=100):
    """Analyze transport processes (model A, model B, linear diffusion)
    on Barabasi-Albert graphs.
    Modify input and output as needed.
    """
    index = sorted(G.degree, key=lambda x: x[1], reverse=True)[0][0] #node of highest degree with lowest node number
    nodes = range(100)
    times = range(101)
    MA1 = modelA(G,index,1,0.5,0.1,tf,Nt)
    MA2 = modelA(G,index,1,-0.5,-0.1,tf,Nt)
    MA3 = modelA(G,index,1,0.5,-0.1,tf,Nt)
    MA4 = modelA(G,index,1,-0.5,0.1,tf,Nt)
    MB1 = modelB(G,1,-0.01,tf,Nt)
    MB2 = modelB(G,1,0.01,tf,Nt)
    MB3 = modelB(G,1,-0.1,tf,Nt)
    MB4 = modelB(G,1,0.1,tf,Nt)
    MB5 = modelB(G,1,-10,tf,Nt)
    plt.figure(0)
    plt.ylim((-500,500))
    plt.plot(times,MA1.sum(1),label='beta=0.5,gamma=0.1') #this takes the sum of all nodes at time t
    plt.plot(times,MA2.sum(1),label='beta=-0.5,gamma=-0.1')
    plt.plot(times,MA3.sum(1),label='beta=0.5,gamma=-0.1')
    plt.plot(times,MA4.sum(1),label='beta=-0.5,gamma=0.1')
    plt.title("Sum of values of all nodes over time for Model A")
    plt.xlabel("Time")
    plt.ylabel("Sum of values at all nodes")
    plt.legend()
    plt.savefig("Figure_7.png")
    plt.show()
    plt.figure(1)
    plt.ylim((-5,5))
    plt.plot(times,MB1.sum(1),label='alpha=-0.01')
    plt.plot(times,MB2.sum(1),label='alpha=0.01')
    plt.plot(times,MB3.sum(1),label='alpha=-0.1')
    plt.plot(times,MB4.sum(1),label='alpha=0.1')
    plt.title("Sum of values of all nodes over time for Model B")
    plt.xlabel("Time")
    plt.ylabel("Sum of values at all nodes")
    plt.legend()
    plt.savefig("Figure_8.png")
    plt.show()
    plt.figure(2)
    plt.plot(nodes,MB1[100],label='alpha=-0.01') #this gives me the value of all my nodes at time, tf
    plt.title("Value of each node at tf = 100 ")
    plt.xlabel("Nodes")
    plt.ylabel("Value at node")
    plt.legend()
    plt.savefig("Figure_9.png")
    plt.show()
    plt.figure(3)
    plt.plot(nodes,MB5[100],label='alpha=-10')
    plt.title("Value of each node at tf = 100 ")
    plt.xlabel("Nodes")
    plt.ylabel("Value at node")
    plt.legend()
    plt.savefig("Figure_10.png")
    plt.show()

 
    return None #modify as needed


if __name__=='__main__':
    Barabasi100=nx.barabasi_albert_graph(100, 5, seed=1)
    transport(Barabasi100)
    

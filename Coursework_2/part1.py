"""Scientific Computation Project 2, part 1
Your CID here: 01349928
"""
from collections import deque



def flightLegs(Alist,start,dest):

    L1 = [0 for l in Alist] #Labels
    L2,L3 = [-1000 for l in L1], [0 for l in Alist] #Distances and Number of distinct paths
    Q=deque() #Make popping O(1) rather than O(n)
    Q.append(start) #Ensure that queue starts with the source node
    L1[start],L2[start],L3[start]=1,0,1 #Assign the source node's values for the above lists
    depth=-100 #Set depth to be negative so that it is only a sensible value once dest node has been reached
    while len(Q)>0: #Carry out my modified BFS until my queue is empty
        x = Q.popleft() #remove node from front of queue
        if L2[x]==depth: #stop iterating once I've reached the layer of my destination node
            Flights = [depth,L3[dest]] 
            return Flights
        for v in Alist[x]:
            if L1[v]==0: #Proceed if node is unvisited
                Q.append(v)  #Add unexplored neighbors to back of queue
                L1[v] = 1 #Mark the node as visited
                L2[v]=1+L2[x] #Distance from source node is one greater than distance of x from source node
                L3[v]=L3[v]+L3[x] # Add number of paths to x to the currently stored number of paths to v
                if v==dest: #if my destination is 'hit' then I record its depth
                    depth=L2[v] #record depth of destination
            else:
                if L2[v]==L2[x]+1: #check if there is another possible set of paths that haven't been accounted for
                    L3[v]=L3[v]+L3[x] #update the total number of paths to v

def safeJourney(Alist,start,dest):
    """
    Question 1.2 i)
    Find safest journey from station start to dest
    Input:
        Alist: List whose ith element contains list of 2-element tuples. The first element
        of the tuple is a station with a direct connection to station i, and the second element is
        the density for the connection.
    start, dest: starting and destination stations for journey.

    Output:
        Slist: Two element list containing safest journey and safety factor for safest journey
               Find the path with the lowest maximum density"""

    #Initialize dictionaries
    dinit=10**6
    Edict = {} #Explored nodes
    Udict = {new_list: (dinit,0) for new_list in range(len(Alist))} #Unexplored nodes and the node from which it is visited
    Udict[start]=(0,start)
    while len(Udict)>0: #Main search
        #Find node with min d in Udict and move to Edict
        dmin = dinit
        for n,w in Udict.items(): #A binary heap based upon the value of w would be more efficient here
            if w[0]<=dmin:
                dmin=w[0]
                nmin=n       
        Edict[nmin] = Udict.pop(nmin) #Mark the node with the minimum lowest maximum density as explored
        if nmin==dest or Edict[nmin][0]==dinit: #if our destination is reached or our node is not connected stop Dijkstra's
                if Edict[nmin][0]==dinit:       #checks if node is reachable from the source
                    Slist = [[],"No path exists between these two stations"]
                    return Slist
                else: break #Our destination is reachable so therefore we work out the path
        #Update provisional distances for unexplored neighbors of nmin
        for n,w in Alist[nmin]:
            if n in Udict: #node will either be in edict or udict so this is sufficient
                dcomp = max(dmin,w) #record maximum lowest density
                if dcomp<Udict[n][0]:
                    Udict[n]=(dcomp,nmin)
    Path = deque([]) #Convert to deque so that leftappend is O(1)
    Path.append(dest) #Work backwards, as this method saves me from having to store every path for each node
    P = dest
    while P != start: #Append to my path as I retrace the steps and terminate when the start has been reached
        P = Edict[P][1] #Extract the node that got me to node P to retrace my steps
        Path.appendleft(P) #Put this at the front of the deque objecct
    return (list(Path),Edict[nmin][0])


def shortJourney(Alist,start,dest):
    """
    Question 1.2 ii)
    Find shortest journey from station start to dest. If multiple shortest journeys
    exist, select journey which goes through the smallest number of stations.
    Input:
        Alist: List whose ith element contains list of 2-element tuples. The first element
        of the tuple is a station with a direct connection to station i, and the second element is
        the time for the connection (rounded to the nearest minute).
    start, dest: starting and destination stations for journey.

    Output:
        Slist: Two element list containing shortest journey and duration of shortest journey
    """

    #Initialize dictionaries
    dinit = 10**6
    Udict = {new_list: (dinit,0,dinit) for new_list in range(len(Alist))} #Unexplored nodes
    Edict = {} #Explored nodes
    Udict[start]=(0,start,dinit) #third parameter keeps track of the length of the shortest path to this node
    #Main search
    while len(Udict)>0:
        #Find node with min d in Udict and move to Edict
        dmin = dinit
        for n,w in Udict.items():#BINARY HEAP would be used here instead if heapq were allowed to be used
            if w[0]<=dmin:
                dmin=w[0]
                nmin=n
        Edict[nmin] = Udict.pop(nmin) #dictionary is used to make everything fast and easy to locate
        if nmin==dest or Edict[nmin][0]==10**6: #This ensures if the dest. node is disconnected, computer doesn't
                                             #have to go through all disconnected nodes until dest is reached                                             
                if Edict[nmin][0]==10**6:
                    Slist = [[],"No path exists between these two stations"]
                    return Slist
                else: break
        #Update provisional distances for unexplored neighbors of nmin
        for n,w in Alist[nmin]:
            if n in Udict: #same as algorithm in 1.2.i except we wish to find the minimum rolling sum instead
                dcomp = dmin + w
                if dcomp<Udict[n][0]:
                    Udict[n]=(dcomp,nmin,Edict[nmin][2]+1)
                elif dcomp==Udict[n][0] and Edict[nmin][2]+1<Udict[n][2]: # Check if path could be shorter as well
                    Udict[n]=(nmin,Edict[nmin][2]+1)
    Path = deque([]) #Identical set up to 1.2.i
    P = dest
    Path.append(dest)
    while P != start:
        P = Edict[P][1]
        Path.appendleft(P)
    return (list(Path),Edict[nmin][0])


def cheapCycling(SList,CList):
    """
    Question 1.3
    Find first and last stations for cheapest cycling trip
    Input:
        Slist: list whose ith element contains cheapest fare for arrival at and
        return from the ith station (stored in a 2-element list or tuple)
        Clist: list whose ith element contains a list of stations which can be
        cycled to directly from station i
    Stations are numbered from 0 to N-1 with N = len(Slist) = len(Clist)
    Output:
        stations: two-element list containing first and last stations of journey
    """
    L = [0 for l in CList] #Labels (1 if checked, else 0)
    setsofconnectednodes=[] #All my connected nodes
    for x in range(len(CList)):
            if L[x] == 0 and len(CList[x])>0: #If station is connected and unmarked, perform BFS (we ignore lonely nodes)
                Q=deque()#deque more efficient 
                Q.append(x)
                connectednodes=[] #sets of connected nodes
                L[x] = 1
                while len(Q)>0: #perform BFS on nodes to discover which are connected
                    y=Q.popleft() #remove first element from queue
                    connectednodes.append([y,SList[y][0],SList[y][1]]) 
                    for v in CList[y]:
                        if L[v]==0:
                            Q.append(v)
                            L[v]=1
                setsofconnectednodes.append(connectednodes) #add new set of connected nodes to the current set      
    minitial = 10**6 # initially use a big number
    minin1 = [-1,minitial]
    minin2 = [-1,minitial]
    minout1 = [-1,minitial]
    minout2 = [-1,minitial]
    rollingsum = [minitial,0,0]
    for nodes in setsofconnectednodes:
      X,Y = nodes.copy(),nodes.copy()
      X1 = min(X, key=lambda x: x[1]) #Tuple containing node with smallest arrival time 
      Y1 = min(Y, key=lambda x: x[2]) #Tuple containing node with smallest arrival time
      minin1[0],minin1[1] = X1[0],X1[1]
      minout1[0],minout1[1] = Y1[0],Y1[2]                                        
      X.remove(X1) 
      Y.remove(Y1)
      X2 = min(X, key=lambda x: x[1]) #Tuple containing node with smallest arrival time
      Y2 = min(Y, key=lambda x: x[2]) #Tuple containing node with smallest arrival time 
      minin2[0],minin2[1] = X2[0],X2[1]
      minout2[0],minout2[1] = Y2[0],Y2[2] 
      #Here we consider possible permutations which will give us the answer we want. 
      #If the minin and minout are different nodes then we don't need to make anymore comparisons.
      if minin1[0]!=minout1[0] and minin1[1]+minout1[1]<rollingsum[0]: 
      
          rollingsum[0],rollingsum[1],rollingsum[2]=minin1[1]+minout1[1],minin1[0],minout1[0] #compare permutations
      else:
          if minin1[0]!=minout2[0] and minin1[1]+minout2[1]<rollingsum[0]:
               rollingsum[0],rollingsum[1],rollingsum[2]=minin1[1]+minout2[1],minin1[0],minout2[0] #compare permutations
          
              
          if minin2[0]!=minout1[0] and minin2[1]+minout1[1]<rollingsum[0]:
               rollingsum[0],rollingsum[1],rollingsum[2]=minin2[1]+minout1[1],minin2[0],minout1[0] #compare permutations

    stations=[rollingsum[1],rollingsum[2]] #Gives us the stations we want
    return stations

if __name__=='__main__':
    #add code here if/as desired
    L=None #modify as needed

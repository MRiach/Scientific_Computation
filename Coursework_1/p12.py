def codonToAA(codon):
    #TAA,TAG, and TGA are superfluous to requirement as they represent the end of the string and only end 
    #up slowing down the function. 
	"""Return amino acid corresponding to input codon.
	Assumes valid codon has been provided as input
	"_" is returned for valid codons that do not
	correspond to amino acids.
	"""
	table = {
		'ATA':'i', 'ATC':'i', 'ATT':'i', 'ATG':'m',
		'ACA':'t', 'ACC':'t', 'ACG':'t', 'ACT':'t',
		'AAC':'n', 'AAT':'n', 'AAA':'k', 'AAG':'k',
		'AGC':'s', 'AGT':'s', 'AGA':'r', 'AGG':'r',
		'CTA':'l', 'CTC':'l', 'CTG':'l', 'CTT':'l',
		'CCA':'p', 'CCC':'p', 'CCG':'p', 'CCT':'p',
		'CAC':'h', 'CAT':'h', 'CAA':'q', 'CAG':'q',
		'CGA':'r', 'CGC':'r', 'CGG':'r', 'CGT':'r',
		'GTA':'v', 'GTC':'v', 'GTG':'v', 'GTT':'v',
		'GCA':'a', 'GCC':'a', 'GCG':'a', 'GCT':'a',
		'GAC':'d', 'GAT':'d', 'GAA':'e', 'GAG':'e',
		'GGA':'g', 'GGC':'g', 'GGG':'g', 'GGT':'g',
		'TCA':'s', 'TCC':'s', 'TCG':'s', 'TCT':'s',
		'TTC':'f', 'TTT':'f', 'TTA':'l', 'TTG':'l',
		'TAC':'y', 'TAT':'y', 'TAA':'_', 'TAG':'_',
		'TGC':'c', 'TGT':'c', 'TGA':'_', 'TGG':'w',
	}
	return table[codon]


def DNAtoAA(S):
    #Convert string to list to make it easier to perform with
    X=list(S)
    N=len(X)
    codons=[]
    AA=[]
    #A codon corresponds to a string of three letters, so the string is split into sets of three which we label
    #codons. Along the way, we check to see if the list has reached 20 characters. If this is the case, then
    #we have exhausted all amino acids and thus the loop is broken.
    for i in range(0,N-1,3):
        codons.append(X[i]+X[i+1]+X[i+2])
        if codonToAA(codons[int(i/3)]) not in AA:
          AA.append(codonToAA(codons[int(i/3)]))
        if len(AA)==20:
          AA=''.join(map(str, AA))
          return AA
    #This amalgamates the individual amino acids to return a string.
    AA=''.join(map(str, AA))
    return AA

def char2base4(S):
 """Convert gene test_sequence
 string to list of ints
 """
 c2b = {}
 c2b['A']=0
 c2b['C']=1
 c2b['G']=2
 c2b['T']=3
 L=[]
 for s in list(S):
  L+=[c2b[s]]
 return L


def heval(L,Base,Prime):
 """Convert list L to base-10 number mod Prime
 where Base specifies the base of L
 """
 f = 0
 for l in L[:-1]:
  f = Base*(l+f)
 h = (f + (L[-1])) % Prime
 return h



def pairSearch(L,pairs):
    """Find locations within adjacent strings (contained in input list,L)
    that match k-mer pairs found in input list pairs. Each element of pairs
    is a 2-element tuple containing k-mer strings
    """
    #Locations are stored in an array which is outputted
    locations = []
    #Length of all strings is constant so I take the length of the first element of my list. 
    N=len(L[0])
    #M here represents the M-mer pairs.
    M=len(pairs[0][0])
    #I iterate through each consecutive pair of strings provided in L, omitting the last as its only neighbour
    #is the string above it. 
    for i in range(0,len(L)-1):
        s1,s2=L[i],L[i+1]
        #I go through each pair that is provided for each pair of strings.
        for j in range(0,len(pairs)):
            #The first element of the set of pairs is assigned a hash value which is the target value. The 
            #rolling hash is evaluated at the beginning of the list requiring M computations. This is then 
            #compared with the target hash value. If both values are equal, then a direct comparisons between
            #the strings is made and the location is appended if the strings are indeed identical for adjacent
            #List1 and List2. Note: List2 is only checked if List1's substring matches the string in the first 
            #element of the pair. 
            targethash=heval(char2base4(pairs[j][0]),4,101)
            rollinghash=heval(char2base4(s1[0:M]),4,101)
            if rollinghash==targethash:
                if pairs[j][0]==s1[0:M] and pairs[j][1]==s2[0:M]:
                    locations.append([0,i,j])
            #The rolling hashes are now calculated in four calculations rather than M, providing for a much
            #more efficient algorithm in the case the rolling hash does not equal the target hash. Again, if
            #the target hash matches the rolling hash in List1, then we directly compare strings in the same
            #indices of List2.
            for k in range(1,N-M+1):
                rollinghash=(4*rollinghash-4**M*char2base4(s1[k-1])[0]+char2base4(s1[k-1+M])[0])%101
                if rollinghash==targethash:
                 if pairs[j][0]==s1[k:M+k] and pairs[j][1]==s2[k:M+k]:
                    locations.append([k,i,j])
    return locations


import time
pairs = [("TCG", "GAT"), ("AGC", "GAT"), ("TCG", "GAT"), ("TCG", "AAA")]
L = ["GCAATTCGT","TCGTTGATC", "ATCGATGTC", "CGGTAATCG", "AGGTTTAAA"]
print(pairSearch(L, pairs))

from __future__ import print_function
import numpy as np
from scipy import sparse
np.random.seed(1331)

def PLSA(X, k):
    # Return pLSA probability matrix p of m*n matrix X  
    # X is the document-word co-occurence matrix  
    # k is the number of the topics--z  
    # document--collums,word--rows  
    epsilon = np.finfo(float).eps
    x = X
    [m, n] = x.shape  # m words, n documents
    pz = np.random.rand(k).astype(np.float16)  # init p(z), k topics
      
    # random initialize
    pdz = np.random.rand(k,n).astype(np.float16)
    pwz = np.random.rand(k,m).astype(np.float16)
    pzdw = np.random.rand(m,n,k).astype(np.float16)
    R=sum(sum(x))
    h=0
      
    #  iteration   
    iteration=0;  
    print('EM Training:');  
    for iteration in xrange(101):
        deno = np.zeros(k).astype(np.float16)                  # denominator of p(d/z) and p(w/z)  
        denopzdw = np.zeros((m,n)).astype(np.float16)          # denominator of p(z/d,w)  
        numepdz = np.zeros((k,n)).astype(np.float16)           # numerator of p(d/z)  
        numepwz = np.zeros((k,m)).astype(np.float16)           # numerator of p(w/z)  
        print("iteration :" ,iteration, end="\r")
        for ki in xrange(k):
            for i in xrange(m):
                for j in xrange(n):
                    deno[ki] = deno[ki] + x[i, j] * pzdw[i, j, ki]
                    
        # p(d/z)  
        for ki in xrange(k):
            for j in xrange(n):
                for i in xrange(m):  
                    # denominator of P(d|z), matrix
                    numepdz[ki,j] = numepdz[ki, j] + x[i, j] * pzdw[i, j, ki]
                pdz[ki,j] = numepdz[ki,j] / deno[ki]  # numerator / denominator
        # disp(pdz);  
          
        # p(w/z)  
        for ki in xrange(k):
            for i in xrange(m):
                for j in xrange(n):
                    # denominator of P(w|z), matrix
                    numepwz[ki,i]=numepwz[ki,i] + x[i,j] * pzdw[i,j,ki]
                # numerator / denominator
                pwz[ki,i] = numepwz[ki,i] / deno[ki]
        # disp(pwz);  
          
        # p(z)  
        for ki in xrange(k):
            pz[ki]=deno[ki] / R  # denominator

        # denominator of p(z/d,w)  
        for i in xrange(m):
            for j in xrange(n):
                for ki in xrange(k):
                    # denominator of P(z|d,w), matrix
                    denopzdw[i, j] = denopzdw[i, j] + pz[ki] * pdz[ki, j] * pwz[ki, i]

        # p(z/d,w)  
        for i in xrange(m):
             for j in xrange(n):
                 for ki in xrange(k):
                     # P(z|d,w), 3D tensor
                     pzdw[i,j,ki] = pz[ki] * pdz[ki,j] * pwz[ki,i] / denopzdw[i,j]
        # disp(pzdw)  
    print()    
    return [pz, pdz, pwz, pzdw]

X = np.random.rand(5, 8)
k=4
[pz, pdz, pwz, pzdw]=PLSA(X,k)
print(pz)
raw_input()
print(pdz)
print(pwz)
print(pzdw)
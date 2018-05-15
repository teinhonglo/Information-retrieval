from __future__ import print_function
import numpy as np
from scipy import sparse
np.random.seed(1331)

def calc_x_pzdw(x, pzdw):
    for ki in xrange(k):
        pzdw[:, :, ki] = x * pzdw[:, :, ki]
    return pzdw

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
    pdz = np.random.rand(k,n)
    pwz = np.random.rand(k,m)
    pzdw = np.random.rand(m,n,k)
    R=sum(sum(x))
    h=0
      
    #  iteration   
    iteration=0;  
    print('EM Training:');  
    for iteration in xrange(101):
        deno = np.zeros(k)                  # denominator of p(d/z) and p(w/z)  
        denopzdw = np.zeros((m,n))          # denominator of p(z/d,w)  
        numepdz = np.zeros((k,n))           # numerator of p(d/z)  
        numepwz = np.zeros((k,m))           # numerator of p(w/z)  
        print("iteration :" ,iteration, end="\r")
        pzdw = calc_x_pzdw(x, pzdw)
        deno = deno + np.sum(np.sum(pzdw, axis=0), axis=0)
                    
        # p(d/z)  
        for ki in xrange(k):
            for j in xrange(n):
                # denominator of P(d|z), matrix
                numepdz[ki,j] = numepdz[ki, j] + np.sum(pzdw, axis=0)[j, ki]
                pdz[ki,j] = numepdz[ki,j] / deno[ki]  # numerator / denominator
        # disp(pdz);  
          
        # p(w/z)  
        for ki in xrange(k):
            for i in xrange(m):
                # denominator of P(w|z), matrix
                numepwz[ki,i]=numepwz[ki,i] + np.sum(pzdw, axis=1)[i, ki]
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
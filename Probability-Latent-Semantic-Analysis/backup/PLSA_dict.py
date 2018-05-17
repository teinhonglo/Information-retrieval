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
      
    # random initialize
    pzd = np.random.rand(n,k)
    pwz = np.random.rand(k,m)
    pzdw = np.random.rand(m,n,k)
    R=sum(sum(x))
    h=0
      
    #  iteration   
    iteration=0;  
    print('EM Training:');  
    for iteration in xrange(101):
        denopzdw = np.zeros((m,n))          # denominator of p(z/d,w)  
        deno = np.zeros((k))                # denominator of p(w/z) and p(d/z)
        numepzd = np.zeros((n,k))           # numerator of p(z/d)  
        numepwz = np.zeros((k,m))           # numerator of p(w/z)  
        print("iteration :" ,iteration, end="\r")
        # E step
        # denominator of p(z/d,w)  
        for i in xrange(m):
            for j in xrange(n):
                for ki in xrange(k):
                    denopzdw[i, j] += pwz[ki, i] * pzd[j, ki]
        # p(z|w,d)
        for i in xrange(m):
            for j in xrange(n):
                for ki in xrange(k):
                    pzdw[i, j, ki] = pwz[ki, i] * pzd[j, ki] / denopzdw[i, j]
        # M step
        # numerator of p(w/z)
        for ki in xrange(k):
            for i in xrange(m):
                for j in xrange(n):
                    numepwz[ki, i] += x[i, j] * pzdw[i, j, ki]
                # denominator of p(w/z)    
                deno[ki] += numepwz[ki, i]
        # p(w/z)
        for ki in xrange(k):
            for i in xrange(m):
                pwz[ki, i] /= deno[ki]
                
        deno = np.zeros((n))                # denominator of p(w/z) and p(d/z)
        
        # p(z|d)  
        for ki in xrange(k):
            for j in xrange(n):
                # denominator of P(z|d)
                for i in xrange(m):
                    numepzd[j, ki] += x[i, j] * pzdw[i, j, ki]
                    deno[j] += x[i, j]
                pzd[j, ki] = numepzd[j, ki] / deno[j]

    print()    
    return [pzd, pwz, pzdw]

X = np.random.rand(5, 8)
k=4
[pzd, pwz, pzdw]=PLSA(X,k)
print(pwz)
raw_input()
print(pzd)
print(pzdw)
from __future__ import print_function
import numpy as np
import itertools
np.random.seed(1331)

def calc_x_pzdw(x, pzdw):
    m, n, k = pzdw.shape
    print(k)
    pzdw_ = np.zeros((m,n,k))
    for ki in xrange(k):
        pzdw_[:, :, ki] = x * np.copy(pzdw[:, :, ki])
    return pzdw_

def PLSA(X, k):
    # Return pLSA probability matrix p of m*n matrix X  
    # X is the document-word co-occurence matrix  
    # k is the number of the topics--z  
    # document--collums,word--rows  
    epsilon = np.finfo(float).eps
    x = X
    nm, nn = np.nonzero(x)
    [m, n] = x.shape  # m words, n documents
    # random initialize
    pzd = np.random.rand(n,k)
    print(pzd)
    print()
    pwz = np.random.rand(k,m)
    print(pwz)
    print()
    pzdw = np.random.rand(m,n,k)
    R=sum(sum(x))
    h=0
      
    #  iteration   
    iteration=0;  
    print('EM Training:');  
    for iteration in xrange(1):
        denopzdw = np.zeros((m,n))          # denominator of p(z/d,w)  
        deno = np.zeros((k)) + epsilon      # denominator of p(w/z) and p(d/z)
        numepzd = np.zeros((n,k))           # numerator of p(z/d)  
        numepwz = np.zeros((k,m))           # numerator of p(w/z)  
        print("iteration :" ,iteration, end="\r")
        # E step
        for i in xrange(m):
            for j in xrange(n):
                for ki in xrange(k):
                    # numerator of p(z/d,w)
                    pzdw[i, j, ki] = pwz[ki, i] * pzd[j, ki]
                    # denominator of p(z/d,w)  
                    denopzdw[i, j] += pwz[ki, i] * pzd[j, ki]
        # p(z|w,d)
        for i in xrange(m):
            for j in xrange(n):
                for ki in xrange(k):
                    # numerator / denominator
                    pzdw[i, j, ki] /= denopzdw[i, j]
        print("p(z|d, w)")                    
        print(pzdw)
        # M step
        # numerator of p(w/z)
        pzdw = calc_x_pzdw(x, pzdw)
        numepwz = np.transpose(np.sum(pzdw, axis = 1))
        # denominator of p(w/z)
        deno += np.sum(numepwz, axis = 1)
        # p(w/z)
        for ki in xrange(k):
            for i in xrange(m):
                pwz[ki, i] = numepwz[ki, i] / deno[ki]
                
        print(pwz)
        deno = np.zeros((n)) + epsilon              # denominator of p(w/z) and p(d/z)
        
        # p(z|d)  
        # numerator of p(z/d)
        pzd = np.sum(pzdw, axis = 0)
        # denominator of p(z/d)
        deno += np.sum(x, axis = 0)
        pzd[j, ki] /= deno[j]

    print()    
    return [pzd, pwz, pzdw]

X = np.random.randint(4, size=(3, 4)).astype("float")
print(X)
k=2
[pzd, pwz, pzdw]=PLSA(X,k)
print(pwz)
from __future__ import print_function
import time
import numpy as np
import itertools
np.random.seed(1331)

def calc_x_pzdw(x, pzdw):
    m, n, k = pzdw.shape
    pzdw *= x[:,:, None]
    return pzdw

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
        print("iteration :" ,iteration)
        print("E step:")
        for i in xrange(m):
            for j in xrange(n):
                for ki in xrange(k):
                    # numerator of p(z/d,w)
                    pzdw[i, j, ki] = pwz[ki, i] * pzd[j, ki]
                    
        print("p(z|w,d)")
        # denominator of p(z/d,w)  
        denopzdw = np.sum(pzdw, axis=2)
        '''
        for i in xrange(m):
            for j in xrange(n):
                for ki in xrange(k):
                    # numerator / denominator
                    pzdw[i, j, ki] /= denopzdw[i, j]
        '''
        pzdw /= denopzdw[:,:,None]
        # M step
        print("M step:")
        # numerator of p(w/z)
        pzdw = calc_x_pzdw(x, pzdw)
        numepwz = np.transpose(np.sum(pzdw, axis = 1))
        # denominator of p(w/z)
        deno += np.sum(numepwz, axis = 1)
        # p(w/z)
        '''
        for ki in xrange(k):
            for i in xrange(m):
                pwz[ki, i] = numepwz[ki, i] / deno[ki]
        '''
        pwz = numepwz / deno[:, None]        
        deno = np.zeros((n)) + epsilon              # denominator of p(w/z) and p(d/z)
        # p(z|d)  
        # numerator of p(z/d)
        pzd = np.sum(pzdw, axis = 0)
        # denominator of p(z/d)
        deno += np.sum(x, axis = 0)
        pzd /= deno[:, None]
        print()
    return [pzd, pwz, pzdw]

#X = np.random.randint(4, size=(14000, 2265)).astype("float")    
X = np.random.randint(4, size=(3, 4)).astype("float")
print(X)
k=2
import time
start_time = time.time()
[pzd, pwz, pzdw]=PLSA(X,k)
end_time = time.time()
print(end_time - start_time)
print(pwz)
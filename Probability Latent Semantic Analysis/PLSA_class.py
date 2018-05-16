from __future__ import print_function
import time
import numpy as np
import itertools
np.random.seed(1331)


class PLSA(object):
    def __init__(self, X, k, pwz = None, pzd = None):
        # X is the document-word co-occurence matrix  
        # word--rows(m), document--collums(n)
        # k is the number of the topics--z  
        self.epsilon = np.finfo(float).eps
        self.x = X
        self.nm, self.nn = np.nonzero(self.x)
        [m, n] = self.x.shape  # m words, n documents
        # p(z|D)
        if pzd == None:
            # random initialize
            self.pzd = np.random.rand(n,k)
        else:
            self.pzd = pzd
        if pwz == None:
            # random initialize
            self.pwz = np.random.rand(k,m)
        else:
            self.pwz
        # p(z | w, D)
        self.pzdw = np.random.rand(m,n,k)
    
    def EM_Trainging(self, iteration=20):
        # Return pLSA probability matrix p of m*n matrix X  
        # co-occurence matrix
        x = self.x
        # Latent variable model
        pzd = self.pzd
        pwz = self.pwz
        pzdw = self.pzdw
        m, n = x.shape
        epsilon = self.epsilon
        #  iteration   
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
            pzdw /= denopzdw[:,:,None]
            # M step
            print("M step:")
            # cache function of p(w/z)
            pzdw = self.calc_x_pzdw(x, pzdw)
            # p(w/z)
            # numerator of p(w/z) 
            numepwz = np.transpose(np.sum(pzdw, axis = 1))
            # denominator of p(w/z)
            deno += np.sum(numepwz, axis = 1)
            pwz = numepwz / deno[:, None]        
            # p(z|d)  
            # initialize denominator of p(z/d)
            deno = np.zeros((n)) + epsilon              
            # numerator of p(z/d)
            pzd = np.sum(pzdw, axis = 0)
            # denominator of p(z/d)
            deno += np.sum(x, axis = 0)
            pzd /= deno[:, None]
            print()
        return [pzd, pwz, pzdw]
        
    def Objetcive(self, x, pwz, pzd):
        m = self.nm
        n = self.nn
        for i, j in itertools.izip(m, n):
            pass
        pass    
    
    def calc_x_pzdw(self, x, pzdw):
        m, n, k = pzdw.shape
        pzdw *= x[:,:, None]
        return pzdw
        
X = np.random.randint(4, size=(14000, 2265)).astype("float")    
#X = np.random.randint(4, size=(3, 4)).astype("float")
print(X)
k=4
model = PLSA(X, k)
import time
start_time = time.time()
[pzd, pwz, pzdw]=model.EM_Trainging(1)
end_time = time.time()
print(end_time - start_time)
print(pwz)
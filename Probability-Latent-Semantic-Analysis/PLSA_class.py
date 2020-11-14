from __future__ import print_function
import time
import numpy as np
import itertools
np.random.seed(1331)


class pLSA(object):
    def __init__(self, X, k, pwz = None, pzd = None):
        # X is the document-word co-occurence matrix  
        # word -> rows(m), document -> collums(n)
        # k is the number of the topics--z  
        self.epsilon = np.finfo(float).eps
        self.x = X
        self.k = k
        [self.nm, self.nn] = np.nonzero(self.x)
        [m, n] = self.x.shape  # m words, n documents
        # p(z|D)
        if pzd is None:
            # random initialize
            self.pzd = np.random.rand(n,k)
        else:
            self.pzd = pzd
        if pwz is None:
            # random initialize
            self.pwz = np.random.rand(k,m)
        else:
            self.pwz = pwz
        # p(z | w, D)
        self.pzdw = np.random.rand(m,n,k)
        print(self.x.shape, self.pwz.shape, self.pzd.shape, self.pzdw.shape)
    
    def EM_Trainging(self, iteration=20):
        # Return pLSA probability matrix
        # co-occurence matrix (m * n)
        x = self.x
        # Latent variable model
        pzd = self.pzd
        pwz = self.pwz
        pzdw = self.pzdw
        k = self.k
        m, n = x.shape
        print(m, n, k)
        epsilon = self.epsilon
        #  iteration   
        print('EM Training:');  
        for iteration in range(iteration):
            start_time = time.time()
            denopzdw = np.zeros((m,n))          # denominator of p(z/d,w)  
            deno = np.zeros((k)) + epsilon      # denominator of p(w/z) and p(d/z)
            #numepzd = np.zeros((n,k))           # numerator of p(z/d)  
            numepwz = np.zeros((k,m))           # numerator of p(w/z)  
            count = 0
            print("iteration :" ,iteration)
            print("E step:")
            for i in range(m):
                for j in range(n):
                    for ki in range(k):
                        # numerator of p(z/d,w)
                        pzdw[i, j, ki] = pwz[ki, i] * pzd[j, ki]

            # denominator of p(z/d,w)  
            denopzdw = np.sum(pzdw, axis=2)
            pzdw /= denopzdw[:,:,None]
            where_are_NaNs = np.isnan(pzdw)
            pzdw[where_are_NaNs] = 0
            
            # M step
            print("M step:")
            # cache function of p(w/z)
            pzdw = self.__calc_x_pzdw(x, pzdw)
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
            self.__obj_function(x, pwz, pzd, pzdw)
            end_time = time.time()
            print(end_time - start_time)
            print()
        return [pzd, pwz, pzdw]
        
    def __obj_function(self, x, pwz, pzd, pzdw):
        print("Estimation")
        # maximum collection likelihood estimation
        k = pzdw.shape[2]
        loss = 0.
        for i, j in zip(self.nm, self.nn):
            pwd = 0.
            for ki in range(k):
                pwd += pwz[ki, i] * pzd[j, ki]
            loss += x[i, j] * np.log(pwd)
        print(loss)
    
    def __calc_x_pzdw(self, x, pzdw):
        m, n, k = pzdw.shape
        pzdw *= x[:,:, None]
        return pzdw

if __name__ == "__main__":
    # Example
    X = np.random.randint(4, size=(14000, 2265)).astype("float")    
    #X = np.random.randint(4, size=(3, 4)).astype("float")
    k=2
    iteration=16
    model = pLSA(X, k)    
    start_time = time.time()
    [pzd, pwz, pzdw]=model.EM_Trainging(iteration)
    end_time = time.time()
    print("Total:", end_time - start_time)
    print(pwz)

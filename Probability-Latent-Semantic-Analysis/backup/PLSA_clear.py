import numpy as np
np.random.seed(1331)

def PLSA(X, k):
    # Return pLSA probability matrix p of m*n matrix X  
    # X is the document-word co-occurence matrix  
    # k is the number of the topics--z  
    # document--collums,word--rows  
    epsilon = np.finfo(float).eps
    err=0.0001;  
    x = X;   
    [m, n]=x.shape;  # m words, n documents
    pz=np.random.rand(k)  # init pz, k topics
    pz2=pz;  
    pz2[0]=pz2[0] + 2 * err  
      
    # random initialize
    pdz=np.random.rand(k,n)
    pwz=np.random.rand(k,m)
    pzdw=np.random.rand(m,n,k)
    h=0
      
    deno=np.zeros(k)        #denominator of p(d/z) and p(w/z)  
    denopzdw=np.zeros((m,n))       #denominator of p(z/d,w)  
    numepdz=np.zeros((k,n))        #numerator of p(d/z)  
    numepwz=np.zeros((k,m))        # numerator of p(w/z)  
    R=sum(sum(x))
      
    for ki in xrange(k):
        for i in xrange(m):
            for j in xrange(n):
                # numerator od P(w|z) and P(d|z), vector
                deno[ki] = deno[ki] + x[i, j] * pzdw[i, j, ki]
                
    # p(d/z)  
    for ki in xrange(k):
        for j in xrange(n):
            for i in xrange(m):  
                # denominator of P(d|z), matrix
                numepdz[ki,j]=numepdz[ki,j]+x[i,j]*pzdw[i,j,ki]
            pdz[ki,j]=numepdz[ki,j]/deno[ki]  # numerator / denominator
    # disp(pdz);  
      
    # p(w/z)  
    for ki in xrange(k):
        for i in xrange(m):
            for j in xrange(n):
                # denominator of P(w|z), matrix
                numepwz[ki,i]=numepwz[ki,i]+x[i,j]*pzdw[i,j,ki]
            # numerator / denominator
            pwz[ki,i]=numepwz[ki,i]/deno[ki]
    # disp(pwz);  
      
    # p(z)  
    for ki in xrange(k):
        pz[ki]=deno[ki] / R  # denominator

    # denominator of p(z/d,w)  
    for i in xrange(m):
        for j in xrange(n):
            for ki in xrange(k):
                # denominator of P(z|d,w), matrix
                denopzdw[i,j]=denopzdw[i,j] + pz[ki] * pdz[ki,j] * pwz[ki,i]

    # p(z/d,w)  
    for i in xrange(m):
         for j in xrange(n):
             for ki in xrange(k):
                 # P(z|d,w), 3D tensor
                 pzdw[i,j,ki] = pz[ki] * pdz[ki,j] * pwz[ki,i] / denopzdw[i,j]
    # disp(pzdw)  
      
    #  iteration   
    iteration=0;  
    print('iteration:');  
    for iteration in xrange(100):
        deno=np.zeros(k)        #denominator of p(d/z) and p(w/z)  
        denopzdw=np.zeros((m,n))       #denominator of p(z/d,w)  
        numepdz=np.zeros((k,n))        #numerator of p(d/z)  
        numepwz=np.zeros((k,m))        # numerator of p(w/z)  
        print("iteration :" ,iteration);  
        for ki in xrange(k):
            for i in xrange(m):
                for j in xrange(n):
                    deno[ki] = deno[ki] + x[i,j] * pzdw[i,j,ki]
        # p(d/z)  
        for ki in xrange(k):
            for j in xrange(n):
                for i in xrange(m):
                    numepdz[ki,j] = numepdz[ki,j] + x[i,j] * pzdw[i,j,ki]
                pdz[ki,j]=numepdz[ki,j] / deno[ki]
        print('p(d/z)')
        # p(w/z)  
        for ki in xrange(k):
            for i in xrange(m):
                for j in xrange(n):
                    numepwz[ki,i] = numepwz[ki,i] + x[i,j] * pzdw[i,j,ki]
                pwz[ki,i] = numepwz[ki,i] / deno[ki]  
        print('p(w/z)');  
        # p(z)  
        pz=pz2;  
        for ki in xrange(k):
            pz2[ki] = deno[ki] / R;  
        
        print('p(z)');  
        #denominator of p(z/d,w)  
        for i in xrange(m):
            for j in xrange(n):
                for ki in xrange(k):
                    denopzdw[i,j] = denopzdw[i,j]+ pz2[ki] * pdz[ki,j] * pwz[ki,i]
        # p(z/d,w)  
        for i in xrange(m):
             for j in xrange(n):
                 for ki in xrange(k):
                     pzdw[i,j,ki]=pz2[ki]*pdz[ki,j]*pwz[ki,i] / (denopzdw[i,j] + epsilon)
    return [pz, pdz, pwz, pzdw]

# function demo  
#   
X = np.random.rand(5, 8)
k=4
[pz, pdz, pwz, pzdw]=PLSA(X,k)
print(pz)
raw_input()
print(pdz)
print(pwz)
print(pzdw)
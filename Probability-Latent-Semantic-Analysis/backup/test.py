import numpy as np
import itertools
np.random.seed(1)

a = np.random.randint(3, size=(4,3))
b = np.random.randint(1, 6, size=(4)).astype("float")
print a, b
print a / b


'''    
a = np.random.randint(2, size=(2, 4))
print a
c_idx, d_idx = np.where(a != 0)
for i, j in itertools.izip(c_idx, d_idx):
    print a[i, j]


m, n, k = 2, 4, 6
pwz = np.zeros((k, m))
pwz_ = np.zeros((k, m))
x = np.random.rand(m, n)
pzwd = np.random.rand(m, n, k)

for ki in xrange(k):
    for i in xrange(m):
        for j in xrange(n):
            pwz[ki, i] +=x[i, j] * pzwd[i, j, ki]

print pwz        
for ki in xrange(k):
    for j in xrange(n):
        for i in xrange(m):
            pwz_[ki, i] +=x[i, j] * pzwd[i, j, ki]

print pwz_ == pwz          

b_i = np.zeros((m, n, k))
for aa in xrange(m):
    for bb in xrange(n):
        for cc in xrange(k):
            b_i[aa, bb, cc] = b[aa, bb, cc]
            
c = np.zeros((m, n, k))
d = np.zeros(k)

for ki in xrange(k):
    b[:, :, ki] *= a
    
b = np.multiply(a, b[:,:, np.newaxis])    
    
for ki in xrange(k):
    for i in xrange(m):
        for j in xrange(n):
            d[ki] = d[ki] + a[i, j] * b_i[i, j, ki]    

print b
print d
print np.sum(np.sum(b, axis=0), axis=0)
'''
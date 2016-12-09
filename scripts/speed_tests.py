from __future__ import print_function
import sys, time

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../build/pybind11/Release')
import ChebTools as CT

N = 100000

c = range(50)
c1 = CT.ChebyshevExpansion(c)
tic = time.clock()
CT.mult_by(c1, 1.0001, N)
toc = time.clock()
print((toc-tic)/N*1e6,'us/call')

c1 = CT.ChebyshevExpansion(c)
tic = time.clock()
CT.mult_by_inplace(c1, 1.0001, N)
toc = time.clock()
print((toc-tic)/N*1e6,'us/call')

tic = time.clock()
c1 = CT.ChebyshevExpansion(c)
for i in range(N):
    c2 = c1*1.0001
toc = time.clock()
print((toc-tic)/N*1e6,'us/call')

Norder = 50
Npoints = 200
x = np.linspace(0, 1, Npoints)
tic = time.clock()
for i in range(N):
    c1.y(x)
toc = time.clock()
print((toc-tic)/N*1e6,'us for 200 values')

tic = time.clock()
A = np.random.random((50,50))
c1 = CT.ChebyshevExpansion(c)
N = 100
for i in range(N):
    eigvals = np.linalg.eigvals(A)
toc = time.clock()
print((toc-tic)/float(N)*1e6,'us/call')

c1 = CT.ChebyshevExpansion([0,1,2,3,4,5])
c2 = CT.ChebyshevExpansion([-1,1])
c1 += c2
print(list(c1.coef()))
print (c1 + c2).coef()

# print (3.0*c1).coef()
# print (c1*3.0).coef()
# c1 *= 3.0
# print c1.coef()
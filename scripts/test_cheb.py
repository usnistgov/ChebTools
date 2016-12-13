from __future__ import print_function
import sys, time

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../build/pybind11/Release')
import ChebTools as CT

f = lambda x: np.exp(-5*(x-1)**2)-0.5

T = 110
import CoolProp, CoolProp.CoolProp as CP
AS = CoolProp.CoolProp.AbstractState('HEOS','n-Propane')
AS.specify_phase(CP.iphase_gas) # Something homogeneous
def f(delta):
    AS.update(CP.DmolarT_INPUTS, delta*AS.rhomolar_reducing(), T)
    return AS.p() - 1e4

xmax = 6
xmin = 0.00000001

xx = np.linspace(0.0001,6,200)
for Npower in [5,10,20,40,80,160,320]:
    ce = CT.generate_Chebyshev_expansion(Npower, f, xmin, xmax)
    N = 100
    tic = time.clock()
    for i in range(N):
        ce.y(xx)
    # print(ce.coef())
    toc = time.clock()
    print((toc-tic)/N*1e6,'us/call to evaluate Chebyshev of order:', Npower)

tic = time.clock()
ce = CT.generate_Chebyshev_expansion(50, f, xmin, xmax)
# print(ce.coef())
toc = time.clock()
print((toc-tic)*1e6,'us to generate Chebyshev')

N = 100
tic = time.clock()
for i in range(N):
    real_rts = ce.real_roots(True)
print(sorted(real_rts))
toc = time.clock()
print((toc-tic)/N*1e6,'us/call (big eigenvalue solve in C++)')

N = 100
tic = time.clock()
for i in range(N):
    roots = ((xmax-xmin)*np.linalg.eigvals(ce.companion_matrix()) + (xmax + xmin))/2.0
    real_rts = [rt for rt in roots if np.isreal(rt) and rt < xmax and rt >= xmin]
print(sorted(real_rts))
toc = time.clock()
print((toc-tic)/N*1e6,'us/call (big eigenvalue solve in python)')

N = 100
tic = time.clock()
for i in range(N):
    intervals = ce.subdivide(20, 5)
    real_rts = ce.real_roots_intervals(intervals, True)
print(sorted(real_rts))
toc = time.clock()
print((toc-tic)/N*1e6,'us/call (subdivided)')

N = 300000
tic = time.clock()
for i in range(N):
    ce.real_roots_approx(100)
toc = time.clock()
print(sorted(ce.real_roots_approx(100)))
print((toc-tic)/N*1e6,'us/call')

data = []
for N in range(10, 50):
    tic = time.clock()
    try:
        ce = CT.generate_Chebyshev_expansion(N, f, xmin, xmax)
        roots = ((xmax-xmin)*np.linalg.eigvals(ce.companion_matrix()) + (xmax + xmin))/2.0
        real_rts = [rt for rt in roots if np.isreal(rt) and rt < xmax and rt >= xmin]
        toc = time.clock()
        data.append((N, toc-tic, len(real_rts)))

    except BaseException as BE:
        print(N, BE)

N, elap, ct = zip(*data)
plt.plot(N, elap)
plt.yscale('log')
plt.show()

print(sorted(real_rts))
for rt in roots:
    if np.isreal(rt) and rt < xmax and rt >= xmin:   
        plt.axvline(np.sqrt(rt))

xx = np.linspace(xmin, xmax, 1000)
plt.plot(np.sqrt(xx), [f(_x) for _x in xx])
plt.plot(np.sqrt(xx), ce.y(xx))

plt.yscale('symlog')
plt.show()
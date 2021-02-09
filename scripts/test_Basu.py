"""
A driver script to test the implementation in cheb2d_Basu for correctness and speed (in Python at least)
"""
import timeit
from cheb2d_Basu import *
import matplotlib.pyplot as plt
import pandas

x_ = 0.7
y_ = 0.7

# f = lambda x, y: np.sin(x)*np.cos(-0.01*y**2)
f = lambda x, y: x**3*np.sin(-x**2)*np.cos(x)*np.exp(-x**2-y**2)

o = []
for M in [4,6,8,10,12,14,16]:
    print(M)
    a = get_coeff_mat(f, m=M, n=M)

    N = 100

    tic = timeit.default_timer()
    for i in range(N):
        eval_naive(amat=a, x=x_, y=y_)
    toc = timeit.default_timer()
    time_naive = (toc-tic)/N*1e6
    print(time_naive, 'us/call for naive implementation')

    tic = timeit.default_timer()
    for i in range(N):
        eval_Clenshaw(amat=a, x=x_, y=y_)
    toc = timeit.default_timer()
    time_Clenshaw = (toc-tic)/N*1e6
    print(time_Clenshaw, 'us/call for Clenshaw implementation in 2D')

    assert(abs(eval_Clenshaw(amat=a, x=x_, y=y_) - eval_naive(amat=a, x=x_, y=y_)) < 1e-14)

    print(
        eval_Clenshaw(amat=a, x=x_, y=y_), 
        eval_naive(amat=a, x=x_, y=y_), 
        f(x_, y_)
    )
    o.append({
        'M': M,
        'naive': time_naive,
        'Clenshaw': time_Clenshaw
        })

df = pandas.DataFrame(o)
plt.plot(df['M'], df['naive'], 'o-')
plt.plot(df['M'], df['Clenshaw'], 'o-')
plt.yscale('log')
plt.gca().set(xlabel='$M$, of $M^2$ matrix', ylabel='time / s')
plt.show()
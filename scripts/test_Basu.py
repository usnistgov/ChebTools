"""
A driver script to test the implementation in cheb2d_Basu for correctness and speed (in Python at least)
"""
import timeit
from cheb2d_Basu import *

x_ = 0.2
y_ = 0.1

# f = lambda x, y: np.sin(x)*np.cos(-0.01*y**2)
f = lambda x, y: x**3*np.sin(-x**2)*np.cos(x)*np.exp(-x**2-y**2)

a = get_coeff_mat(f, m=10, n=10)

N = 100

tic = timeit.default_timer()
for i in range(N):
    eval_naive(amat=a, x=x_, y=y_)
toc = timeit.default_timer()
print((toc-tic)/N*1e6)

tic = timeit.default_timer()
for i in range(N):
    eval_Clenshaw(amat=a, x=x_, y=y_)
toc = timeit.default_timer()
print((toc-tic)/N*1e6)

assert(abs(eval_Clenshaw(amat=a, x=x_, y=y_) - eval_naive(amat=a, x=x_, y=y_)) < 1e-14)

print(
    eval_Clenshaw(amat=a, x=x_, y=y_), 
    eval_naive(amat=a, x=x_, y=y_), 
    f(x_, y_)
)
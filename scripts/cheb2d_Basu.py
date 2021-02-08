# Naive implementation of::
# Basu, SIAM, 1973: 10.1137/0710045

import numpy as np

def TCheb(n, x):
    if n == 0: 
        return 1.0
    elif n == 1: 
        return x
    else:
        return 2.0*x*TCheb(n-1, x) - TCheb(n-2, x)

def get_coeff_mat(f, *, m, n):

    xroots = np.cos((2*np.arange(0, m+1, 1) + 1)*np.pi/(2*(m+1)))
    yroots = np.cos((2*np.arange(0, n+1, 1) + 1)*np.pi/(2*(n+1)))

    a = np.zeros((m+1, n+1))
    for i in range(m+1):
        for j in range(n+1):
            dsum = 0
            for r in range(0, m+1):
                for s in range(0, n+1):
                    xr = xroots[r]
                    ys = yroots[s]
                    dsum += f(xr, ys)*TCheb(i, xr)*TCheb(j, ys)
            a[i,j] = dsum
    return 4.0/((m+1)*(n+1))*a

def eval_naive(*, amat, x, y):
    m, n = amat.shape
    m -= 1
    n -= 1

    dsum = 0
    for i in range(m+1):
        for j in range(n+1):
            contrib = amat[i, j]*TCheb(i, x)*TCheb(j, y)
            if i == j == 0:
                contrib /= 4
            elif i == 0 and j > 0:
                contrib /= 2
            elif j == 0 and i > 0:
                contrib /= 2
            dsum += contrib
    return dsum

def eval_Clenshaw(*, amat, x, y):
    m, n = amat.shape
    m -= 1
    n -= 1

    def Clenshaw(c, ind):
        N = len(c) - 1
        u_k = 0; u_kp1 = 0; u_kp2 = 0
        k = 0
        for k in range(N, -1, -1):
            # Do the recurrent calculation
            u_k = 2.0*ind*u_kp1 - u_kp2 + c[k]
            if k > 0:
                # Update the values
                u_kp2 = u_kp1; u_kp1 = u_k
        return (u_k - u_kp2)/2

    b = np.zeros((m+1))
    for i in range(len(b)):
        b[i] = Clenshaw(amat[i,:], y)
    return Clenshaw(b, x)
import numpy as np

# See Mason p. 34, and example in https://github.com/numpy/numpy/blob/master/numpy/polynomial/chebyshev.py#L868-L964

def deriv(c, Nderiv = 1):
    for deriv_counter in range(Nderiv):
        N = len(c) - 1
        Nd = N-1 
        cd = np.zeros((Nd+1,))
        for r in range(Nd+1):
            cd[r] = 0
            for k in range(r+1, Nd+2):
                if (k-r)%2 == 1:
                    cd[r] += 2*k*c[k]
                    print (r, k, 2*k*c[k])
            if r == 0:
                cd[r] /= 2
        c = cd
    return c

if __name__ == '__main__':
    c = (1,2,3,4)
    print(deriv(c))
    print(deriv(c,3))

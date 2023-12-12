import ChebTools
import numpy as np
from scipy.special import factorial 
import matplotlib.pyplot as plt

class TaylorExtrapolator():
    def __init__(self, ce, x, degree):
        self.coef = [ce.deriv(d).y(x) for d in range(degree+1)]
        self.x0 = x

    def __call__(self, x):
        dx = x - self.x0
        return sum([c_n*dx**n/factorial(n) for n, c_n in enumerate(self.coef)])

if __name__ == '__main__':
    f = lambda x: np.cosh(x)
    ce = ChebTools.generate_Chebyshev_expansion(21, f, -1, 1)
    te = TaylorExtrapolator(ce, 0.8, 8)
    tay = ChebTools.make_Taylor_extrapolator(ce, 0.8, 8)

    print(tay(2.0), f(2.0))

    x = np.linspace(0.8, 3)
    plt.plot(x, f(x))
    plt.plot(x, te(x))
    plt.plot(x, tay(x))
    plt.show()
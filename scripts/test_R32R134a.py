"""
A driver script to test the implementation in cheb2d_Basu for correctness and speed (in Python at least)
"""
import timeit, json

from cheb2d_Basu import *
import matplotlib.pyplot as plt
import pandas
import scipy.optimize
import ChebTools

x_ = 0.7
y_ = 0.7

# f = lambda x, y: np.sin(x)*np.cos(-0.01*y**2)

path = r'C:\Users\ihb\Code\REFPROP-interop\teqp\dev\mixtures\mixture_departure_functions.json'

for dep in json.load(open(path)):
    if dep['Name'] == 'KWS':
        print(dep)
        n = dep['n']
        t = dep['t']
        d = dep['d']
        l = dep['l']

def to_real_world(x, xmin, xmax):
    return ((xmax - xmin) * x + (xmax + xmin))*0.5

taumin, taumax = (1e-6, 5)
deltamin, deltamax = (1e-6, 3)

def func(tau,delta):
    return sum([n[i]*delta**d[i]*tau**t[i]*np.exp(-np.sign(l[i])*delta**l[i]) for i in range(len(n))])

def f(xhat, yhat):
    tau = to_real_world(xhat, taumin, taumax)
    delta = to_real_world(yhat, deltamin, deltamax)
    return func(tau, delta)

from scipy.stats import qmc
sampler = qmc.Sobol(d=2, scramble=False)
sample = sampler.random_base2(m=6)
samples = qmc.scale(sample, [-1,-1], [1,1])
samplesrw = [to_real_world(samples[:,0], taumin, taumax), to_real_world(samples[:,1], deltamin, deltamax)]
print(len(samplesrw[0]))

def plot_fitting(M0=4):
    o = []
    for M in [M0,6,8,10,12]:
        print(M)
        a = get_coeff_mat(f, m=M, n=M)
        print(a.T)
        return

        valsnaive = eval_naive(amat=a, x=samples[:,0], y=samples[:,1])
        valsfunc = func(samplesrw[0], samplesrw[1])
        errors = valsnaive-valsfunc
        print(np.mean(np.abs(errors))/(np.max(valsnaive) - np.min(valsnaive)))

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

        # print(
        #     eval_Clenshaw(amat=a, x=x_, y=y_), 
        #     eval_naive(amat=a, x=x_, y=y_), 
        #     f(x_, y_)
        # )
        o.append({
            'M': M,
            'naive': time_naive,
            'Clenshaw': time_Clenshaw,
            'mean fractional error': np.mean(np.abs(errors))/(np.max(valsnaive) - np.min(valsnaive))
            })

    df = pandas.DataFrame(o)
    plt.plot(df['M'], df['naive'], 'o-', label='naive')
    plt.plot(df['M'], df['Clenshaw'], 'o-', label='Clenshaw')
    plt.yscale('log')
    plt.gca().set(xlabel='$M$, of $M^2$ matrix', ylabel='time / $\mu$s/call')
    plt.legend()
    plt.show()

    plt.plot(df['M'], df['mean fractional error'], 'o-', label='naive')
    plt.yscale('log')
    plt.gca().set(xlabel='$M$, of $M^2$ matrix', ylabel='mean fractional error')
    plt.legend()
    plt.show()

def inverse_fitting(N):
    
    exact = func(*samplesrw)
    denom = np.max(exact) - np.min(exact)

    def objective(params):
        amat = np.array(params).reshape(N+1,N+1)
        vals = [ChebTools.Clenshaw2DEigencomplex(amat, x_, y_) for x_,y_ in zip(samples[:,0],samples[:,1])]
        error = np.sum(((vals - exact)/denom)**2)
        return error
        
    # res = scipy.optimize.minimize(objective, [0.0]*((N+1)*(N+1)), method='CG', options=dict(disp=True, maxfev=1e8))
    res = scipy.optimize.minimize(objective, [0.0]*((N+1)*(N+1)), jac='cs', options=dict(disp=True))
    print(res.x.reshape(N+1, N+1))
    
    # res = scipy.optimize.differential_evolution(objective, [(-10,10)]*((N+1)*(N+1)), disp=True)
    # print(res)

if __name__ == '__main__':
    N = 8
    plot_fitting(N)
    inverse_fitting(N)
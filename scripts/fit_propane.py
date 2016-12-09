from __future__ import print_function
import sys, time

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../build/pybind11/Release')
import ChebTools as CT

import json
EOS = json.loads(open(r'n-Propane.json','r').read())['EOS'][0]['alphar']

def check(ce,f,deltamax,deltamin = 0):
    data = []
    for delta in np.linspace(deltamin, deltamax, 200):
        data.append((delta, ce.y(delta), f(delta)))
    xx,y1,y2 = zip(*data)

    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(6, 6))
    ax1.plot(xx,np.array(y2),lw=2, label='Given')
    ax1.plot(xx,np.array(y1),dashes=[2,2], label='Chebyshev')
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('$y$')
    ax1.legend()

    ax2.plot(xx,(np.array(y2)-np.array(y1)),lw=2, label='Given')
    ax2.set_xlabel('$x$')
    lbl = ax2.set_ylabel('$\Delta y$')
    fig.tight_layout(pad=0.3)

for _l, _d in zip(EOS[0]['l'], EOS[0]['d']):
    if _l == 0: 
        _c = 0
    else:
        _c = 1
    deltamin = 1e-6
    deltamax = 6
    func = lambda delta: delta**_d*np.exp(-delta**_l)*(-_c*_l*delta**_l + _d)

    Npower = 10
    while Npower < 100:
        ce = CT.generate_Chebyshev_expansion(Npower, func, deltamin, deltamax)
        c = ce.coef()
        coeffratio = (c[-2::]**2).sum()/(c[0:2]**2).sum()
        if coeffratio > 1e-4:
            Npower *= 2
        else:
            break
    print(Npower, _l, _d, coeffratio)
    check(ce,func,deltamin,deltamax)
    
Npower = 40
for _eta, _epsilon,_d in zip(EOS[1]['eta'], EOS[1]['epsilon'], EOS[1]['d']):
    deltamin = 1e-10
    deltamax = 6
    func = lambda delta: delta**_d*np.exp(-_eta*(delta-_epsilon)**2)*(-2*_eta*delta*(delta-_epsilon) + _d)
    Npower = 10
    while Npower < 100:
        ce = CT.generate_Chebyshev_expansion(Npower, func, deltamin, deltamax)
        c = ce.coef()
        coeffratio = (c[-2::]**2).sum()/(c[0:2]**2).sum()
        if coeffratio > 1e-4:
            Npower = int(Npower*1.25)
        else:
            break

    print(Npower, _eta, _epsilon, _d, coeffratio)
    check(ce,func,deltamax,deltamin=deltamin)

plt.close('all')
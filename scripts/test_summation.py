from __future__ import print_function
import sys, time

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../build/pybind11/Debug')
import ChebTools as CT

import CoolProp, CoolProp.CoolProp as CP, json
T = 340
fluid = 'n-Propane'
Tc = CP.PropsSI("Tcrit",fluid)

Ndelta = 100
Ntau = Ndelta
deltamin = 0.00000001
deltamax = 6

def coeffs_from_CoolProp():
    AS = CoolProp.CoolProp.AbstractState('HEOS',fluid)
    AS.specify_phase(CP.iphase_gas) # Something homogeneous
    def f(delta):
        AS.update(CP.DmolarT_INPUTS, delta*AS.rhomolar_reducing(), T)
        return AS.alphar()

    ce = CT.generate_Chebyshev_expansion(Ndelta, f, deltamin, deltamax)
    return ce.coef()

def coeffs_from_summation(plot = False):
    EOS = json.loads(open(fluid+'.json','r').read())['EOS'][0]['alphar']
    n = EOS[0]['n'] + EOS[1]['n']
    t = EOS[0]['t'] + EOS[1]['t']
    d = EOS[0]['d'] + EOS[1]['d']
    
    cdelta = [int(EOS[0]['l'][i]>0) for i in range(len(EOS[0]['n']))] + [1]*6 + [0]*len(EOS[1]['n'])
    ldelta = EOS[0]['l'] + [0]*len(EOS[1]['n'])

    eta = [0]*len(EOS[0]['l']) + EOS[1]['eta']
    epsilon = [0]*len(EOS[0]['l']) + EOS[1]['epsilon']
    beta = [0]*len(EOS[0]['l']) + EOS[1]['beta']
    gamma = [0]*len(EOS[0]['l']) + EOS[1]['gamma']

    terms = []
    for i in range(len(n)):
        funcF = lambda tau: tau**t[i]*np.exp(-beta[i]*(tau-gamma[i])**2)
        F = CT.generate_Chebyshev_expansion(Ntau, funcF, deltamin, deltamax)
        funcG = lambda delta: delta**d[i]*np.exp(-cdelta[i]*delta**ldelta[i]-eta[i]*(delta-epsilon[i])**2)
        G = CT.generate_Chebyshev_expansion(Ndelta, funcG, deltamin, deltamax)

        st = CT.SumElement(n[i], F, G)
        terms.append(st)

        if plot:
            tau = np.linspace(0.1, 6, 10000)
            plt.plot(tau, funcF(tau), tau, F.y(tau))

    cs = CT.ChebyshevSummation(terms)
    if plot:
        plt.yscale('symlog')
        plt.show()
    return cs.get_coefficients(Tc/T)

if __name__=='__main__':
    c_CP = coeffs_from_CoolProp()
    c_sum = coeffs_from_summation()
    ce_CP = CT.ChebyshevExpansion(c_CP, deltamin, deltamax)
    ce_sum = CT.ChebyshevExpansion(c_sum, deltamin, deltamax)

    def get_roots(ce):
        return [((deltamax-deltamin)*np.real(rt) + (deltamax + deltamin))/2.0 for rt in np.linalg.eigvals(ce.companion_matrix()) if np.isreal(rt) and rt > -1 and rt < 1]

    print(get_roots(ce_CP))
    print(get_roots(ce_sum))

    delta = np.linspace(deltamin, deltamax, 10000)
    plt.plot(delta, ce_CP.y(delta))
    plt.plot(delta, ce_sum.y(delta))
    plt.show()

    AS = CoolProp.CoolProp.AbstractState('HEOS',fluid)
    AS.specify_phase(CP.iphase_gas) # Something homogeneous
    def f(delta):
        AS.update(CP.DmolarT_INPUTS, delta*AS.rhomolar_reducing(), T)
        return AS.alphar()
    ff = np.array([f(_) for _ in delta])

    _range = np.max(ff) - np.min(ff)
    plt.plot(delta, (ce_CP.y(delta)-ff)/_range, label='Direct fit from CoolProp')
    plt.plot(delta, (ce_sum.y(delta)-ff)/_range, label='Summation')
    plt.legend(loc='best')
    plt.show()
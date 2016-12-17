from __future__ import print_function
import sys, time

import numpy as np, math
import matplotlib.pyplot as plt

sys.path.append('../build/pybind11/Debug')
import ChebTools as CT

import CoolProp, CoolProp.CoolProp as CP, json
T = 600
fluid = 'n-Propane'
Tc = CP.PropsSI("Tcrit",fluid)

Ndelta = 12
Ntau = 400
deltamin = 1.5
deltamax = 6

def check_deriv():
    Ndelta = 6
    f = lambda x: np.exp(x)
    ce = CT.generate_Chebyshev_expansion(Ndelta, f, 1, 6)
    print(ce.coef())
    print(ce.deriv(1).coef())

AS = CoolProp.CoolProp.AbstractState('HEOS',fluid)
AS.specify_phase(CP.iphase_gas) # Something homogeneous
def f(delta):
    AS.update(CP.DmolarT_INPUTS, delta*AS.rhomolar_reducing(), T)
    return AS.alphar()
def fprime(delta):
    AS.update(CP.DmolarT_INPUTS, delta*AS.rhomolar_reducing(), T)
    return AS.dalphar_dDelta()

def coeffs_from_CoolProp():
    ce = CT.generate_Chebyshev_expansion(Ndelta, fprime, deltamin, deltamax)
    return ce.coef()

def get_EOS(fluid):
    EOS = json.loads(open(fluid+'.json','r').read())['EOS'][0]['alphar']
    n = EOS[0]['n'] + EOS[1]['n']
    t = EOS[0]['t'] + EOS[1]['t']
    d = EOS[0]['d'] + EOS[1]['d']
    
    N0 = len(EOS[0]['n'])
    N1 = len(EOS[1]['n'])
    cdelta = [int(EOS[0]['l'][i]>0) for i in range(N0)] + [0]*N1
    ldelta = EOS[0]['l'] + [0]*N1

    eta = [0]*N0 + EOS[1]['eta']
    epsilon = [0]*N0 + EOS[1]['epsilon']
    beta = [0]*N0 + EOS[1]['beta']
    gamma = [0]*N0 + EOS[1]['gamma']
    class struct: pass
    del N0, N1, i, fluid, EOS
    g = locals(); s = struct(); s.__dict__.update(g)
    for k in ['struct']:
        s.__dict__.pop(k)
    return s

def coeffs_from_summation(plot = True):
    
    E = get_EOS(fluid)

    terms = []
    if plot:
        fig, (ax1,ax2) = plt.subplots(1,2)
    nF = np.zeros((len(E.n),))
    for i in range(len(E.n)):
        funcF = lambda tau: tau**E.t[i]*np.exp(-E.beta[i]*(tau-E.gamma[i])**2)
        F = CT.generate_Chebyshev_expansion(Ntau, funcF, 0.00001, 6)
        funcG = lambda delta: delta**E.d[i]*np.exp(-E.cdelta[i]*delta**E.ldelta[i]-E.eta[i]*(delta-E.epsilon[i])**2)
        G = CT.generate_Chebyshev_expansion(Ndelta+1, funcG, deltamin, deltamax)

        st = CT.SumElement(E.n[i], F, G.deriv(1))
        terms.append(st)

        nF[i] = E.n[i]*funcF(Tc/T)

        _delta = np.linspace(deltamin, deltamax, 10000)
        _tau = np.linspace(0.1, 6, 10000)
        print(np.sum((funcG(_delta)-G.y(_delta))**2), np.sum((funcF(_tau)-F.y(_tau))**2))

        if plot:
            tau = np.linspace(0.1, 6, 10000)
            ax1.plot(tau, E.n[i]*funcF(tau))
            ax1.plot(tau, E.n[i]*F.y(tau), dashes = [2,2])
            _delta = np.linspace(deltamin, deltamax, 10000)
            ax2.plot(_delta, funcG(_delta))
            ax2.plot(_delta, G.deriv(1).y(_delta), dashes = [2,2])

    cs = CT.ChebyshevSummation(terms)

    cs.build_independent_matrix()
    C = cs.get_matrix()

    PP = C * nF[:, np.newaxis]
    c0 = np.sum(PP, axis = 0)
    c2 = cs.get_coefficients(Tc/T)

    if plot:
        # plt.ylim(ymin=1e-10)
        ax1.set_yscale('symlog', linthreshy = 0.01)
        ax2.set_yscale('symlog', linthreshy = 0.01)
        plt.show()
    
    return c2

if __name__=='__main__':
    check_deriv()

    c_CP = coeffs_from_CoolProp()
    c_sum = coeffs_from_summation()
    ce_CP = CT.ChebyshevExpansion(c_CP, deltamin, deltamax)
    ce_sum = CT.ChebyshevExpansion(c_sum, deltamin, deltamax)

    def get_roots(ce):
        return [((deltamax-deltamin)*np.real(rt) + (deltamax + deltamin))/2.0 for rt in np.linalg.eigvals(ce.companion_matrix()) if np.isreal(rt) and rt > -1 and rt < 1]

    print(get_roots(ce_CP))
    print(get_roots(ce_sum))

    plt.plot(c_CP - c_sum)
    plt.ylabel('diff in coeff')
    plt.show()

    delta = np.linspace(deltamin, deltamax, 10000)
    plt.plot(delta, ce_CP.y(delta))
    plt.plot(delta, ce_sum.y(delta))
    plt.show()

    # AS = CoolProp.CoolProp.AbstractState('HEOS',fluid)
    # AS.specify_phase(CP.iphase_gas) # Something homogeneous
    # def f(delta):
    #     AS.update(CP.DmolarT_INPUTS, delta*AS.rhomolar_reducing(), T)
    #     return AS.alphar()
    # ff = np.array([f(_) for _ in delta])
    # _range = np.max(ff) - np.min(ff)
    # plt.plot(delta, (ce_CP.y(delta)-ff)/_range, label='Direct fit from CoolProp')
    # plt.plot(delta, (ce_sum.y(delta)-ff)/_range, label='Summation')
    # plt.legend(loc='best')
    # plt.show()
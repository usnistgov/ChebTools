from __future__ import print_function
import sys, time

import numpy as np, math
import matplotlib.pyplot as plt

sys.path.append('../build/pybind11/Release')
import ChebTools as CT

import CoolProp, CoolProp.CoolProp as CP, json

T = 200

Ndelta = 40
Ntau = 40
deltamin = 1e-12
deltamax = 6

one = CT.ChebyshevExpansion([1], deltamin, deltamax)
_delta = CT.generate_Chebyshev_expansion(1, lambda delta: delta, deltamin, deltamax)

def check_deriv():
    Ndelta = 6
    f = lambda x: np.exp(x)
    ce = CT.generate_Chebyshev_expansion(Ndelta, f, -1, 1)
    print(ce.coef())
    print(ce.deriv(1).coef())

def coeffs_from_CoolProp():
    AS = CoolProp.CoolProp.AbstractState('HEOS',fluid)
    AS.specify_phase(CP.iphase_gas) # Something homogeneous
    rhomolar_reducing = AS.rhomolar_reducing()
    R = AS.gas_constant()

    def f(delta):
        AS.update(CP.DmolarT_INPUTS, delta*AS.rhomolar_reducing(), T)
        return AS.alphar()
    def fprime(delta):
        AS.update(CP.DmolarT_INPUTS, delta*AS.rhomolar_reducing(), T)
        return AS.dalphar_dDelta()
    ce = CT.generate_Chebyshev_expansion(Ndelta, fprime, deltamin, deltamax)
    return ce.coef()

def get_EOS(fluid):
    JSON_string = CP.get_fluid_param_string(fluid, "JSON")
    js = json.loads(JSON_string)[0]
    EOS = js['EOS'][0]['alphar']
    if len(EOS) == 2:
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
        del N0, N1
    elif len(EOS) == 1:
        n = EOS[0]['n']
        t = EOS[0]['t']
        d = EOS[0]['d']
        
        N0 = len(EOS[0]['n'])
        cdelta = [int(EOS[0]['l'][i]>0) for i in range(N0)]
        ldelta = EOS[0]['l']

        eta = [0]*N0
        epsilon = [0]*N0
        beta = [0]*N0
        gamma = [0]*N0
        del N0
    else:
        ValueError()

    class struct: pass
    del i, fluid, EOS
    g = locals(); s = struct(); s.__dict__.update(g)
    for k in ['struct']:
        s.__dict__.pop(k)
    return s

def get_terms_in_summation(E, plot = False):
    terms = []
    nF = np.zeros((len(E.n),1))
    for i in range(len(E.n)):
        funcF = lambda tau: tau**E.t[i]*np.exp(-E.beta[i]*(tau-E.gamma[i])**2)
        F = CT.generate_Chebyshev_expansion(Ntau, funcF, 0.4, 3)
        funcG = lambda delta: delta**E.d[i]*np.exp(-E.cdelta[i]*delta**E.ldelta[i]-E.eta[i]*(delta-E.epsilon[i])**2)
        G = CT.generate_Chebyshev_expansion(Ndelta+1, funcG, deltamin, deltamax)

        st = CT.SumElement(E.n[i], F, G.deriv(1))
        terms.append(st)

        Tc = 1000000000
        nF[i] = E.n[i]*funcF(Tc/T)

        # _delta = np.linspace(deltamin, deltamax, 10000)
        # _tau = np.linspace(0.1, 6, 10000)
        # print(np.sum((funcG(_delta)-G.y(_delta))**2), np.sum((funcF(_tau)-F.y(_tau))**2))

        if plot:
            tau = np.linspace(0.1, 6, 10000)
            ax1.plot(tau, E.n[i]*funcF(tau))
            ax1.plot(tau, E.n[i]*F.y(tau), dashes = [2,2])
            _delta = np.linspace(deltamin, deltamax, 10000)
            ax2.plot(_delta, funcG(_delta))
            ax2.plot(_delta, G.deriv(1).y(_delta), dashes = [2,2])

    return terms, nF

def summatrix_for_fluid(fluid, plot = False, check_against_exact = False):
    
    E = get_EOS(fluid)

    if plot:
        fig, (ax1,ax2) = plt.subplots(1,2)

    terms, nF = get_terms_in_summation(E, plot)

    cs = CT.ChebyshevSummation(terms)
    cs.build_independent_matrix()
    if not check_against_exact:
        return cs
    else:
        
        C = cs.get_matrix()
        c0 = (C.T.dot(nF)).T.squeeze()

        tic = time.clock()
        c2 = cs.get_coefficients(Tc/T)
        toc = time.clock()

        if plot:
            # plt.ylim(ymin=1e-10)
            ax1.set_yscale('symlog', linthreshy = 0.01)
            ax2.set_yscale('symlog', linthreshy = 0.01)
            plt.show()
        
        return c0

def build_summation_structure(fluids):
    N = 0
    summats = []
    for fluid in fluids:
        # Construct the summation matrix for each fluid individually
        summat = summatrix_for_fluid(fluid)
        # Store it
        summats.append(summat)
        # Get the matrix associated with this term
        C = summat.get_matrix()
        # Determine whether the maximum length needs to be modified
        N = max(N, C.shape[1])
    return summats, N

def mixture_expansion_of_p(sumstruct, AS, T, N):
    z = AS.get_mole_fractions()
    Tr = AS.T_reducing()
    tau = Tr/T

    # Ncomp = len(z)
    # A = np.zeros((N, Ncomp))
    
    # for i, summat in enumerate(sumstruct):
    #     c = summat.get_coefficients(tau).squeeze()
    #     A[0:len(c)+1, i] = c
    # dalphar_dDelta = A.dot(z)

    # Method #2, with C++ class
    cm = CT.ChebyshevMixture(sumstruct, N-1)
    dalphar_dDelta = cm.get_expansion(Tr/T, z).coef()
    # return

    p = (CT.ChebyshevExpansion(dalphar_dDelta, deltamin, deltamax)*_delta + one)*(AS.rhomolar_reducing()*T*AS.gas_constant()*_delta)
    return p

if __name__=='__main__':
    
    p_target = 0
    neg_p_target = -p_target*one

    fluids = ['Methane','HydrogenSulfide']
    sumstruct, N = build_summation_structure(fluids)
    AS = CoolProp.CoolProp.AbstractState('HEOS','&'.join(fluids))
    AS.specify_phase(CP.iphase_gas) # Something homogeneous
    z0 = 0.5
    AS.set_mole_fractions([z0,1-z0])
    
    tic = time.clock()
    p_mix = mixture_expansion_of_p(sumstruct, AS, T, N)
    toc = time.clock()
    print((toc - tic)*1e6,'us elapsed')
    if p_mix is None:
        sys.exit(-1)
        
    p_sum = p_mix + neg_p_target

    def get_p(delta, AS, T):
        y = []
        for _d in delta:
            AS.update(CP.DmolarT_INPUTS, _d*AS.rhomolar_reducing(), T)
            p = (AS.dalphar_dDelta()*_d + 1.0)*AS.rhomolar_reducing()*_d*AS.gas_constant()*AS.T()
            # p = AS.p()
            y.append(p)
        return np.array(y)

    def get_roots(ce):
        return [((deltamax-deltamin)*np.real(rt) + (deltamax + deltamin))/2.0 for rt in np.linalg.eigvals(ce.companion_matrix()) if np.isreal(rt) and rt > -1 and rt < 1]

    delta = np.linspace(deltamin, deltamax, 10000)
    plt.plot(delta, p_sum.y(delta), lw = 2)
    plt.plot(delta, get_p(delta,AS,T)-p_target)
    plt.yscale('symlog')
    plt.axhline(p_target)
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
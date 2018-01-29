
import scipy.special
import ChebTools
# The 0-th Bessel function (for code concision)
J0 = lambda x: scipy.special.jn(0,x)
# Make a 200-th order expansion of the 0-th Bessel function in [0,30]
f = ChebTools.generate_Chebyshev_expansion(200, J0, 0, 30)
# Roots of the function
rts = f.real_roots(True)
# Extrema of the function (roots of the derivative, where dy/dx =0)
extrema = f.deriv(1).real_roots(True)

import matplotlib.pyplot as plt, numpy as np, matplotlib as mpl
from cycler import cycler
from matplotlib import rcParams
rcParams['font.family'] = 'serif'

# mpl.rcParams['axes.prop_cycle'] = cycler(color= ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728','#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
# plt.style.use('classic')
x = np.linspace(0,30,1000); y = J0(x)
plt.figure(figsize=(4,3))
plt.axhline(0,color='k',dashes=[2,2])
plt.plot(x,y)
plt.plot(rts,np.zeros_like(rts),'o',label='roots')
plt.plot(extrema,J0(extrema),'*',label='extrema')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.tight_layout(pad=0.2)
plt.savefig('Bessel.png', transparent=True,dpi=600)
plt.show()
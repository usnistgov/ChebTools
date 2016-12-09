from __future__ import print_function
import sys, time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('../build/pybind11/Release')
import ChebTools as CT

f = lambda x: np.exp(-5*(x-1)**2)-0.5

def fit(xbreaks, full_output = False):
    errors,output = [],[]
    for i in range(len(xbreaks)-1):
        xmin = xbreaks[i]
        xmax = xbreaks[i+1]
        ce = CT.generate_Chebyshev_expansion(10, f, xmin, xmax)

        xx = np.linspace(xmin, xmax, 1000)
        y_func = np.array([f(_x) for _x in xx])
        y_fit = ce.y(xx)
        worst_error = np.max(np.abs(y_func-y_fit))
        errors.append(worst_error)
        if full_output:
            output.append(dict(errors = errors,
                    y_fit = y_fit,
                    y_func = y_func,
                    xx = xx))

    if not full_output:
        return errors
    else:
        return output

xmin = 0
xmax = 6

xbreaks = [xmin, xmax]

done = False
err_tol = 1e-5
while not done:
    # Fit!
    errors = fit(xbreaks)

    # Work backwards through the errors so that as we insert we don't mess up the indices
    all_ok = True
    for i in reversed(range(len(xbreaks)-1)):
        if errors[i] > err_tol:
            xbreaks.insert(i+1, (xbreaks[i] + xbreaks[i+1])/2.0)
            all_ok = False
    
    print(xbreaks, errors)

    if all_ok: 
        break

fig, (ax1, ax2) = plt.subplots(2,1)
res = fit(xbreaks, full_output = True)
for el in res:
    ax1.plot(el['xx'], el['y_func'])
    ax1.plot(el['xx'], el['y_fit'])
    ax2.plot(el['xx'], el['y_func']-el['y_fit'])
plt.show()
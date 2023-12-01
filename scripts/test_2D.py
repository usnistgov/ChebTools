import timeit

import numpy as np
import ChebTools
import matplotlib.pyplot as plt
import pandas

# https://www.chebfun.org/examples/approx2/PrettyFunctions.html
f = lambda x, y: np.cos(10*(x*x+y))*np.sin(10*(x+y*y))

# o = []
# for N in range(1,20):
#     tic = timeit.default_timer()
#     b = ChebTools.ChebyshevExpansion2DBounds()
#     ce = ChebTools.generate_Chebyshev_expansion2D(f, [N, N], b)
#     toc = timeit.default_timer()
#     elap = toc-tic
#     o.append({'N': N, 'elap': elap})
# df = pandas.DataFrame(o)
# plt.plot(df.N, df.elap)
# plt.yscale('log')
# plt.show()

b = ChebTools.ChebyshevExpansion2DBounds()

for w in [1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64]:

    b.xmin = -w
    b.xmax = w
    b.ymin = -w
    b.ymax = w

    ce = ChebTools.generate_Chebyshev_expansion2D(f, [8, 8], b)

    o = []
    for x_ in np.linspace(-w, w, 100):
        for y_ in np.linspace(-w, w, 100):
            val = f(x_, y_)
            err = ce.eval_Clenshaw(x_, y_) - val
            o.append({'x': x_, 'y': y_, 'val': val, 'err': err})
    df = pandas.DataFrame(o)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    sc = ax1.scatter(df.x, df.y, c=df.val, marker='s', s=4)
    plt.sca(ax1)
    cb = plt.colorbar(sc)
    cb.set_label('val')

    sc = ax2.scatter(df.x, df.y, c=df.err, s=1)
    plt.sca(ax2)
    cb = plt.colorbar(sc)
    cb.set_label('error')

    plt.tight_layout(pad=0.2)
    plt.show()
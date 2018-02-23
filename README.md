# ChebTools

Chebyshev-basis expansions, and more broadly, orthogonal polynomial expansions, are commonly used as numerical approximations of continuous functions on closed domains.   One of the most successful projects that makes use of the Chebyshev expansions is the ``chebfun`` library for MATLAB.  Other similar libraries are [pychebfun](https://github.com/pychebfun), [chebpy](https://github.com/chebpy/chebpy), and [Approxfun](https://github.om/JuliaApproximation/ApproxFun.jl). Our library ``ChebTools`` fills a similar niche as that of ``chebfun`` -- working with Chebyshev expansions.

The primary motivation for the development of ``ChebTools`` is the need for a highly optimized and fast C++11 library for working with Chebyshev expansions.

Automatic tests status on TravisCI: [![Build Status](https://travis-ci.org/usnistgov/ChebTools.svg?branch=master)](https://travis-ci.org/usnistgov/ChebTools)

Paper about ChebTools in JOSS: [![DOI](http://joss.theoj.org/papers/10.21105/joss.00569/status.svg)](https://doi.org/10.21105/joss.00569)

## Example:

Try it in your browser: [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/usnistgov/ChebTools/master)

Suppose we wanted to calculate the roots and extrema of the 0-th Bessel function in [0, 30].  That results in a picture like this: 

<img src="JOSS/Bessel.png" alt="Roots and extrema of the 0-th Bessel function" style="width: 200px;"/>

For which the Python code would read
``` python
import scipy.special
import ChebTools
# Only keep the roots that are in [-1,1] in scaled coordinates
only_in_domain = True
# The 0-th Bessel function (for code concision)
def J0(x): return scipy.special.jn(0,x)
# Make a 200-th order expansion of the 0-th Bessel function in [0,30]
f = ChebTools.generate_Chebyshev_expansion(200, J0, 0, 30)
# Roots of the function
rts = f.real_roots(only_in_domain)
# Extrema of the function (roots of the derivative, where dy/dx =0)
extrema = f.deriv(1).real_roots(only_in_domain)
```

## License

*MIT licensed (see LICENSE for specifics), not subject to copyright in the USA.

## Contributing

If you would like to contribute to ``ChebTools``, please open a pull request or submit an issue if you run into troubles.  Especially welcome would be additional tests.

## Installation

### Prerequisites

You will need:

* cmake (on windows, install from cmake, on linux ``sudo apt install cmake`` should do it, on OSX, ``brew install cmake``)
* Python (the anaconda distribution is used by the authors)
* a compiler (on windows, Visual Studio 2015+ (express version is fine), g++ on linux/OSX)

If on linux you use Anaconda and end up with an error like
```
ImportError: /home/theuser/anaconda3/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.20' not found (required by /home/theuser/anaconda3/lib/python3.5/site-packages/ChebTools.cpython-35m-x86_64-linux-gnu.so)
```
it can be sometimes fixed by installing ``libgcc`` with conda: ``conda install libgcc``.  [This is due to an issue in Anaconda](https://github.com/ContinuumIO/anaconda-issues/issues/483)

## To install in one line from github (easiest)

This will download the sources into a temporary directory and build and install the python extension so long as you have the necessary prerequisites:
```
pip install git+git://github.com/usnistgov/ChebTools.git
```

### From a cloned repository

Alternatively, you can clone (recursively!) and run the ``setup.py`` script

```
git clone --recursive https://github.com/usnistgov/ChebTools
cd ChebTools
python setup.py install
```

to install, or 

```
python setup.py develop
```

to use a locally-compiled version for testing.  If you want to build a debug version, you can do so with

```
python setup.py build -g develop
```
With a debug build, you can step into the debugger to debug the C++ code, for instance.  

### Cmake build

Starting in the root of the repo (a debug build with the default compiler, here on linux):

``` 
git clone --recursive https://github.com/usnistgov/ChebTools
cd ChebTools
mkdir build
cd build
cmake ..
cmake --build .
```
For those using Anaconda on Linux, please use the following for cmake:
```
mkdir build
cd build
cmake .. -DPYTHON_EXECUTABLE=`which python`
cmake --build .
```
For Visual Studio 2015 (64-bit) in release mode, you would do:
``` 
git clone --recursive https://github.com/usnistgov/ChebTools
cd ChebTools
mkdir build
cd build
cmake .. -G "Visual Studio 14 2015 Win64"
cmake --build . --config Release
```

If you need to update your submodules (pybind11 and friends)

```
git submodule update --init
```

For other options, see the cmake docs

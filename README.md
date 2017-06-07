# ChebTools
C++ tools for working with Chebyshev expansion interpolants

[![Build Status](https://travis-ci.org/usnistgov/ChebTools.svg?branch=master)](https://travis-ci.org/usnistgov/ChebTools)

## To install in one line from github

```
pip install git+git://github.com/usnistgov/ChebTools.git
```

Alternatively, you can clone and run

```
python setup.py install
```

to install, or 


```
python setup.py develop
```

to use a locally-compiled version for testing.  If you want to build a debug version, do can do so with

```
python setup.py build -g develop
```

## Cmake build

Starting in the root of the repo (a debug build with the default compiler, here on linux):

``` 
git clone --recursive https://github.com/usnistgov/ChebTools
cd ChebTools
mkdir build
cd build
cmake ..
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
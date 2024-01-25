# Based on https://github.com/pybind/cmake_example

import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

VERSION = '1.11'
with open('src/ChebToolsVersion.hpp','w') as fp:
    fp.write(f'#include <string>\nconst std::string CHEBTOOLSVERSION = "{VERSION}";')

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

os.environ['MACOSX_DEPLOYMENT_TARGET'] = '14.0'

class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        cmake_args.append('-DCHEBTOOLS_NO_MONOLITH=ON')
        cmake_args.append('-DCHEBTOOLS_NO_CATCH=ON')
        cmake_args.append('-DCHEBTOOLS_NANOBIND=ON')
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.', '--target','ChebTools'] + build_args, cwd=self.build_temp)

setup(
    name='ChebTools',
    version=VERSION,
    author='Ian Bell',
    author_email='ian.bell@nist.gov',
    description='Tools for working with Chebyshev expansion',
    long_description='',
    ext_modules=[CMakeExtension('ChebTools')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)

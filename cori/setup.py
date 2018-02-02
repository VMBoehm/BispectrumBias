# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 09:15:58 2016

@author: traveller
"""

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy
import Cython
#from Cython.Compiler.Options import directive_defaults
directive_defaults=Cython.Compiler.Options.directive_defaults
directive_defaults['linetrace'] = True
directive_defaults['binding'] = True

extensions = [
    Extension("Tools", ["Tools.pyx"])]#,
#    libraries=cython_gsl.get_libraries(),
#    library_dirs=[cython_gsl.get_library_dir()],
#    include_dirs=[cython_gsl.get_cython_include_dir()])]

#extensions = [
#    Extension("TypeA", ["TypeAdom.pyx"],
#    libraries=cython_gsl.get_libraries(),
#    library_dirs=[cython_gsl.get_library_dir()],
#    include_dirs=[cython_gsl.get_cython_include_dir()])]    
#,
#    Extension("TypeA", ["TypeAdom.pyx"])]
    
#setup(
#  ext_modules = cythonize(extensions, include_path = [numpy.get_include(), cython_gsl.get_include()],)
#)


setup(
  name = 'Tools',
  ext_modules = cythonize(extensions),
  include_dirs=[numpy.get_include()]#, cython_gsl.get_include()
)

setup(
  name = 'TypeA',
  ext_modules = cythonize("TotBias.pyx"),
  include_dirs=[numpy.get_include()]
)

from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext

GSL_INCLUDE_DIR = r'C:\Users\anowack\Desktop\GSLBUILD'
GSL_LIBRARY_DIR = r'C:\Users\anowack\Desktop\GSLBUILD\Debug'
GSL_BINARY_DIR = r'C:\Users\anowack\Desktop\GSLBUILD\bin\Debug'

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("tomo",
                 sources=["tomo.pyx", "_tomo.c"],
                 include_dirs=[numpy.get_include(), GSL_INCLUDE_DIR],
                 library_dirs=[GSL_LIBRARY_DIR, GSL_BINARY_DIR],
                 libraries=["gsl", "gslcblas"])],
)
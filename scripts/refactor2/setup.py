from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension("tomo",
                  sources=["tomo.pyx", "_tomo.c"],
                  include_dirs=[
                      numpy.get_include()],
                  extra_compile_args=['/openmp'],
                  extra_link_args=['/openmp']
                  )],
)

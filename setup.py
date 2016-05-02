try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(
    name='PyTracer',
    version='0.1dev',
    packages=['pytracer'],
    ext_modules=cythonize(["pytracer/*.pyx"]),
    include_dirs=[numpy.get_include()]
)

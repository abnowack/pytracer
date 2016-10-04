try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Build import cythonize
import numpy

setup(
    name='pytracer',
    version='0.1dev',
    package_dir={'pytracer': 'pytracer'},
    packages=['pytracer'],
    ext_modules=cythonize(['pytracer/transmission_c.pyx', 'pytracer/geometry_c.pyx',
                           'pytracer/fission_c.pyx']),
    include_dirs=[numpy.get_include()]
)

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

import numpy

intersect_module = Extension('pytracer.intersect_module',
                             sources=['pytracer/_fast_intersect_c.c'],
                             include_dirs=[numpy.get_include()])

setup(
    name='PyTracer',
    version='0.1dev',
    package_dir={'pytracer': 'pytracer'},
    packages=['pytracer'],
    ext_modules=[intersect_module]
)

from setuptools import find_packages
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension('aurora.nn.im2col', ['aurora/nn/im2col.pyx'],
              include_dirs=[numpy.get_include()]
              ),
    Extension('aurora.nn.fast_pooling', ['aurora/nn/fast_pooling.pyx'],
              include_dirs=[numpy.get_include()]
              ),
]

setup(
    name='aurora',
    version='0.01',
    description='Minimal Deep Learning library is written in Python/Numpy and a bit of C++',
    url='https://github.com/upul/Aurora',
    author='Upul Bandara',
    author_email='upulbandara@gmail.com',
    license='MIT',
    ext_modules=cythonize(extensions),
    packages=find_packages(exclude=['Aurora.tests'])

)

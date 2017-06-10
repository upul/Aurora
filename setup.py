from setuptools import setup, find_packages

setup(
    name='aurora',
    version='0.01',
    description='Minimal Deep Learning library is written in Python/Numpy and a bit of C++',
    url='https://github.com/upul/Aurora',
    author='Upul Bandara',
    author_email='upulbandara@gmail.com',
    license='MIT',
    packages=find_packages(exclude=['Aurora.tests'])
)
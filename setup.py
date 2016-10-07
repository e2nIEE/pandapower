# Copyright (c) 2010-2015 Richard Lincoln. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

from setuptools import setup, find_packages


setup(
    name='pandapower',
    version='1.0.0',
    author='Leon Thurner, Alexander Scheidler',
    author_email='leon.thurner@uni-kassel.de, alexander.scheidler@iwes.fraunhofer.de',
    description='Convenient Power System Modelling and Analysis based on PYPOWER and pandas',
#    long_description='\n\n'.join(
#        open(f, 'rb').read().decode('utf-8')
#        for f in ['README.rst']),
    url='https://github.com/IWESUniKS/pandapower',
    license='BSD',
    install_requires=[
        # Deactivated to avoid problems with system packages.
        # Manual installation of the following packages required:
        # - numpy
        # - scipy
        # - pandas
        # - pypower
    ],
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering',
    ],
)

# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.


from setuptools import setup, find_packages

setup(
    name='pandapower',
    version='1.0.1',
    author='Leon Thurner, Alexander Scheidler',
    author_email='leon.thurner@uni-kassel.de, alexander.scheidler@iwes.fraunhofer.de',
    description='Convenient Power System Modelling and Analysis based on PYPOWER and pandas',
    url='www.uni-kassel.de/go/pandapower',
    license='BSD',
    install_requires=['pypower>=5.0.1'],# dependencies for basic scientific packages
                                        # (numpy, scipy, pandas etc.) are in a seperate
                                        # requirements.txt file
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
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering',
    ],
)

# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from setuptools import setup, find_packages
import re

with open('README.rst', 'rb') as f:
    install = f.read().decode('utf-8')

with open('CHANGELOG.rst', 'rb') as f:
    changelog = f.read().decode('utf-8')

classifiers = [
    'Development Status :: 5 - Production/Stable',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Natural Language :: English',
    'Operating System :: OS Independent',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3']

with open('.travis.yml', 'rb') as f:
    lines = f.read().decode('utf-8')
    for version in re.findall('python: 3.[0-9]', lines):
        classifiers.append('Programming Language :: Python :: 3.%s' % version[-1])

long_description = '\n\n'.join((install, changelog))

setup(
    name='pandapower',
    version='2.4.0',
    author='Leon Thurner, Alexander Scheidler',
    author_email='leon.thurner@iee.fraunhofer.de, alexander.scheidler@iee.fraunhofer.de',
    description='An easy to use open source tool for power system modeling, analysis and optimization with a high degree of automation.',
    long_description=long_description,
	long_description_content_type='text/x-rst',
    url='http://www.pandapower.org',
    license='BSD',
    install_requires=["pandas>=0.17",
                      "networkx",
                      "scipy",
                      "numpy>=0.11",
                      "packaging",
					  "xlsxwriter",
					  "xlrd",
					  "cryptography"],
    extras_require={
		"docs": ["numpydoc", "sphinx", "sphinx_rtd_theme"],
		"plotting": ["plotly", "matplotlib", "python-igraph"],
		"test": ["pytest", "pytest-xdist"]},
    packages=find_packages(),
    include_package_data=True,
    classifiers=classifiers
)

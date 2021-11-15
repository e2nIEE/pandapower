# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
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

with open('.github/workflows/github_test_action.yml', 'rb') as f:
    lines = f.read().decode('utf-8')
    versions = set(re.findall('3.[0-9]', lines))
    for version in versions:
        classifiers.append('Programming Language :: Python :: 3.%s' % version[-1])

long_description = '\n\n'.join((install, changelog))

setup(
    name='pandapower',
    version='2.7.0',
    author='Leon Thurner, Alexander Scheidler',
    author_email='leon.thurner@iee.fraunhofer.de, alexander.scheidler@iee.fraunhofer.de',
    description='An easy to use open source tool for power system modeling, analysis and optimization with a high degree of automation.',
    long_description=long_description,
	long_description_content_type='text/x-rst',
    url='http://www.pandapower.org',
    license='BSD',
    install_requires=["pandas>=0.17",
                      "networkx>=2.5",
                      "scipy<=1.6.0",
                      "numpy>=0.11",
                      "packaging"],
    extras_require={
        "docs": ["numpydoc", "sphinx", "sphinx_rtd_theme"],
        "plotting": ["plotly", "matplotlib", "python-igraph", "geopandas", "base64", "hashlib", "zlib"],  # "shapely", "pyproj" are depedencies of geopandas and so already available
        "test": ["pytest", "pytest-xdist"],
        "performance": ["ortools"],
        "fileio": ["xlsxwriter", "openpyxl", "cryptography", "geopandas"],  # "fiona" is a depedency of geopandas and so already available
        "all": ["numpydoc", "sphinx", "sphinx_rtd_theme",
                "plotly", "matplotlib", "python-igraph", "geopandas", "zlib",
                "pytest", "pytest-xdist",
                "ortools",
                "xlsxwriter", "openpyxl", "cryptography"
                ]},  # "shapely", "pyproj", "fiona" are depedencies of geopandas and so already available
    # "hashlib", "base64" produces problems, so it is not included to "all"
    packages=find_packages(),
    include_package_data=True,
    classifiers=classifiers
)

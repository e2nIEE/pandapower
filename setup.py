# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
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
    versions = set(re.findall('3.[8-9]', lines)) | set(re.findall('3.1[0-9]', lines))
    for version in sorted(versions):
        classifiers.append('Programming Language :: Python :: %s' % version)

long_description = '\n\n'.join((install, changelog))

setup(
    name='pandapower',
    version='2.13.1',
    author='Leon Thurner, Alexander Scheidler',
    author_email='leon.thurner@retoflow.de, alexander.scheidler@iee.fraunhofer.de',
    description='An easy to use open source tool for power system modeling, analysis and optimization with a high degree of automation.',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    url='http://www.pandapower.org',
    license='BSD',
    python_requires='>=3.8',
    install_requires=["pandas>=1.0",
                      "networkx>=2.5",
                      "scipy",
                      "numpy",
                      "packaging",
                      "tqdm",
                      "deepdiff"],
    extras_require={
        "docs": ["numpydoc", "sphinx", "sphinx_rtd_theme"],
        "plotting": ["plotly", "matplotlib", "python-igraph", "geopandas"],
        # "shapely", "pyproj" are depedencies of geopandas and so already available;
        # "base64", "hashlib", "zlib" produce installing problems, so they are not included
        "test": ["pytest<=7.0", "pytest-xdist"],
        "performance": ["ortools"],  # , "lightsim2grid"],
        "fileio": ["xlsxwriter", "openpyxl", "cryptography", "geopandas", "psycopg2"],
        # "fiona" is a depedency of geopandas and so already available
        "converter": ["matpowercaseframes"],
        "all": ["numpydoc", "sphinx", "sphinx_rtd_theme",
                "plotly", "matplotlib", "python-igraph", "geopandas",
                "pytest<=7.0", "pytest-xdist",
                "ortools",  # lightsim2grid,
                "xlsxwriter", "openpyxl", "cryptography",
                "psycopg2",  # for PostgreSQL I/O
                "matpowercaseframes"
                ]},  # "shapely", "pyproj", "fiona" are depedencies of geopandas and so already available
    # "hashlib", "zlib", "base64" produce installing problems, so it is not included
    packages=find_packages(),
    include_package_data=True,
    classifiers=classifiers
)

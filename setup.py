# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

from setuptools import setup, find_packages
import re

with open('README.rst', 'rb') as f:
    install = f.read().decode('utf-8')

with open('CHANGELOG.rst', 'rb') as f:
    changelog = f.read().decode('utf-8')

# parse version from _version.py file.
VERSIONFILE = "pandapower/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

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
    for ver in sorted(versions):
        classifiers.append('Programming Language :: Python :: %s' % ver)

long_description = '\n\n'.join((install, changelog))

setup(
    name='pandapower',
    version=version,
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
        "plotting": ["plotly>=3.1.1", "matplotlib", "igraph", "geopandas", "geojson"],
        # "shapely", "pyproj" are dependencies of geopandas and so already available;
        # "base64", "hashlib", "zlib" produce installing problems, so they are not included
        "test": ["pytest~=8.1", "pytest-xdist"],
        "performance": ["ortools", "numba>=0.25", "lightsim2grid>=0.8.1"],
        "fileio": ["xlsxwriter", "openpyxl", "cryptography", "geopandas", "psycopg2"],
        # "fiona" is a depedency of geopandas and so already available
        "converter": ["matpowercaseframes"],
        "pgm": ["power-grid-model-io"],
        "all": ["numpydoc", "sphinx", "sphinx_rtd_theme",
                "plotly>=3.1.1", "matplotlib", "igraph", "geopandas", "geojson",
                "pytest~=8.1", "pytest-xdist",
                "ortools", "numba>=0.25", "lightsim2grid>=0.8.1",
                "xlsxwriter", "openpyxl", "cryptography", "psycopg2",  # for PostgreSQL I/O
                "matpowercaseframes",
                "power-grid-model-io"
                ]},  # "shapely", "pyproj", "fiona" are dependencies of geopandas and so already available
    # "hashlib", "zlib", "base64" produce installing problems, so it is not included
    packages=find_packages(),
    include_package_data=True,
    classifiers=classifiers
)

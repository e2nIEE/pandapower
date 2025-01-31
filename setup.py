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

with open('.travis.yml', 'rb') as f:
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
    install_requires=["pandas>=0.17",  # 2.2.3
                      "networkx>=2.5",  # 3.4.2
                      "scipy",  # 1.14.1
                      "numpy>=0.11",  # 2.0.0
                      "packaging",  # 24.2
                      "urllib3",  # 2.2.3
                      "certifi",  # 2024.12.14
                      "requests",  # 2.32.3
                      "joblib",  # 1.4.2
		              "xlsxwriter",  # 3.2.0
		              "xlrd",
		              "cryptography",
                      "numba",  # 0.60.0
                      "julia",  # 0.6.2
                      "lightsim2grid",  # 0.10.0
		              # "legacy-cgi",
                      "openpyxl",
                      "cryptography"],
    extras_require={
        "docs": ["numpydoc", "sphinx", "sphinx_rtd_theme"],
        "plotting": ["plotly", "matplotlib", "python-igraph"],
        "test": ["pytest", "pytest-xdist"],
        "performance": ["ortools"]},
    packages=find_packages(),
    include_package_data=True,
    classifiers=classifiers
)

pandapower combines the data analysis library `pandas <http://pandas.pydata.org>`_ and the power flow solver `PYPOWER <https://pypi.python.org/pypi/PYPOWER>`_ to create an easy to use network calculation program
aimed at automation of analysis and optimization in power systems.

pandapower is a joint development of the research group Energy Management and Power System Operation, University of Kassel and the Department for Distribution System
Operation at the Fraunhofer Institute for Wind Energy and Energy System Technology (IWES), Kassel.

For more information, go to `<http://www.uni-kassel.de/go/pandapower>`_.

Installation
==============
To install pandapower, simply use: ::

    pip install pandapower

This will install the following dependencies:
    - pypower>=5.0.1
    - numpy>1.8
    - pandas
    - networkx
    
To use all of pandapowers functionalites, you will need the following additional packages:
    - numba>=0.25.0 (for accelerated loadflow calculation)
    - matplotlib (for plotting)
    - python-igraph (for plotting networks without geographical information)
    - xlrd (for loading/saving files from/to excel)
    - openpyxl (for loading/saving files from/to excel)
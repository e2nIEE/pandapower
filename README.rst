=============
pandapower
=============

.. image:: https://api.codacy.com/project/badge/Grade/6770beaed39546f8bc4942abcbfa0d37
   :alt: Codacy Badge
   :target: https://www.codacy.com/app/jhmenke/pandapower?utm_source=github.com&utm_medium=referral&utm_content=lthurner/pandapower&utm_campaign=badger

.. image:: https://img.shields.io/pypi/v/pandapower.svg
   :target: https://pypi.python.org/pypi/pandapower

.. image:: https://img.shields.io/pypi/pyversions/pandapower.svg
    :target: https://pypi.python.org/pypi/pandapower

.. image:: https://travis-ci.org/lthurner/pandapower.svg?branch=master
    :target: https://travis-ci.org/lthurner/pandapower

.. image:: https://coveralls.io/repos/github/lthurner/pandapower/badge.svg?branch=master
    :target: https://coveralls.io/github/lthurner/pandapower?branch=master

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
    :target: https://github.com/lthurner/pandapower/blob/master/LICENSE  

pandapower combines the data analysis library `pandas <http://pandas.pydata.org>`_ and the power flow solver `PYPOWER <https://pypi.python.org/pypi/PYPOWER>`_ to create an easy to use network calculation program
aimed at automation of analysis and optimization in power systems.

pandapower is a joint development of the research group Energy Management and Power System Operation, University of Kassel and the Department for Distribution System
Operation at the Fraunhofer Institute for Wind Energy and Energy System Technology (IWES), Kassel.

.. image:: https://www.uni-kassel.de/eecs/typo3temp/pics/f26880008d.png
    :target: https://www.uni-kassel.de/eecs/en/fachgebiete/e2n/home.html

.. image:: http://www.energiesystemtechnik.iwes.fraunhofer.de/content/dam/iwes-neu/energiesystemtechnik/iwes_190x52.gif
   :target: http://www.energiesystemtechnik.iwes.fraunhofer.de/en.html
    

For more information, go to `<http://www.uni-kassel.de/go/pandapower>`_.

Installation
==============
To install pandapower, simply use: ::

    pip install pandapower

This will install the following dependencies:
    - pypower>=5.0.1
    - pandas
    - networkx

You will also need numpy and scipy, which are dependencies of pypower.
We recommend the `Anaconda Distribution <https://www.continuum.io/downloads>`_, which already contains a
numpy and scipy as well as a lot of other modules for scientific computing that are needed for working with 
pandapower.
   
To use all of pandapowers functionalites, you will need the following additional packages:
    - numba>=0.25.0 (for accelerated loadflow calculation)
    - matplotlib (for plotting)
    - python-igraph (for plotting networks without geographical information)
    - xlrd (for loading/saving files from/to excel)
    - openpyxl (for loading/saving files from/to excel)
    
All of these packages (except python-igraph) are already included in the anaconda distribution.
    
Minimal Example
=====================

A network in pandapower is represented in a pandapowerNet object, which is a collection of pandas Dataframes.
Each dataframe in a pandapowerNet contains the information about one pandapower element, such as line, load transformer etc.

We consider the following simple 3-bus example network as a minimal example:

.. image:: http://www.uni-kassel.de/eecs/fileadmin/datas/fb16/Fachgebiete/energiemanagement/Software/pandapower-doc/_images/3bus-system.png
		:width: 20em
		:align: center 

This network can be created in pandapower as follows: ::
    
    import pandapower as pp
    #create empty net
    net = pp.create_empty_network() 
    
    #create buses
    b1 = pp.create_bus(net, vn_kv=20., name="Bus 1")
    b2 = pp.create_bus(net, vn_kv=0.4, name="Bus 2")
    b3 = pp.create_bus(net, vn_kv=0.4, name="Bus 3")

    #create bus elements
    pp.create_ext_grid(net, bus=b1, vm_pu=1.02, name="Grid Connection")
    pp.create_load(net, bus=b3, p_kw=100, q_kvar=50, name="Load")
  
    #create branch elements
    tid = pp.create_transformer(net, hv_bus=b1, lv_bus=b2, std_type="0.4 MVA 20/0.4 kV",
                                name="Trafo")
    pp.create_line(net, from_bus=b2, to_bus=b3, length_km=0.1, name="Line",
                   std_type="NAYY 4x50 SE")   
                   
Note that you do not have to calculate any impedances or tap ratio for the equivalent circuit, this is handled internally by pandapower according to the pandapower `transformer model <http://www.uni-kassel.de/eecs/fileadmin/datas/fb16/Fachgebiete/energiemanagement/Software/pandapower-doc/elements/trafo.html#electric-model>`_.
The `standard type library <http://www.uni-kassel.de/eecs/fileadmin/datas/fb16/Fachgebiete/energiemanagement/Software/pandapower-doc/std_types.html>`_ allows comfortable creation of line and transformer elements. 

The pandapower representation now looks like this:

.. image:: http://www.uni-kassel.de/eecs/fileadmin/datas/fb16/Fachgebiete/energiemanagement/Software/pandapower-doc/_images/pandapower_datastructure.png
		:width: 40em

**Running a Power Flow**  

A powerflow can be carried out with the `runpp function <http://www.uni-kassel.de/eecs/fileadmin/datas/fb16/Fachgebiete/energiemanagement/Software/pandapower-doc/powerflow/ac.html>`_: ::
     
    pp.runpp(net)
    
When a power flow is run, pandapower combines the information of all element tables into one pypower case file and uses pypower to run the power flow.
The results are then processed and written back into pandapower:
        
.. image:: http://www.uni-kassel.de/eecs/fileadmin/datas/fb16/Fachgebiete/energiemanagement/Software/pandapower-doc/_images/pandapower_powerflow.png
		:width: 40em

For the 3-bus example network, the result tables look like this:

.. image:: http://www.uni-kassel.de/eecs/fileadmin/datas/fb16/Fachgebiete/energiemanagement/Software/pandapower-doc/_images/pandapower_results.png
		:width: 30em

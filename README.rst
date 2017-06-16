=============
pandapower
=============

.. image:: https://readthedocs.org/projects/pandapower/badge/?version=v1.3.1
   :target: http://pandapower.readthedocs.io/en/v1.3.1
   :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/pandapower.svg
   :target: https://pypi.python.org/pypi/pandapower

.. image:: https://img.shields.io/pypi/pyversions/pandapower.svg
    :target: https://pypi.python.org/pypi/pandapower

.. image:: https://travis-ci.org/lthurner/pandapower.svg?branch=master
    :target: https://travis-ci.org/lthurner/pandapower

.. image:: https://codecov.io/github/lthurner/pandapower/coverage.svg?branch=master
   :target: https://codecov.io/github/lthurner/pandapower?branch=master
    
.. image:: https://api.codacy.com/project/badge/Grade/5d749ed6772e47f6b84fb9afb83903d3
    :target: https://www.codacy.com/app/lthurner/pandapower?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=lthurner/pandapower&amp;utm_campaign=Badge_Grade

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
    :target: https://github.com/lthurner/pandapower/blob/master/LICENSE

pandapower combines the data analysis library `pandas <http://pandas.pydata.org>`_ and the power flow solver `PYPOWER <https://pypi.python.org/pypi/PYPOWER>`_ to create an easy to use network calculation program
aimed at automation of analysis and optimization in power systems.

pandapower is a joint development of the research group Energy Management and Power System Operation, University of Kassel and the Department for Distribution System
Operation at the Fraunhofer Institute for Wind Energy and Energy System Technology (IWES), Kassel.

.. image:: https://www.uni-kassel.de/eecs/typo3temp/pics/f26880008d.png
    :target: https://www.uni-kassel.de/eecs/en/fachgebiete/e2n/home.html
    :width: 25em

.. image:: http://www.energiesystemtechnik.iwes.fraunhofer.de/content/dam/iwes-neu/energiesystemtechnik/iwes_190x52.gif
   :target: http://www.energiesystemtechnik.iwes.fraunhofer.de/en.html
    :width: 25em

|

Element Models
---------------

pandapower is an element based network calculation tool that supports the following components:

	- lines
	- two-winding and three-winding transformers
	- ideal bus-bus and bus-branch switches
	- static generators
	- ZIP loads
	- shunts
	- external grid connections
	- synchronous generators
	- DC lines
	- unsymmetric impedances
	- ward equivalents
	
Network Analysis
------------------

pandapower supports the following network analysis functions:

	- power flow
	- optimal power flow
	- state estimation
	- short-circuit calculation according to IEC 60909
	- topological graph searches

For more information, please refer to the `documentation <https://pandapower.readthedocs.io>`_.

Installation notes can be found `here <http://pandapower.readthedocs.io/en/v1.3.1/getting_started/installation.html>`_, for a comfortable introduction into pandapower see the `interactive tutorials <http://pandapower.readthedocs.io/en/develop/getting_started/tutorials.html>`_.

If you are interested in getting release notes for new pandapower versions, please subscribe to the pandapower `mailing list <http://www.uni-kassel.de/go/pandapower>`_.

There is a project to develop a GUI for pandapower: https://github.com/Tooblippe/pandapower_gui - developers wanted!

Citing pandapower
==================

We are currently working on publishing a peer-reviewed paper about pandapower that should be used as reference in the future.

For now, please acknowledge the use of pandapower in publucations by citing the Technical Report: ::

    @TechReport{pandapower,
      author =      {Leon Thurner and Alexander Scheidler and Julian Dollichon and Florian Schäfer and Jan-Hendrik Menke and Friederike Meier and Steffen Meinecke and others},
      title =       {pandapower - Convenient Power System Modelling and Analysis based on PYPOWER and pandas},
      institution = {University of Kassel and Fraunhofer Institute for Wind Energy and Energy System Technology},
      year =        {2017},
      url =         {http://pandapower.readthedocs.io/en/v1.3.1/_downloads/pandapower.pdf}
    }

Minimal Example
===============

A network in pandapower is represented in a pandapowerNet object, which is a collection of pandas Dataframes.
Each dataframe in a pandapowerNet contains the information about one pandapower element, such as line, load transformer etc.

We consider the following simple 3-bus example network as a minimal example:

.. image:: http://pandapower.readthedocs.io/en/v1.3.1/_images/3bus-system.png
		:width: 20em
		:align: center 

Creating a Network
------------------------------

The above network can be created in pandapower as follows: ::
    
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
                   
Note that you do not have to calculate any impedances or tap ratio for the equivalent circuit, this is handled internally by pandapower according to the pandapower `transformer model <http://pandapower.readthedocs.io/en/v1.3.1/elements/trafo.html#electric-model>`_.
The `standard type library <http://pandapower.readthedocs.io/en/v1.3.1/std_types.html>`_ allows comfortable creation of line and transformer elements. 

The pandapower representation now looks like this:

.. image:: http://pandapower.readthedocs.io/en/v1.3.1/_images/pandapower_datastructure.png
		:width: 40em

Running a Power Flow
------------------------------

A powerflow can be carried out with the `runpp function <http://pandapower.readthedocs.io/en/v1.3.1/powerflow/ac.html>`_: ::
     
    pp.runpp(net)
    
When a power flow is run, pandapower combines the information of all element tables into one pypower case file and uses pypower to run the power flow.
The results are then processed and written back into pandapower:
        
.. image:: http://pandapower.readthedocs.io/en/v1.3.1/_images/pandapower_powerflow.png
		:width: 40em

For the 3-bus example network, the result tables look like this:

.. image:: http://pandapower.readthedocs.io/en/v1.3.1/_images/pandapower_results.png
		:width: 30em
		
All other pandapower elements and network analysis functionality (e.g. optimal power flow, state estimation or short-circuit calculation) is also fully integrated into the tabular pandapower datastructure.

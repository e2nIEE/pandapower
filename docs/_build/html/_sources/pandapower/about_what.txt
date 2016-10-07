=====================
What is pandapower?
=====================

pandapower combines the data analysis library `pandas <http://pandas.pydata.org/>`_ and the power flow solver `PYPOWER <https://pypi.python.org/pypi/PYPOWER>`_ to create an easy to use network calculation program.
pandapower is aimed at automation of power system analysis and optimization in distribution and sub-transmission networks.

pandapower is based on electric elements rather than on generic loadflow attributes. For example, in PYPOWER busses have a power demand and shunt admittance, even though these are in reality the attributes of electric
elements (such as loads, pv generators or capacitor banks) which are connected to the busses. In pandapower, we model each electric bus element instead of considering summed values for each bus.
The same goes for branches: in reality, busses in a network are connected by electric elements like lines and transformers that can be defined by a length and cable type (lines) or short circuit 
voltages and rated power (transformers). Since the electric models for lines and transformers are implemented in pandapower, it is possible to model the electric elements with these common nameplate
attributes. All parameters which are necessary for the loadflow (like branch per unit impedances, shunt impedances, bus power, bus loadflow type etc.) are then calculated and handled internally by pandapower.

A network in pandapower is represented in a pandapowerNet object, which is a collection of pandas Dataframes.
Each dataframe in a pandapowerNet contains the information about one pandapower element, such as line, load transformer etc.

For the following simple 2-bus example network:

.. image:: /pandapower/pics/2bus-system.png
		:width: 20em
		:alt: alternate Text
		:align: center 

the pandapower representation looks like this:

.. image:: /pandapower/pics/pandapower_datastructure.png
		:width: 40em
		:alt: alternate Text
		:align: center

The network can be created with the :ref:`pandapower create functions <create_functions>`, but it also possible to directly manipulate data in the pandapower dataframes.

When a loadflow is run, pandapower combines the information of all element tables into one pypower case file and uses pypower to run the loadflow. The results are then processed and written back into pandapower:
        
.. image:: /pandapower/pics/pandapower_loadflow.png
		:width: 40em
		:alt: alternate Text
		:align: center

For the 2-bus example network, the result tables look like this:

.. image:: /pandapower/pics/pandapower_results.png
		:width: 40em
		:alt: alternate Text
		:align: center

You can download the python script that creates this 2-bus system :download:`here  <../_downloads/pandapower_2bus_system.py>`. For more complicated pandapower example, see the :ref:`pandapower example networks <example_networks>`.
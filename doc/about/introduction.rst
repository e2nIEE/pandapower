=====================
A short Introduction
=====================

**Electric Model**

The development of pandapower started as an extension of the widely used power flow solver MATPOWER and its port to python, PYPOWER. 
The electric attributes of the network are defined in the MATPOWER casefile in the form of a bus/branch model. The bus/branch model 
formulation is mathematically very close the loadflow, which is why it is easy to 

It also  

pandapower is network calculation tool based on electric elements rather than on generic loadflow attributes. For example, each electric bus element (like load, static generator, external grid) is 
modeled individually instead of considering summed values for each bus. Branch elements like lines and transformers can be defined by a length and cable type (lines) or short circuit 
voltages and rated power (transformers). Since the electric models for lines and transformers are implemented in pandapower, it is possible to model the electric elements with these common nameplate
attributes. All parameters which are necessary for the loadflow (like branch per unit impedances, shunt impedances, bus power, bus loadflow type etc.) are then calculated and handled internally by pandapower.

**Datastructure**

A network in pandapower is represented in a pandapowerNet object, which is a collection of pandas Dataframes.
Each dataframe in a pandapowerNet contains the information about one pandapower element, such as line, load transformer etc.

For the following simple 2-bus example network:

.. image:: /pics/2bus-system.png
		:width: 20em
		:alt: alternate Text
		:align: center 

the pandapower representation looks like this:

.. image:: /pics/pandapower_datastructure.png
		:width: 40em
		:alt: alternate Text
		:align: center

The network can be created with the pandapower create functions, but it also possible to directly manipulate data in the pandapower dataframes.
You can download the python script that creates this 2-bus system :download:`here  <pandapower_2bus_system.py>`.

**Loadflow**

When a loadflow is run, pandapower combines the information of all element tables into one pypower case file and uses pypower to run the loadflow.
The results are then processed and written back into pandapower:
        
.. image:: /pics/pandapower_loadflow.png
		:width: 40em
		:alt: alternate Text
		:align: center

For the 2-bus example network, the result tables look like this:

.. image:: /pics/pandapower_results.png
		:width: 40em
		:alt: alternate Text
		:align: center
        
The same workflow applies to running a DC or an optimal power flow.

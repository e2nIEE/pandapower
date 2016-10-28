=====================
A short Introduction
=====================

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

When a power flow is run, pandapower combines the information of all element tables into one pypower case file and uses pypower to run the power flow.
The results are then processed and written back into pandapower:
        
.. image:: /pics/pandapower_power flow.png
		:width: 40em
		:alt: alternate Text
		:align: center

For the 2-bus example network, the result tables look like this:

.. image:: /pics/pandapower_results.png
		:width: 40em
		:alt: alternate Text
		:align: center
        
The same workflow applies to running a DC or an optimal power flow.

You can download the python script that creates this 2-bus system :download:`here  <pandapower_2bus_system.py>`. For more extensive examples, refer to the :ref:`pandapower tutorials<tutorial>`.


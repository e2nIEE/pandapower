=====================
A Short Introduction
=====================

A network in pandapower is represented in a pandapowerNet object, which is a collection of pandas Dataframes.
Each dataframe in a pandapowerNet contains the information about one pandapower element, such as line, load transformer etc.

For the following simple 2-bus example network:

.. image:: /pics/2bus-system.png
		:width: 20em
		:alt: alternate Text
		:align: center 

The network can be created with the pandapower create functions: ::
    
    import pandapower as pp
    net = pp.create_empty_network()
    pp.create_bus(net, vn_kv=0.4, name="Bus 1")
    pp.create_ext_grid(net, bus=b1, vm_pu=1.02, name="Grid Connection")
    pp.create_bus(net, vn_kv=0.4, name="Bus 2")
    pp.create_line(net, from_bus=b1, to_bus=b2, length_km=0.1, std_type="NAYY 4x50 SE", name="Line")     
    pp.create_load(net, bus=b2, p_kw=100, q_kvar=50, name="Load")

the pandapower representation then looks like this:

.. image:: /pics/pandapower_datastructure.png
		:width: 40em
		:alt: alternate Text
		:align: center

When a power flow is run: ::
    
    pp.runpp(net)
    
pandapower combines the information of all element tables into one pypower case file and uses pypower to run the power flow.
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
        
You can download the python script that creates this 2-bus system :download:`here  <pandapower_2bus_system.py>`.

For a more in depth introduction into pandapower modeling and analysis functionality, see the :ref:`pandapower tutorials<tutorial>`
about network creation, standard type libraries, power flow, topological searches, plotting and more.
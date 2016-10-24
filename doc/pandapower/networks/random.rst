
Random pandapower networks
==========================

Random line network
-------------------

.. autofunction:: pandapower.networks.random_line_network

.. code:: python 

 import pandapower as pp

 net1_1 = pp.networks.random_line_network(nr_buses_main=3, p_pv=0.0, p_wp=0.0)
 net1_2 = pp.networks.random_line_network(nr_buses_main=3, p_pv=1.0, p_wp=1.0)
 net1_3 = pp.networks.random_line_network(nr_buses_main=3, p_pv=0.5, p_wp=0.5, branches=[(1, 3), (2, 1)])
 
 
.. image:: /pandapower/pics/random_nw_random_line_network.png
	:width: 30em
	:alt: alternate Text
	:align: center


---------------------------

Random empty grid
-----------------

.. autofunction:: pandapower.networks.random_empty_grid

 Graphical Examples:

.. code:: python 

 import pandapower as pp

 net2_1 = pp.networks.random_empty_grid(num_buses=5, p=0.0)
 net2_2 = pp.networks.random_empty_grid(num_buses=5, p=0.5)
 net2_3 = pp.networks.random_empty_grid(num_buses=5, p=1.0)


.. image:: /pandapower/pics/random_nw_random_empty_grid.png
	:width: 40em
	:alt: alternate Text
	:align: center	

---------------------------

Set up random empty grid with loads
-----------------

.. autofunction:: pandapower.networks.setup_grid
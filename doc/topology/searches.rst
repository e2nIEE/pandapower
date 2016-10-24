======================
Topological Searches
======================

Once you converted your network into a MultiGraph there are several functions to perform topological searches and analyses at your disposal.
You can either use the general-purpose functions that come with NetworkX (see http:/networkx.github.io/documentation/networkx-1.10/reference/algorithms.html)
or topology's own ones which are specialized on electrical networks.


calc_distance_to_bus
---------------------

.. autofunction:: pandapower.topology.calc_distance_to_bus

connected_component
---------------------

.. autofunction:: pandapower.topology.connected_component

connected_components
---------------------

.. autofunction:: pandapower.topology.connected_components


unsupplied_buses
---------------------

.. autofunction:: pandapower.topology.unsupplied_buses

determine_stubs
---------------------

.. autofunction:: pandapower.topology.determine_stubs

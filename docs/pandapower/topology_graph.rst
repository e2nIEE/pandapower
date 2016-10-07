======================
Create networkx graph
======================

The basis of all topology functions is the conversion of a padapower network into a NetworkX MultiGraph. A MultiGraph is a simplified representation of a network's topology, reduced to nodes and edges.
Busses are being represented by nodes (Note: only busses with in_service = 1 appear in the graph), edges represent physical connections between busses (typically lines or trafos). 
Multiple parallel edges between nodes are possible.

This is a very simple example of a pandapower network being converted to a MultiGraph. (Note: The MultiGraph's shape is completely arbitrary since MultiGraphs have no inherent shape unless geodata is provided.)

.. image:: /pandapower/pics/multigraph_example.png
	:width: 42em
	:alt: alternate Text
	:align: center
	
Nodes have the same indicees as the busses they originate from. Edges are defined by the nodes they connect.
Additionally nodes and edges can hold key/value attribute pairs.

The following attributes get transferred into the MultiGraph:

+--------------------------+-------------------+
| lines                    | trafos            |
+==========================+===================+
|   - from_bus             |    - hv_bus       |
|   - to_bus               |    - lv_bus       |
|   - length_km            |    - index        |
|   - index                |    - in_service   |
|   - in_service           |                   |
|   - imax_ka              |                   |
+--------------------------+-------------------+

Apart from these there are no element attributes contained in the MultiGraph!

**Creating a multigraph from a pandapower network**


The function create_nxgraph function from the pandapower.topology package allows you to convert a pandapower network into a MultiGraph:


.. autofunction:: pandapower.topology.create_nxgraph



**Examples**

.. code:: python

	create_nxgraph(net, respect_switches = False)

.. image:: /pandapower/pics/multigraph_example_respect_switches.png
	:width: 42em
	:alt: alternate Text
	:align: center


.. code:: python

	create_nxgraph(net, include_lines = False)

.. image:: /pandapower/pics/multigraph_example_include_lines.png
	:width: 42em
	:alt: alternate Text
	:align: center
	

.. code:: python

	create_nxgraph(net, include_trafos = False)

.. image:: /pandapower/pics/multigraph_example_include_trafos.png
	:width: 42em
	:alt: alternate Text
	:align: center
	

.. code:: python

	create_nxgraph(net, nogobusses = [4])

.. image:: /pandapower/pics/multigraph_example_nogobusses.png
	:width: 42em
	:alt: alternate Text
	:align: center
	
	

.. code:: python

	create_nxgraph(net, notravbusses = [4])

.. image:: /pandapower/pics/multigraph_example_notravbusses.png
	:width: 42em
	:alt: alternate Text
	:align: center
=============================
Generic Coordinates
=============================

If there are no geocoordinates in a network, generic coordinates can be created. There are two possibilities:

    - with python-igraph: http:/igraph.org/python/ (recommended)
    - with networkx and graphviz (http:/www.graphviz.org)

Generically created geocoordinates can then be plotted in the same way as real geocoordinates.


.. autofunction:: pandapower.plotting.create_generic_coordinates


Example plot with mv_oberrhein network from the pandapower.networks package as geographical plan (respect_switches=False):

.. image:: /pics/plotting_tutorial2.png
	:width: 30em
	:align: center
    
and as structural plan (respect_switches=True):

.. image:: /pics/plotting_tutorial3.png
	:width: 20em
	:align: center




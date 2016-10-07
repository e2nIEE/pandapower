=============================
Generic Coordinates
=============================

If there are no geocoordinates in a network, generic coordinates can be created. 

There are two possibilities: GraphViz can be used in conjunction with NetworkX or igraph's layout library can be chosen.
GraphViz has to be installed separately (http://www.graphviz.org).

Igraph can be installed from pre-compiled wheels (http://www.lfd.uci.edu/~gohlke/pythonlibs/#python-igraph).

 
.. autofunction:: pandapower.plotting.create_generic_coordinates


Generically created geocoordinates can then be plotted in the same way as real geocoordinates:

.. image:: /pandapower/pics/plotting_tutorial6.png
	:scale: 100%
	:align: center




#############################
Plotting pandapower Networks
#############################

pandapower provides the functionality to translate pandapower network elements into matplotlib collections. The different collections for lines, busses or transformers can than be drawn with pyplot.

If no coordinates are available for the busses, pandapower provides possibility to create generic coordinates through the igraph package. If no geocoordinates are available for the lines, they
can be plotted as direct connections between the busses.

.. toctree:: 
    :maxdepth: 1
    
    plotting_collections
    plotting_draw
    plotting_generic




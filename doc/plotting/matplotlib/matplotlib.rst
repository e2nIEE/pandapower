#############################
Matplotlib Network Plots
#############################

pandapower provides the functionality to translate pandapower network elements into matplotlib collections. The different collections for lines, buses or transformers can than be drawn with pyplot.

If no coordinates are available for the buses, pandapower provides possibility to create generic coordinates through the igraph package. If no geocoordinates are available for the lines, they
can be plotted as direct connections between the buses.

.. toctree::
    :maxdepth: 1

    simple_plot
    create_collections
    create_colormaps
    draw
    generic

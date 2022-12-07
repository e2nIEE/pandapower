##########################
Topological Searches
##########################

pandapower provides the possibility of graph searches using the networkx package, which is "a Python language software package for the creation, manipulation, and study of the structure, dynamics,
and function of complex networks." (see NetworkX documentation https://networkx.org/documentation/stable/ )

pandapower provides a function to translate pandapower networks into networkx graphs. Once the electric network is translated into an abstract networkx graph, all network operations that
are available in networkx can be used to analyse the network. For example you can find the shortest path between two nodes, find out if two areas in a network are connected to each other or
if there are cycles in a network.  For a complete list of all NetworkX algorithms see https://networkx.org/documentation/stable/reference/algorithms/index.html

pandapower also provides some search algorithms specialiced for electric networks, such as finding all buses that are connected to a slack node.

.. toctree::
    :maxdepth: 1

    topology/create_graph
    topology/searches
    topology/examples
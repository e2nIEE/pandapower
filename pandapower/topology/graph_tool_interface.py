import copy
from graph_tool import Graph


class GraphToolInterface(Graph):
    """
    A interface to Graph Tool from https://graph-tool.skewed.de, which looks like networkx so that it can be
    used by the pandapower topology package without any code changes.

    Notice:
    The graph-tool graph needs to have consecutive buses starting with 0 and be in range(0, max_bus).
    This means that "nogobuses" are not working (yet) and also you should use
    pandapower.toolbox.create_continuous_bus_index() to have a graph without many "unnecessary" buses
    """

    def __init__(self, bus_indices, g=None, directed=False, prune=False, vorder=None):
        Graph.__init__(self, g, directed, prune, vorder)

        # init edge properties (=edge_data of networkx)
        self.properties["e", "edge_data"] = self.new_edge_property("object")
        self.pp_buses = set(copy.copy(bus_indices))

    def add_nodes_from(self, nodes):
        """
        adds multiple nodes to graph

        INPUT:
        **nodes** (iterable or int) - list of nodes (example: bus.index) or single int

        """
        self.add_vertex(nodes)

    def add_node(self, node):
        # adds node
        self.add_vertex(node)

    def remove_node(self, node, fast=False):
        # removes single node
        self.remove_vertex(vertex=node, fast=fast)

    def remove_nodes(self, nodes):
        # removes nodes in iterable
        return self.vertices(nodes)

    def nodes(self):
        return set(self.get_vertices()) & self.pp_buses

    ### EDGE Functions

    def add_edge_data(self, edge):
        # Todo: Add new_edge_property?
        pass

    def get_edge_data(self, source, target, key=None):
        edges = self.edge(source, target, all_edges=True)
        if not len(edges):
            # no edges for these buses
            return None

        edge_data = dict()
        for e in edges:
            e_key, e_val = self.edge_properties["edge_data"][e].popitem()
            if key is not None:
                if key == e_key: return e_val
            edge_data[e_key] = e_val
        return edge_data

    def add_edge(self, source, target, add_missing=True, **kwargs):

        # add the edge
        edge = super(GraphToolInterface, self).add_edge(source, target, add_missing=add_missing)
        # add the properties from kwargs
        if "key" in kwargs:
            key = kwargs.pop("key")
            self.edge_properties["edge_data"][edge] = {key: kwargs}

    ### MISC functions
    def is_multigraph(self):
        return True

# -*- coding: utf-8 -*-
from __future__ import division
__author__ = "Jan-Hendrik Menke"

import pandas as pd
import networkx as nx
import pandapower.topology as top
import copy

def build_igraph_from_pp(net, respect_switches=False):
    """
    This function uses the igraph library to create an igraph graph for a given pandapower network.
    Lines, transformers and switches are respected.
    Performance vs. networkx: https://graph-tool.skewed.de/performance

    Input:

        **net** - Pandapower network

    Example:

        graph = build_igraph_from_pp(net

    """
    import igraph as ig
    g = ig.Graph(directed=True)
    g.add_vertices(net.bus.shape[0])
    g.vs["label"] = net.bus.index.tolist()  # [s.encode('unicode-escape') for s in net.bus.name.tolist()]
    pp_bus_mapping = dict(list(zip(net.bus.index, list(range(net.bus.index.shape[0])))))

    # add lines
    nogolines = set(net.switch.element[(net.switch.et == "l") & (net.switch.closed == 0)]) \
                if respect_switches else set()
    for lix in (ix for ix in net.line.index if ix not in nogolines):
        l = net.line.ix[lix]
        g.add_edge(pp_bus_mapping[l.from_bus], pp_bus_mapping[l.to_bus])
    g.es["weight"] = net.line.length_km.values

    # add trafos
    for tix in net.trafo.index:
        t = net.trafo.ix[tix]
        g.add_edge(pp_bus_mapping[t.hv_bus], pp_bus_mapping[t.lv_bus], weight=0.01)

    # add switches
    bs = net.switch[(net.switch.et == "b") & (net.switch.closed == 1)]
    for fb, tb in zip(bs.bus, bs.element):
        g.add_edge(pp_bus_mapping[fb], pp_bus_mapping[tb], weight=0.001)

    return g, (not g.is_dag())

                            
                            
def create_generic_coordinates(net, mg=None, preferNx=False, respect_switches=False):
    """
    This function will add arbitrary geo-coordinates for all buses based on an analysis of branches and rings.
    It will remove out of service buses/lines from the net. The coordinates will be created either by igraph or by
    using networkx library.

    Input:

        **net** - Pandapower network

    Optional:

        **mg** - Existing networkx multigraph, if available. Convenience to save computation time.

        **preferNx** - Use networkx for coordinates even when igraph is installed and available

    Output:

        **net** - Pandapower network with added geo coordinates for the buses

    Example:

        net = create_generic_coordinates(net)

    """
    if "bus_geodata" in net and net.bus_geodata.shape[0]:
        print("Please delete all geodata. This function cannot be used with pre-existing geodata.")
        return
    if not "bus_geodata" in net:
        net.bus_geodata = pd.DataFrame(columns=["x","y"])
    gnet = copy.deepcopy(net)
    gnet.bus = gnet.bus[gnet.bus.in_service == True]
    if not preferNx:
        try:
            import igraph as ig
            graph, meshed = build_igraph_from_pp(gnet, respect_switches)
            if meshed:
                layout = graph.layout("kk")
            else:
                graph.to_undirected(mode="each", combine_edges="first")
                layout = graph.layout("rt")
            coords = list(zip(*layout.coords))
        except ImportError:
            preferNx = True
    if preferNx:
        if mg is None:
            nxg = top.create_nxgraph(gnet, respect_switches)
        else:
            nxg = copy.deepcopy(mg)
        # workaround for bug in agraph
        for u, v in nxg.edges_iter(data=False, keys=False):
            if 'key' in nxg[int(u)][int(v)]:
                del nxg[int(u)][int(v)]['key']
            if 'key' in nxg[int(u)][int(v)][0]:
                del nxg[int(u)][int(v)][0]['key']
        coords = list(zip(*(list(nx.drawing.nx_agraph.graphviz_layout(nxg, prog='neato').values()))))
    net.bus_geodata.x = coords[0]
    net.bus_geodata.y = coords[1]
    return net


def fuse_geodata(net):
    mg = top.create_nxgraph(net, include_lines=False, respect_switches=False)
    geocoords = set(net.bus_geodata.index)
    for area in top.connected_components(mg):
        if len(area & geocoords) > 1:
            geo = net.bus_geodata.loc[area & geocoords].values[0]
            for bus in area:
                net.bus_geodata.loc[bus] = geo


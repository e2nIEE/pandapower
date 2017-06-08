# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import copy

import networkx as nx
import pandas as pd

import pandapower.topology as top


def build_igraph_from_pp(net, respect_switches=False):
    """
    This function uses the igraph library to create an igraph graph for a given pandapower network.
    Lines, transformers and switches are respected.
    Performance vs. networkx: https://graph-tool.skewed.de/performance

    Input:

        **net** - pandapower network

    Example:

        graph = build_igraph_from_pp(net

    """
    try:
        import igraph as ig
    except (DeprecationWarning, ImportError):
        raise ImportError("Please install python-igraph")
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
    bs = net.switch[(net.switch.et == "b") & (net.switch.closed == 1)] if respect_switches else \
                    net.switch[(net.switch.et == "b")]
    for fb, tb in zip(bs.bus, bs.element):
        g.add_edge(pp_bus_mapping[fb], pp_bus_mapping[tb], weight=0.001)

    meshed = False
    for i in range(1, net.bus.shape[0]):
        if len(g.get_all_shortest_paths(0, i, mode="ALL")) > 1:
            meshed = True
            break

    roots = [pp_bus_mapping[s] for s in net.ext_grid.bus.values]
    return g, meshed, roots  # g, (not g.is_dag())


def create_generic_coordinates(net, mg=None, library="igraph", respect_switches=False):
    """
    This function will add arbitrary geo-coordinates for all buses based on an analysis of branches and rings.
    It will remove out of service buses/lines from the net. The coordinates will be created either by igraph or by
    using networkx library.

    INPUT:
        **net** - pandapower network

    OPTIONAL:
        **mg** - Existing networkx multigraph, if available. Convenience to save computation time.

        **library** - "igraph" to use igraph package or "networkx" to use networkx package

    OUTPUT:
        **net** - pandapower network with added geo coordinates for the buses

    EXAMPLE:
        net = create_generic_coordinates(net)

    """
    if "bus_geodata" in net and net.bus_geodata.shape[0]:
        print("Please delete all geodata. This function cannot be used with pre-existing geodata.")
        return
    if not "bus_geodata" in net:
        net.bus_geodata = pd.DataFrame(columns=["x", "y"])
    gnet = copy.deepcopy(net)
    gnet.bus = gnet.bus[gnet.bus.in_service == True]
    if library == "igraph":
        try:
            import igraph
        except ImportError:
            raise UserWarning("The library igraph is selected for plotting, "
                              "but not installed correctly.")
        graph, meshed, roots = build_igraph_from_pp(gnet, respect_switches)
        if meshed:
            layout = graph.layout("kk")
        else:
            graph.to_undirected(mode="each", combine_edges="first")
            layout = graph.layout("rt", root=roots)
        coords = list(zip(*layout.coords))
    elif library == "networkx":
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
        # ToDo: Insert fallback layout for nxgraph
        coords = list(zip(*(list(nx.drawing.nx_agraph.graphviz_layout(nxg, prog='neato').values()))))
    else:
        raise ValueError("Unknown library %s - chose 'igraph' or 'networkx'"%library)
    net.bus_geodata.x = coords[0]
    net.bus_geodata.y = coords[1]
    net.bus_geodata.index = net.bus.index
    return net


def fuse_geodata(net):
    mg = top.create_nxgraph(net, include_lines=False, respect_switches=False)
    geocoords = set(net.bus_geodata.index)
    for area in top.connected_components(mg):
        if len(area & geocoords) > 1:
            geo = net.bus_geodata.loc[area & geocoords].values[0]
            for bus in area:
                net.bus_geodata.loc[bus] = geo


# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import sys
import copy

import networkx as nx
import pandas as pd
import numpy as np

from pandapower.auxiliary import soft_dependency_error
import pandapower.topology as top

try:
    import igraph
    IGRAPH_INSTALLED = True
except ImportError:
    IGRAPH_INSTALLED = False

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def build_igraph_from_pp(net, respect_switches=False, buses=None, trafo_length_km=0.01, switch_length_km=0.001):
    """
    This function uses the igraph library to create an igraph graph for a given pandapower network.
    Lines, transformers and switches are respected.
    Performance vs. networkx: https://graph-tool.skewed.de/performance

    :param net: pandapower network
    :type net: pandapowerNet
    :param respect_switches: if True, exclude edges for open switches (also lines that are \
        connected via line switches)
    :type respect_switches: bool, default False

    :Example:
        graph, meshed, roots = build_igraph_from_pp(net)
    """
    if not IGRAPH_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name)+"()", "igraph")
    g = igraph.Graph(directed=True)
    bus_index = net.bus.index if buses is None else np.array(buses)
    nr_buses = len(bus_index)
    g.add_vertices(nr_buses)
    # g.vs["label"] = [s.encode('unicode-escape') for s in net.bus.name.tolist()]
    g.vs["label"] = list(bus_index)
    pp_bus_mapping = dict(list(zip(bus_index, list(range(nr_buses)))))
    if respect_switches:
        open_switches = ~net.switch.closed.values.astype(bool)
    # add lines
    mask = _get_element_mask_from_nodes(net, "line", ["from_bus", "to_bus"], buses)
    if respect_switches:
        mask &= _get_switch_mask(net, "line", "l", open_switches)
    for line in net.line[mask].itertuples():
        g.add_edge(pp_bus_mapping[line.from_bus],
                   pp_bus_mapping[line.to_bus],
                   weight=line.length_km)

    # add trafos
    mask = _get_element_mask_from_nodes(net, "trafo", ["hv_bus", "lv_bus"], buses)
    if respect_switches:
        mask &= _get_switch_mask(net, "trafo", "t", open_switches)
    for trafo in net.trafo[mask].itertuples():
        g.add_edge(pp_bus_mapping[trafo.hv_bus],
                   pp_bus_mapping[trafo.lv_bus], weight=trafo_length_km)

    # add trafo3w
    mask = _get_element_mask_from_nodes(net, "trafo3w", ["hv_bus", "mv_bus", "lv_bus"], buses)
    if respect_switches:
        mask &= _get_switch_mask(net, "trafo3w", "t3", open_switches)
    for trafo3w in net.trafo3w[mask].itertuples():
        g.add_edge(pp_bus_mapping[trafo3w.hv_bus],
                   pp_bus_mapping[trafo3w.lv_bus], weight=trafo_length_km)
        g.add_edge(pp_bus_mapping[trafo3w.hv_bus],
                   pp_bus_mapping[trafo3w.mv_bus], weight=trafo_length_km)

    # add switches
    mask = net.switch.et.values == "b"
    if respect_switches:
        mask &= ~open_switches
    bus_mask = _get_element_mask_from_nodes(net, "switch", ["element", "bus"], buses)
    for switch in net.switch[mask & bus_mask].itertuples():
        g.add_edge(pp_bus_mapping[switch.element],
                   pp_bus_mapping[switch.bus], weight=switch_length_km)

    meshed = _igraph_meshed(g)

    roots = [pp_bus_mapping[b] for b in net.ext_grid.bus.values if b in bus_index]
    return g, meshed, roots  # g, (not g.is_dag())


def _igraph_meshed(g):
    for i in range(1, g.vcount()):
        if len(g.get_all_shortest_paths(0, i, mode="ALL")) > 1:
            return True
    return False

def _get_element_mask_from_nodes(net, element, node_elements, nodes=None):
    mask = np.ones(len(net[element])).astype(bool)
    if nodes is not None:
        for node_element in node_elements:
            mask &= np.isin(net[element][node_element].values, nodes)
    return mask

def _get_switch_mask(net, element, switch_element, open_switches):
    element_switches = net.switch.et.values == switch_element
    open_elements = net.switch.element.values[open_switches & element_switches]
    open_element_mask = np.in1d(net[element].index, open_elements, invert=True)
    return open_element_mask

def coords_from_igraph(graph, roots, meshed=False, calculate_meshed=False):
    """
    Create a list of generic coordinates from an igraph graph layout.

    :param graph: The igraph graph on which the coordinates shall be based
    :type graph: igraph.Graph
    :param roots: The root buses of the graph
    :type roots: iterable
    :param meshed: determines if the graph has any meshes
    :type meshed: bool, default False
    :param calculate_meshed: determines whether to calculate the meshed status
    :type calculate_meshed: bool, default False
    :return: coords - list of coordinates from the graph layout
    """
    if calculate_meshed:
        meshed = False
        for i in range(1, len(graph.vs)):
            if len(graph.get_all_shortest_paths(0, i, mode="ALL")) > 1:
                meshed = True
                break
    if meshed is True:
        layout = graph.layout("kk")
    else:
        graph.to_undirected(mode="each", combine_edges="first")
        layout = graph.layout("rt", root=roots)
    return list(zip(*layout.coords))


def coords_from_nxgraph(mg=None, layout_engine='neato'):
    """
    Create a list of generic coordinates from a networkx graph layout.

    :param mg: The networkx graph on which the coordinates shall be based
    :type mg: networkx.Graph
    :param layout_engine: GraphViz Layout Engine for layouting a network. See https://graphviz.org/docs/layouts/
    :type layout_engine: str
    :return: coords - list of coordinates from the graph layout
    """
    # workaround for bug in agraph
    for u, v in mg.edges(data=False):
        if 'key' in mg[int(u)][int(v)]:
            del mg[int(u)][int(v)]['key']
        if 'key' in mg[int(u)][int(v)].get(0, ()):
            del mg[int(u)][int(v)][0]['key']
    # ToDo: Insert fallback layout for nxgraph
    return list(zip(*(list(nx.drawing.nx_agraph.graphviz_layout(mg, prog=layout_engine).values()))))


def create_generic_coordinates(net, mg=None, library="igraph",
                               respect_switches=False,
                               geodata_table="bus_geodata",
                               buses=None,
                               overwrite=False,
                               layout_engine='neato',
                               trafo_length_km=0.01,
                               switch_length_km=0.001):
    """
    This function will add arbitrary geo-coordinates for all buses based on an analysis of branches
    and rings. It will remove out of service buses/lines from the net. The coordinates will be
    created either by igraph or by using networkx library.

    :param net: pandapower network
    :type net: pandapowerNet
    :param mg: Existing networkx multigraph, if available. Convenience to save computation time.
    :type mg: networkx.Graph
    :param respect_switches: respect switches in a network for generic coordinates
    :type respect_switches: bool
    :param library: "igraph" to use igraph package or "networkx" to use networkx package
    :type library: str
    :param geodata_table: table to write the generic geodatas to
    :type geodata_table: str
    :param buses: buses for which generic geodata are created, all buses will be used by default
    :type buses: list
    :param overwrite: overwrite existing geodata
    :type overwrite: bool
    :param layout_engine: GraphViz Layout Engine for layouting a network. See https://graphviz.org/docs/layouts/
    :type layout_engine: str
    :return: net - pandapower network with added geo coordinates for the buses

    :Example:
        net = create_generic_coordinates(net)
    """

    _prepare_geodata_table(net, geodata_table, overwrite)
    if library == "igraph":
        if not IGRAPH_INSTALLED:
            soft_dependency_error("build_igraph_from_pp()", "igraph")
        graph, meshed, roots = build_igraph_from_pp(net, respect_switches, buses=buses,
                                                    trafo_length_km=trafo_length_km, switch_length_km=switch_length_km)
        coords = coords_from_igraph(graph, roots, meshed)
    elif library == "networkx":
        if mg is None:
            nxg = top.create_nxgraph(net, respect_switches=respect_switches,
                                     include_out_of_service=True,
                                     trafo_length_km=trafo_length_km, switch_length_km=switch_length_km)
        else:
            nxg = copy.deepcopy(mg)
        coords = coords_from_nxgraph(nxg, layout_engine=layout_engine)
    else:
        raise ValueError("Unknown library %s - chose 'igraph' or 'networkx'" % library)
    if len(coords):
        net[geodata_table].x = coords[1]
        net[geodata_table].y = coords[0]
        net[geodata_table].index = net.bus.index if buses is None else buses
    return net

def _prepare_geodata_table(net, geodata_table, overwrite):
    if geodata_table in net and net[geodata_table].shape[0]:
        if overwrite:
            net[geodata_table] = net[geodata_table].drop(net[geodata_table].index)
        else:
            raise UserWarning("Table %s is not empty - use overwrite=True to overwrite existing geodata"%geodata_table)

    if geodata_table not in net or net[geodata_table] is None:
        net[geodata_table] = pd.DataFrame(columns=["x", "y"])

def fuse_geodata(net):
    mg = top.create_nxgraph(net, include_lines=False, include_impedances=False,
                            respect_switches=False)
    geocoords = set(net.bus_geodata.index)
    for area in top.connected_components(mg):
        if len(area & geocoords) > 1:
            geo = net.bus_geodata.loc[list(area & geocoords)].values[0]
            for bus in area:
                net.bus_geodata.loc[bus] = geo

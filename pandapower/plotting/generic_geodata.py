# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import sys
import copy
from typing import TYPE_CHECKING, Iterable

import geojson
import networkx
import networkx as nx
import pandas as pd
import numpy as np

from pandapower.auxiliary import pandapowerNet, soft_dependency_error
from pandapower.create.utils import add_column_to_df
from pandapower.topology.create_graph import create_nxgraph
from pandapower.topology.graph_searches import connected_components

try:
    import igraph
    IGRAPH_INSTALLED = True
except ImportError:
    IGRAPH_INSTALLED = False

import logging

if TYPE_CHECKING:
    import igraph

logger = logging.getLogger(__name__)


def build_igraph_from_pp(
        net: pandapowerNet,
        respect_switches: bool = False,
        buses=None,
        trafo_length_km=0.01,
        switch_length_km=0.001,
        dcline_length_km=1.0
):
    """
    This function uses the igraph library to create an igraph graph for a given pandapower network.
    Lines, transformers and switches are respected.
    Performance vs. networkx: https://graph-tool.skewed.de/performance

    Parameters:
        net: the pandapower network
        respect_switches: if True, exclude edges for open switches (also lines that are connected via line switches)

    Example:
        >>> graph, meshed, roots = build_igraph_from_pp(net)
    """
    if not IGRAPH_INSTALLED:
        soft_dependency_error(str(sys._getframe().f_code.co_name)+"()", "igraph")
    g = igraph.Graph(directed=True)
    bus_index = net.bus.index if buses is None else np.array(buses)
    nr_buses = len(bus_index)
    g.add_vertices(nr_buses)
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

    # add dclines
    mask = _get_element_mask_from_nodes(net, "dcline", ["from_bus", "to_bus"], buses)
    if respect_switches:
        mask &= _get_switch_mask(net, "dcline", "l", open_switches)
    for dcline in net.dcline[mask].itertuples():
        g.add_edge(pp_bus_mapping[dcline.from_bus],
                   pp_bus_mapping[dcline.to_bus],
                   weight=dcline_length_km)

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
    open_element_mask = np.isin(net[element].index, open_elements, invert=True)
    return open_element_mask


def coords_from_igraph(
        graph: "igraph.Graph",
        roots: Iterable,
        meshed: bool = False,
        calculate_meshed: bool = False
) -> list[list[float]]:
    """
    Create a list of generic coordinates from an igraph graph layout.

    Parameters:
        graph: The igraph graph on which the coordinates shall be based
        roots: The root buses of the graph
        meshed: determines if the graph has any meshes
        calculate_meshed: determines whether to calculate the meshed status

    Return:
        list of coordinates from the graph layout
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


def coords_from_nxgraph(
        mg: networkx.Graph = None,
        layout_engine: str = 'neato'
) -> list[list[float]]:
    """
    Create a list of generic coordinates from a networkx graph layout.

    Parameters:
        mg: The networkx graph on which the coordinates shall be based
        layout_engine: GraphViz Layout Engine for layouting a network. See https://graphviz.org/docs/layouts/

    Return:
         list of coordinates from the graph layout
    """
    # workaround for bug in agraph
    for u, v in mg.edges(data=False):
        if 'key' in mg[int(u)][int(v)]:
            del mg[int(u)][int(v)]['key']
        if 'key' in mg[int(u)][int(v)].get(0, ()):
            del mg[int(u)][int(v)][0]['key']
    # ToDo: Insert fallback layout for nxgraph
    return list(zip(*(list(nx.drawing.nx_agraph.graphviz_layout(mg, prog=layout_engine).values()))))


def create_generic_coordinates(
        net: pandapowerNet,
        mg: networkx.Graph = None,
        library: str = "igraph",
        respect_switches: bool = False,
        geodata_table: str = "bus",
        buses: Iterable[int] = None,
        overwrite: bool = False,
        layout_engine: str = 'neato',
        trafo_length_km: float = 0.01,
        switch_length_km: float = 0.001
) -> pandapowerNet:
    """
    This function will add arbitrary geo-coordinates for all buses based on an analysis of branches
    and rings. It will remove out of service buses/lines from the net. The coordinates will be
    created either by igraph or by using networkx library.

    Parameters:
        net: pandapower network
        mg: Existing networkx multigraph, if available. Convenience to save computation time.
        respect_switches: respect switches in a network for generic coordinates
        library: "igraph" to use igraph package or "networkx" to use networkx package
        geodata_table: table to write the generic geodatas to
        buses: buses for which generic geodata are created, all buses will be used by default
        overwrite: overwrite existing geodata
        layout_engine: GraphViz Layout Engine for layouting a network. See https://graphviz.org/docs/layouts/

    Return:
         the pandapower network with added geo coordinates for the buses.
         Does not copy the network, so the original network will be modified!

    Example:
        >>> net = create_generic_coordinates(net)
    """
    if buses is None:
        buses = net[geodata_table].index.tolist()
    _prepare_geodata_table(net, geodata_table, overwrite, buses)
    if library == "igraph":
        if not IGRAPH_INSTALLED:
            soft_dependency_error("build_igraph_from_pp()", "igraph")
        graph, meshed, roots = build_igraph_from_pp(net, respect_switches, buses=buses,
                                                    trafo_length_km=trafo_length_km, switch_length_km=switch_length_km)
        coords = coords_from_igraph(graph, roots, meshed)
    elif library == "networkx":
        if mg is None:
            nxg = create_nxgraph(net, respect_switches=respect_switches,
                                     include_out_of_service=True,
                                     trafo_length_km=trafo_length_km, switch_length_km=switch_length_km)
        else:
            nxg = copy.deepcopy(mg)
        coords = coords_from_nxgraph(nxg, layout_engine=layout_engine)
    else:
        raise ValueError("Unknown library %s - chose 'igraph' or 'networkx'" % library)
    if len(coords):
        geojson_strings: list[str] = list(
            map(lambda x: geojson.dumps(geojson.Point((x[1], x[0])), sort_keys=True), zip(*coords))
        )
        net[geodata_table].loc[buses, "geo"] = pd.Series(data=geojson_strings, index=buses)
    return net


def _prepare_geodata_table(
        net: pandapowerNet, geodata_table: str, overwrite: bool, elements: Iterable[int] | None
) -> None:
    if geodata_table not in net or "geo" not in net[geodata_table]:
        try:
            add_column_to_df(net, geodata_table, "geo")
        except KeyError as e:
            logger.warning("Creating geodata for a unknown table")
            if geodata_table not in net:
                net[geodata_table] = pd.DataFrame(columns=["geo"], index=elements, dtype=pd.StringDtype())
            else:
                net[geodata_table]["geo"] = pd.NA
    if elements is None:
        elements = net[geodata_table].index.tolist()
    try:
        net[geodata_table].loc[elements]
    except KeyError as e:
        logger.error(f"While preparing geodata table for {geodata_table} a nonexistent bus was passed!")
        raise e
    if net[geodata_table].loc[elements, "geo"].dropna().shape[0]:
        if overwrite:
            net[geodata_table].loc[elements, "geo"] = pd.NA
        else:
            raise UserWarning(f"Table {geodata_table} is not empty - use overwrite=True to overwrite existing geodata")


def fuse_geodata(net):
    mg = create_nxgraph(net, include_lines=False, include_impedances=False, respect_switches=False)
    geocoords = set(net.bus.dropna(subset=['geo']).index)
    for area in connected_components(mg):
        if len(area & geocoords) > 1:
            geo = net.bus.loc[list(area & geocoords), 'geo'].apply(geojson.loads)
            for bus in area:
                if len(geo) > 1:
                    coordinates = [point['coordinates'] for point in geo]
                    mean_lat = np.mean([coord[1] for coord in coordinates])
                    mean_lon = np.mean([coord[0] for coord in coordinates])
                else:
                    mean_lon, mean_lat = geo.coordinates
                geojson_string: str = geojson.dumps(geojson.Point((mean_lon, mean_lat)))
                net.bus.at[bus, "geo"] = geojson_string

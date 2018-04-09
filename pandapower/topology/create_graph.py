# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


from itertools import combinations

import networkx as nx

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def create_nxgraph(net, respect_switches=True, include_lines=True, include_trafos=True,
                   include_impedances=True, nogobuses=None, notravbuses=None, multi=True):
    """
     Converts a pandapower network into a NetworkX graph, which is a is a simplified representation
     of a network's topology, reduced to nodes and edges. Busses are being represented by nodes
     (Note: only buses with in_service = 1 appear in the graph), edges represent physical
     connections between buses (typically lines or trafos).

     INPUT:
        **net** (pandapowerNet) - variable that contains a pandapower network


     OPTIONAL:
        **respect_switches** (boolean, True) - True: open switches (line, trafo, bus) are being \
            considered (no edge between nodes)
            False: open switches are being ignored

        **include_lines** (boolean, True) - determines, whether lines get converted to edges

        **include_impedances** (boolean, True) - determines, whether per unit impedances
            (net.impedance) are converted to edges

        **include_trafos** (boolean, True) - determines, whether trafos get converted to edges

        **nogobuses** (integer/list, None) - nogobuses are not being considered in the graph

        **notravbuses** (integer/list, None) - lines connected to these buses are not being
            considered in the graph

        **multi** (boolean, True) - True: The function generates a NetworkX MultiGraph, which allows
            multiple parallel edges between nodes
            False: NetworkX Graph (no multiple parallel edges)

     OUTPUT:
        **mg** - Returns the required NetworkX graph

     EXAMPLE:
         import pandapower.topology as top

         mg = top.create_nx_graph(net, respect_switches = False)
         # converts the pandapower network "net" to a MultiGraph. Open switches will be ignored.

    """

    if multi:
        mg = nx.MultiGraph()
    else:
        mg = nx.Graph()
    mg.add_nodes_from(net.bus.index)
    if include_lines:
        # lines with open switches can be excluded
        nogolines = set(net.switch.element[
            (net.switch.et == "l") & (net.switch.closed == 0)]) if respect_switches else set()
        mg.add_edges_from((int(fb), int(tb), {"weight": float(l), "key": int(idx), "type": "l",
                                              "capacity": float(imax), "path": 1})
                          for fb, tb, l, idx, inservice, imax in
                          list(zip(net.line.from_bus, net.line.to_bus, net.line.length_km,
                                   net.line.index, net.line.in_service, net.line.max_i_ka))
                          if inservice == 1 and idx not in nogolines)

    if include_impedances:
        # due to changed behaviour: give a warning to the user
        if not include_lines and len(net.impedance) > 0:
            logger.warning('Change notice: per unit impedance elements are included in the graph, '
                           'even though lines are not. If this behaviour is undesired, set the '
                           'parameter "include_impedances" to False')

        mg.add_edges_from((int(fb), int(tb), {"weight": 0, "key": int(idx), "type": "i", "path": 1})
                          for fb, tb, idx, inservice in
                          list(zip(net.impedance.from_bus, net.impedance.to_bus,
                                   net.impedance.index, net.impedance.in_service))
                          if inservice == 1)

    if include_trafos:
        nogotrafos = set(net.switch.element[
            (net.switch.et == "t") & (net.switch.closed == 0)]) if respect_switches else set()
        mg.add_edges_from((int(hvb), int(lvb), {"weight": 0, "key": int(idx), "type": "t"})
                          for hvb, lvb, idx, inservice in
                          list(zip(net.trafo.hv_bus, net.trafo.lv_bus,
                                   net.trafo.index, net.trafo.in_service))
                          if inservice == 1 and idx not in nogotrafos)
        for trafo3, t3tab in net.trafo3w.iterrows():
            mg.add_edges_from((int(bus1), int(bus2), {"weight": 0, "key": int(trafo3),
                                                      "type": "t3"}) for bus1, bus2 in
                              combinations([t3tab.hv_bus,
                                            t3tab.mv_bus, t3tab.lv_bus], 2) if t3tab.in_service)
    if respect_switches:
        # add edges for closed bus-bus switches
        bs = net.switch[(net.switch.et == "b") & (net.switch.closed == 1)]
    else:
        # add edges for any bus-bus switches
        bs = net.switch[net.switch.et == "b"]
    mg.add_edges_from((int(b), int(e), {"weight": 0, "key": int(i), "type": "s"})
                      for b, e, i in list(zip(bs.bus, bs.element, bs.index)))

    # nogobuses are a nogo
    if nogobuses is not None:
        for b in nogobuses:
            mg.remove_node(b)
    if notravbuses is not None:
        for b in notravbuses:
            for i in list(mg[b].keys()):
                try:
                    del mg[b][i]  # networkx versions < 2.0
                except:
                    del mg._adj[b][i]  # networkx versions 2.0
    mg.remove_nodes_from(net.bus[~net.bus.in_service].index)
    return mg

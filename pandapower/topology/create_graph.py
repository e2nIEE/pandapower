from builtins import zip
# -*- coding: utf-8 -*-
__author__ = 'ascheidler, lthurner'

import networkx as nx
from itertools import combinations

def create_nxgraph(net, respect_switches=True, include_lines=True, include_trafos=True,
                   nogobusses=None, notravbusses=None, multi=True):
    """
     Converts a pandapower network into a NetworkX graph, which is a is a simplified representation 
     of a network's topology, reduced to nodes and edges. Busses are being represented by nodes 
     (Note: only busses with in_service = 1 appear in the graph), edges represent physical 
     connections between busses (typically lines or trafos). 
     
     INPUT:
     
        **net** (PandapowerNet) - variable that contains a pandapower network
         
         
     OPTIONAL:
     
        **respect_switches** (boolean, True) - True: open line switches are being considered
                                                     (no edge between nodes)
                                               False: open line switches are being ignored
                                                
        **include_lines** (boolean, True) - determines, whether lines get converted to edges
        
        **include_trafos** (boolean, True) - determines, whether trafos get converted to edges
        
        **nogobusses** (integer/list, None) - nogobusses are not being considered in the graph
        
        **notravbusses** (integer/list, None) - lines connected to these busses are not being
                                              considered in the graph
                                              
        **multi** (boolean, True) - True: The function generates a NetworkX MultiGraph, which allows
                                    multiple parallel edges between nodes                                    
                                    False: NetworkX Graph (no multiple parallel edges)
         
     RETURN:
     
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
    nogolines = {}
    mg.add_nodes_from(net.bus[net.bus.in_service==1].index)
    if include_lines:
        # lines with open switches can be excluded
        if respect_switches:
            nogolines = set(net.switch.element[(net.switch.et == "l") &
                                             (net.switch.closed == 0)])
        mg.add_edges_from((int(fb), int(tb), {"weight": float(l), "key": int(idx), "type": "l",
                                              "capacity": float(imax), "path": 1})
                             for fb, tb, l, idx, inservice, imax in
                             zip(net.line.from_bus, net.line.to_bus, net.line.length_km,
                                 net.line.index, net.line.in_service, net.line.imax_ka)
                             if inservice == 1 and not idx in nogolines)
        mg.add_edges_from((int(fb), int(tb), {"weight": 0, "key": int(idx), "type": "i",
                                              "path": 1})
                                              for fb, tb, idx, inservice in
                             zip(net.impedance.from_bus, net.impedance.to_bus, 
                                 net.impedance.index, net.impedance.in_service)
                             if inservice == 1)

    if include_trafos:
        nogotrafos = set(net.switch.element[(net.switch.et == "t") &
                                             (net.switch.closed == 0)])
        mg.add_edges_from((int(hvb), int(lvb), {"weight": 0, "key": int(idx), "type": "t"})
                             for hvb, lvb, idx, inservice in
                             zip(net.trafo.hv_bus, net.trafo.lv_bus,
                                 net.trafo.index, net.trafo.in_service)
                             if inservice == 1 and not idx in nogotrafos)
        for trafo3, t3tab in net.trafo3w.iterrows():
            mg.add_edges_from((int(bus1), int(bus2), {"weight": 0, "key": int(trafo3),
                                  "type": "t3"}) for bus1, bus2 in combinations([t3tab.hv_bus,
                                  t3tab.mv_bus, t3tab.lv_bus], 2) if t3tab.in_service)
    # add bus-bus switches
    bs = net.switch[(net.switch.et == "b") &
                    ((net.switch.closed == 1) | (not respect_switches))]
    mg.add_edges_from((int(b), int(e), {"weight": 0, "key": int(i), "type": "s"})
                       for b, e, i in zip(bs.bus, bs.element, bs.index))
    # nogobusses are a nogo
    if nogobusses is not None:
        for b in nogobusses:
            mg.remove_node(b)
    if notravbusses is not None:
        for b in notravbusses:
            for i in list(mg[b].keys()):
                del mg[b][i]
    return mg
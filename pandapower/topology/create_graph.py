# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import networkx as nx
import numpy as np
from itertools import combinations
try:
    import pplog as logging
except ImportError:
    import logging

from pandapower.pd2ppc import _init_ppc
from pandapower.auxiliary import _init_nx_options, _select_is_elements_numba 
from pandapower.idx_bus import BASE_KV

from pandapower.build_branch import _calc_impedance_parameters_from_dataframe,\
                                    _calc_branch_values_from_trafo_df, \
                                    _trafo_df_from_trafo3w
from pandapower.build_bus import _build_bus_ppc

INDEX = 0
F_BUS = 1
T_BUS = 2

WEIGHT = 0
BR_R = 1
BR_X = 2
BR_Z = 3

logger = logging.getLogger(__name__)


def create_nxgraph(net, respect_switches=True, include_lines=True,
                   include_trafos=True, include_impedances=True,
                   nogobuses=None, notravbuses=None, multi=True,
                   calc_z=False):
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

        **calc_r_ohm** (boolean, False) - True: The function calculates absolute resistance in Ohm
            and adds it as a weight to the graph
            False: All resistance weights are set to zero

        **calc_z_ohm** (boolean, False) - True: The function calculates magnitude of the impedance in Ohm
            and adds it as a weight to the graph
            False: All impedance weights are set to zero

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
    for b in net.bus.index:
        mg.add_node(b)
   
    open_sw = net.switch.closed.values == False   
    if calc_z:
        ppc = get_nx_ppc(net)        
        
    if include_lines:        
        line = net.line
        indices, parameter, in_service = init_par(line, calc_z)
        indices[:, F_BUS] = line.from_bus.values
        indices[:, T_BUS] = line.to_bus.values

        if respect_switches:
            mask = (net.switch.et.values == "l") & open_sw
            if mask.any():
                open_lines = net.switch.element.values[mask]
                open_lines_mask = np.in1d(indices[:, INDEX], open_lines)
                in_service &= ~open_lines_mask

        parameter[:, WEIGHT] = line.length_km.values
        if calc_z:
            line_length = line.length_km.values / line.parallel.values
            r = line.r_ohm_per_km.values * line_length
            x = line.x_ohm_per_km.values * line_length
            parameter[:, BR_R] = r
            parameter[:, BR_X] = x
            
        add_edges(mg, indices, parameter, in_service, net, "line", calc_z)

    if include_impedances and len(net.impedance):
        impedance = net.impedance
        indices, parameter, in_service = init_par(impedance, calc_z)
        indices[:, F_BUS] = impedance.from_bus.values
        indices[:, T_BUS] = impedance.to_bus.values
        
        if calc_z:
            baseR = get_baseR(net, ppc, impedance.from_bus.values)
            r, x, _, _ = _calc_impedance_parameters_from_dataframe(net)
            parameter[:, BR_R] = r * baseR
            parameter[:, BR_X] = x * baseR
            
        add_edges(mg, indices, parameter, in_service, net, "impedance", calc_z)
        
    if include_trafos:
        trafo = net.trafo
        if len(trafo.index):
            indices, parameter, in_service = init_par(trafo, calc_z)
            indices[:, F_BUS] = trafo.hv_bus.values
            indices[:, T_BUS] = trafo.lv_bus.values
    
            if respect_switches:
                mask = (net.switch.et.values == "t") & open_sw
                if mask.any():
                    open_trafos = net.switch.element.values[mask]
                    open_trafos_mask = np.in1d(indices[:, INDEX], open_trafos)
                    in_service &= ~open_trafos_mask

            if calc_z:
                baseR = get_baseR(net, ppc, trafo.hv_bus.values)
                r, x, _, _, _ = _calc_branch_values_from_trafo_df(net, ppc, trafo)
                parameter[:, BR_R] = r * baseR
                parameter[:, BR_X] = x * baseR
            
            add_edges(mg, indices, parameter, in_service, net, "trafo", calc_z)

        trafo3w = net.trafo3w
        if len(trafo3w):
            sides = ["hv", "mv", "lv"]
            if calc_z:     
                trafo_df = _trafo_df_from_trafo3w(net)
                r_all, x_all, _, _, _ = _calc_branch_values_from_trafo_df(net, ppc, trafo_df)
                r = {side: r for side, r in zip(sides, np.split(r_all, 3))}
                x = {side: x for side, x in zip(sides, np.split(x_all, 3))}
                baseR = get_baseR(net, ppc, trafo3w.hv_bus.values)
            open_switch = np.zeros(trafo3w.shape[0], dtype=bool)
            if respect_switches:
                #for trafo3ws the bus where the open switch is located also matters. Open switches
                #are therefore defined by the tuple (idx, b) where idx is the trafo3w index and b
                #is the bus. To make searching for the open 3w trafos a 1d problem, open 3w switches
                #are represented with imaginary numbers as idx + b*1j
                mask = (net.switch.et.values == "t3") & open_sw
                open_trafo3w_index = net.switch.element.values[mask]
                open_trafo3w_buses = net.switch.bus.values[mask]
                open_trafo3w = (open_trafo3w_index + open_trafo3w_buses*1j).flatten()
            for f, t in combinations(sides, 2):
                indices, parameter, in_service = init_par(trafo3w, calc_z)
                indices[:, F_BUS] = trafo3w["%s_bus"%f].values
                indices[:, T_BUS] = trafo3w["%s_bus"%t].values                
                if respect_switches and len(open_trafo3w):
                    for BUS in [F_BUS, T_BUS]:
                        open_switch = np.in1d(indices[:, INDEX] + indices[:, BUS]*1j, open_trafo3w)
                        in_service &= ~open_switch
                if calc_z:     
                    parameter[:, BR_R] = (r[f] + r[t]) * baseR
                    parameter[:, BR_X] = (x[f] + x[t]) * baseR
                        
                add_edges(mg, indices, parameter, in_service, net, "trafo3w", calc_z)


    switch = net.switch
    if len(switch):
        if respect_switches:
            # add edges for closed bus-bus switches
            in_service = (switch.et.values == "b") & ~open_sw
        else:
            # add edges for any bus-bus switches
            in_service = (switch.et.values == "b")
        indices, parameter = init_par(switch, calc_z)
        indices[:, F_BUS] = switch.bus.values
        indices[:, T_BUS] = switch.element.values
        add_edges(mg, indices, parameter, in_service, net, "switch", calc_z)
                
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
    for b in net.bus.index[~net.bus.in_service.values]:
        mg.remove_node(b)
    return mg


def add_edges(mg, indices, parameter, in_service, net, element, calc_z):
    if calc_z:
        parameter[:, BR_Z] = np.sqrt(parameter[:, BR_R]**2 + parameter[:, BR_X]**2)
    for idx, p in zip(indices[in_service], parameter[in_service]):
        if calc_z:
            weights = {"weight": p[WEIGHT], "path": 1, "r_ohm": p[BR_R], "x_ohm": p[BR_X],
                       "z_ohm": p[BR_Z]}
        else:
             weights = {"weight": p[WEIGHT], "path": 1}           
        mg.add_edge(idx[F_BUS], idx[T_BUS], key=(element, idx[INDEX]),  **weights)


def get_baseR(net, ppc, buses):
    bus_lookup = net._pd2ppc_lookups["bus"]
    base_kv = ppc["bus"][bus_lookup[buses], BASE_KV]
    return np.square(base_kv) / net.sn_mva
  
    
def init_par(tab, calc_z):
    n = tab.shape[0]
    indices = np.zeros((n, 3), dtype=np.int)
    indices[:, INDEX] = tab.index
    if calc_z:
        parameters = np.zeros((n, 4), dtype=np.float)
    else:
        parameters = np.zeros((n, 1), dtype=np.float)
    
    if "in_service" in tab:
        return indices, parameters, tab.in_service.values.copy()
    else:
        return indices, parameters
            

def get_nx_ppc(net):
    _init_nx_options(net)
    ppc = _init_ppc(net)
    _build_bus_ppc(net, ppc)
    return ppc
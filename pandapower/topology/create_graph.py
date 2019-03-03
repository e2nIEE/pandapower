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
from pandapower.auxiliary import _init_runpp_options, _select_is_elements_numba 
from pandapower.idx_brch import BR_R, BR_X, F_BUS, T_BUS
from pandapower.idx_bus import BASE_KV

from pandapower.build_branch import _calc_impedance_parameters_from_dataframe,\
                                    _calc_branch_values_from_trafo_df, \
                                    _trafo_df_from_trafo3w
from pandapower.build_bus import _build_bus_ppc
BR_Z = 4
WEIGHT = 5
INDEX = 6
logger = logging.getLogger(__name__)


def create_nxgraph(net, respect_switches=True, include_lines=True,
                   include_trafos=True, include_impedances=True,
                   nogobuses=None, notravbuses=None, multi=True,
                   add_impedances=False):
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
    mg.add_nodes_from(net.bus.index)
    
    if add_impedances:
        ppc = get_nx_ppc(net)
    if respect_switches:
        closed_sw = net.switch.closed.values == False
    if include_lines:        
        if respect_switches:
            mask = (net.switch.et.values == "l") & closed_sw
            nogolines = set(net.switch.element.values[mask])
        else:
            nogolines = None
        line = net.line
        edge_par = init_par(line)
        edge_par[:, F_BUS] = line.from_bus.values
        edge_par[:, T_BUS] = line.to_bus.values
        if add_impedances:
            line_length = line.length_km.values / line.parallel.values
            r = line.r_ohm_per_km.values * line_length
            x = line.x_ohm_per_km.values * line_length
            edge_par[:, BR_R] = r
            edge_par[:, BR_X] = x
        edge_par[:, WEIGHT] = line.length_km.values
        add_edges(mg, edge_par, net, "line", add_impedances, nogolines)

    if include_impedances and len(net.impedance):
        impedance = net.impedance
        edge_par = init_par(impedance)
        edge_par[:, F_BUS] = impedance.from_bus.values
        edge_par[:, T_BUS] = impedance.to_bus.values
        if add_impedances:
            baseR = get_baseR(net, ppc, impedance.bus.values)
            r, x, _, _ = _calc_impedance_parameters_from_dataframe(net)
            edge_par[:, BR_R] = r
            edge_par[:, BR_X] = x
        add_edges(mg, edge_par, net, "impedance", add_impedances)
        
    if include_trafos:
        trafo = net.trafo
        if len(trafo.index):
            if respect_switches:
                mask = (net.switch.et.values == "t") & closed_sw
                nogotrafos = set(net.switch.element.values[mask])
            else:
                nogotrafos = None
            edge_par = init_par(trafo)
            edge_par[:, F_BUS] = trafo.hv_bus.values
            edge_par[:, T_BUS] = trafo.lv_bus.values
            if add_impedances:
                baseR = get_baseR(net, ppc, trafo.hv_bus.values)
                r, x, _, _, _ = _calc_branch_values_from_trafo_df(net, ppc, trafo)
                edge_par[:, BR_R] = r * baseR
                edge_par[:, BR_X] = x * baseR
            add_edges(mg, edge_par, net, "trafo", add_impedances, nogotrafos)

        trafo3w = net.trafo3w
        if len(trafo3w):
            if respect_switches:
                mask = (net.switch.et.values == "t3") & closed_sw
                nogotrafo3w = net.switch[["element", "bus"]].values[mask].tolist()
            else:
                nogotrafo3w = None
            sides = ["hv", "mv", "lv"]
            if add_impedances:     
                trafo_df = _trafo_df_from_trafo3w(net)
                r_all, x_all, _, _, _ = _calc_branch_values_from_trafo_df(net, ppc, trafo_df)
                r = {side: r for side, r in zip(sides, np.split(r_all, 3))}
                x = {side: x for side, x in zip(sides, np.split(x_all, 3))}
                baseR = get_baseR(net, ppc, trafo3w.hv_bus.values)
            for f, t in combinations(sides, 2):
                edge_par = init_par(trafo3w)
                edge_par[:, F_BUS] = trafo3w["%s_bus"%f].values
                edge_par[:, T_BUS] = trafo3w["%s_bus"%t].values
                if add_impedances:     
                    edge_par[:, BR_R] = (r[f] + r[t]) * baseR
                    edge_par[:, BR_X] = (x[f] + x[t]) * baseR
                add_edges(mg, edge_par, net, "trafo3w", add_impedances, 
                          nogotrafo3w)
                
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
    mg.remove_nodes_from(net.bus.index[~net.bus.in_service.values])
    return mg

def add_edges(mg, parameters, net, element, add_impedances, nogo=None):
    parameters[:, BR_Z] = np.sqrt(parameters[:, BR_R]**2 + parameters[:, BR_X]**2)
    is_par = parameters[net[element].in_service.values]
    for p in is_par:
        fb = int(p[F_BUS])
        tb = int(p[T_BUS])
        index = int(p[INDEX])
        if nogo is not None:
            #for trafo3w the bus of the open switch is relevant
            if element == "trafo3w":
                #don't add an edge with an open switch
                if [index, fb] in nogo or [index, tb] in nogo:
                    continue        
            elif index in nogo:
                continue
        weights = {"weight": p[WEIGHT], "path": 1}
        if add_impedances:
            weights.update({"r_ohm": p[BR_R], "x_ohm": p[BR_X], "z_ohm": p[BR_Z]})   
        mg.add_edge(fb, tb, key=(element, index),  **weights)

def get_baseR(net, ppc, buses):
    bus_lookup = net._pd2ppc_lookups["bus"]
    base_kv = ppc["bus"][bus_lookup[buses], BASE_KV]
    return np.square(base_kv) / net.sn_mva
    
def init_par(tab):
    parameters = np.zeros((tab.shape[0], 8))
    parameters[:, INDEX] = tab.index
    return parameters
            

def get_nx_ppc(net):
    _init_runpp_options(net, algorithm='nr', calculate_voltage_angles="auto",
                        init="auto", max_iteration="auto", tolerance_mva=1e-8,
                        trafo_model="t", trafo_loading="current",
                        enforce_q_lims=False, check_connectivity=True,
                        voltage_depend_loads=True)
    net["_is_elements"] = _select_is_elements_numba(net)
    ppc = _init_ppc(net)
    _build_bus_ppc(net, ppc)
    return ppc

    
if __name__ == '__main__':
    import pandapower.networks as nw
    net = nw.case9241pegase()
    net = nw.mv_oberrhein(include_substations=True)
#    net = nw.example_simple()
#    pp.runpp(net)
#    net.line.at[0, "in_service"] = False

#    ppc = get_nx_ppc(net)
#    
#    line = net.line
#    r = line.r_ohm_per_km.values * line.length_km.values / line.parallel.values
#    x = line.x_ohm_per_km.values * line.length_km.values / line.parallel.values
#    
##    pp.runpp(net)
#
##    line = net.line.iloc[0]
##    print(mg.get_edge_data(line.from_bus, line.to_bus))
##    
##    trafo = net.trafo.iloc[0]
##    print(mg.get_edge_data(trafo.hv_bus, trafo.lv_bus))
#    
#    
    net = nw.example_multivoltage()
    trafo3 = net.trafo3w.iloc[0]
#    pp.create_switch(net, bus=trafo3.mv_bus, element=0, et="t3", closed=False)

    mg = create_nxgraph(net)
    print(mg.get_edge_data(trafo3.hv_bus, trafo3.lv_bus))

    print(mg.get_edge_data(trafo3.hv_bus, trafo3.mv_bus))

    print(mg.get_edge_data(trafo3.mv_bus, trafo3.lv_bus))
#    r, x, z = get_z_from_ppc(net)
#    f, t = net._pd2ppc_lookups["branch"]["trafo"]
#    pp.runpp(net)
#    print(net._ppc["branch"][f, BR_R])
#    print(net._ppc["branch"][f, BR_X])
#    max(r[f:t] - (net.line.length_km*net.line.r_ohm_per_km).values)

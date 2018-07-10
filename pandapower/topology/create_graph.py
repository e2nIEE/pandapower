# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import networkx as nx
import numpy as np

try:
    import pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def create_nxgraph(net, respect_switches=True, include_lines=True, include_trafos=True,
                   include_impedances=True, nogobuses=None, notravbuses=None, multi=True,
                   calc_r_ohm=False, calc_z_ohm=False):
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
    if include_lines:
        # lines with open switches can be excluded
        nogolines = set(net.switch.element[
            (net.switch.et == "l") & (net.switch.closed == 0)]) if respect_switches else set()

        line_edge_data=net.line[net.line.in_service & ~net.line.index.isin(nogolines)]

        if calc_r_ohm:
            line_edge_data['R_ohm']=net.line.r_ohm_per_km * net.line.length_km
        if calc_z_ohm:
            line_edge_data['Z_ohm']=np.sqrt((net.line.r_ohm_per_km * net.line.length_km) ** 2 + (net.line.x_ohm_per_km * net.line.length_km) ** 2)

        for line in line_edge_data.itertuples():
            weights={'weight': line.length_km, 'key_tmp': line.Index, 'type': 'l', 'path': 1}
            if calc_r_ohm:
                weights['R_ohm']=line.R_ohm
            if calc_z_ohm:
                weights['Z_ohm']=line.Z_ohm
            mg.add_edge(line.from_bus, line.to_bus, **weights)

    if include_impedances:
        # due to changed behaviour: give a warning to the user
        if not include_lines and len(net.impedance) > 0:
            logger.warning('Change notice: per unit impedance elements are included in the graph, '
                           'even though lines are not. If this behaviour is undesired, set the '
                           'parameter "include_impedances" to False')

        imp_edge_data=net.impedance[net.impedance.in_service]

        if calc_r_ohm or calc_z_ohm:
            base_Z=(net.sn_kva/1000) / net.bus.loc[imp_edge_data.from_bus].vn_kv.values ** 2
        if calc_r_ohm:
            imp_edge_data['R_ohm']=imp_edge_data.rft_pu.abs() / base_Z
        if calc_z_ohm:
            imp_edge_data['Z_ohm']=np.sqrt(imp_edge_data.rft_pu ** 2 + imp_edge_data.xft_pu ** 2) / base_Z

        for imp in imp_edge_data.itertuples():
            weights={'weight': 0, 'key_tmp': imp.Index, 'type': 'i', 'path': 1}
            if calc_r_ohm:
                weights['R_ohm']=imp.R_ohm
            if calc_z_ohm:
                weights['Z_ohm']=imp.Z_ohm
            mg.add_edge(imp.from_bus, imp.to_bus, **weights)

    if include_trafos:
        nogotrafos = set(net.switch.element[
            (net.switch.et == "t") & (net.switch.closed == 0)]) if respect_switches else set()

        trafo_edge_data=net.trafo[net.trafo.in_service & ~net.trafo.index.isin(nogotrafos)]

        if calc_r_ohm or calc_z_ohm:
            base_Z=(trafo_edge_data.sn_kva/1000) / (trafo_edge_data.vn_hv_kv ** 2) 
        if calc_r_ohm:
            trafo_edge_data['R_ohm']=(trafo_edge_data.vscr_percent/100) / base_Z
        if calc_z_ohm:
            trafo_edge_data['Z_ohm']=(trafo_edge_data.vsc_percent/100)  / base_Z

        for trafo in trafo_edge_data.itertuples():
            weights={'weight': 0, 'key_tmp': trafo.Index, 'type': 't'}
            if calc_r_ohm:
                weights['R_ohm']=trafo.R_ohm
            if calc_z_ohm:
                weights['Z_ohm']=trafo.Z_ohm
            mg.add_edge(trafo.hv_bus, trafo.lv_bus, **weights)

        #Three-winding transformers:
        trafo3w_edge_data=net.trafo3w[net.trafo3w.in_service]

        if calc_r_ohm or calc_z_ohm:
            base_Z_hv = (trafo3w_edge_data[['sn_hv_kva', 'sn_mv_kva']].min(axis=1)/1000) / (trafo3w_edge_data.vn_hv_kv ** 2)
            base_Z_mv = (trafo3w_edge_data[['sn_mv_kva', 'sn_lv_kva']].min(axis=1)/1000) / (trafo3w_edge_data.vn_hv_kv ** 2)
            base_Z_lv = (trafo3w_edge_data[['sn_hv_kva', 'sn_lv_kva']].min(axis=1)/1000) / (trafo3w_edge_data.vn_hv_kv ** 2) 
        if calc_r_ohm:
            trafo3w_edge_data['R_hv_ohm']= (trafo3w_edge_data.vscr_hv_percent/100) / base_Z_hv
            trafo3w_edge_data['R_mv_ohm']= (trafo3w_edge_data.vscr_mv_percent/100) / base_Z_mv
            trafo3w_edge_data['R_lv_ohm']= (trafo3w_edge_data.vscr_lv_percent/100) / base_Z_lv
        if calc_z_ohm:
            trafo3w_edge_data['Z_hv_ohm']= (trafo3w_edge_data.vsc_hv_percent/100) / base_Z_hv
            trafo3w_edge_data['Z_mv_ohm']= (trafo3w_edge_data.vsc_mv_percent/100) / base_Z_mv
            trafo3w_edge_data['Z_lv_ohm']= (trafo3w_edge_data.vsc_lv_percent/100) / base_Z_lv

        for trafo3w in trafo3w_edge_data.itertuples():
            weights_hv={'weight': 0, 'key_tmp': trafo3w.Index, 'type': 't3'}
            weights_mv={'weight': 0, 'key_tmp': trafo3w.Index, 'type': 't3'}
            weights_lv={'weight': 0, 'key_tmp': trafo3w.Index, 'type': 't3'}
            if calc_r_ohm:
                weights_hv['R_ohm']=trafo3w.R_hv_ohm
                weights_mv['R_ohm']=trafo3w.R_mv_ohm
                weights_lv['R_ohm']=trafo3w.R_lv_ohm
            if calc_z_ohm:
                weights_hv['Z_ohm']=trafo3w.Z_hv_ohm
                weights_mv['Z_ohm']=trafo3w.Z_mv_ohm
                weights_lv['Z_ohm']=trafo3w.Z_lv_ohm

            mg.add_edge(trafo3w.hv_bus, trafo3w.mv_bus, **weights_hv)
            mg.add_edge(trafo3w.mv_bus, trafo3w.lv_bus, **weights_mv)
            mg.add_edge(trafo3w.hv_bus, trafo3w.lv_bus, **weights_lv)

    if respect_switches:
        # add edges for closed bus-bus switches
        bs = net.switch[(net.switch.et == "b") & (net.switch.closed == 1)]
    else:
        # add edges for any bus-bus switches
        bs = net.switch[net.switch.et == "b"]

    for switch in bs.itertuples():
        weights={"weight": 0, "key_tmp": switch.Index, "type": "s"}
        if calc_r_ohm:
            weights['R_ohm']=0
        if calc_z_ohm:
            weights['Z_ohm']=0
        mg.add_edge(switch.bus, switch.element, **weights)

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

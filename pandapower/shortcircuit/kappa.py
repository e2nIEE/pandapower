# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import numpy as np
import networkx as nx

from pypower.idx_bus import GS, BS
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X

def calc_kappa(net):
    network_structure = net._options_sc["network_structure"]
    bus = net._is_elems["bus"]
    bus["kappa_korr"] = 1.
    if network_structure == "meshed":
        bus["kappa_korr"] = 1.15
    elif network_structure == "auto":
        mg = nxgraph_from_ppc(net)
        for bidx in net._is_elems["bus"].index:
            ppc_index = net._pd2ppc_lookups["bus"][bidx]
            paths = list(nx.all_simple_paths(mg, ppc_index, "earth"))
            if len(paths) > 1:
                bus.kappa_korr.at[bidx] = 1.15
                for path in paths:
                    r = sum([mg[b1][b2][0]["r"] for b1, b2 in zip(path, path[1:])])
                    x = sum([mg[b1][b2][0]["x"] for b1, b2 in zip(path, path[1:])])
                    if r / x < .3:                                     
                        bus.kappa_korr.at[bidx] = 1.
                        break           
    rx_equiv = np.real(bus.z_equiv) / np.imag(bus.z_equiv)
    kappa = 1.02 + .98 * np.exp(-3 * rx_equiv)
    bus["kappa"] = np.clip(bus.kappa_korr * kappa, 1, bus.kappa_max)
    
def nxgraph_from_ppc(net):
    ppc = net._ppc
    bus_lookup = net._pd2ppc_lookups["bus"]
    mg = nx.MultiGraph()
    mg.add_nodes_from(ppc["bus"][:, 0].astype(int))
    mg.add_edges_from((int(branch[T_BUS]), int(branch[F_BUS]),
                       {"r": branch[BR_R], "x": branch[BR_X]}) for branch in ppc["branch"].real)
    mg.add_node("earth")
    vs_buses_pp = list(set(net._is_elems["ext_grid"].bus.values) | set(net._is_elems["gen"].bus))
    vs_buses = bus_lookup[vs_buses_pp]
    z = 1 / (ppc["bus"][vs_buses, GS] + ppc["bus"][vs_buses, BS] * 1j)
    mg.add_edges_from(("earth", int(bus), {"r": z.real, "x": z.imag}) 
                        for bus, z in zip(vs_buses, z))
    return mg
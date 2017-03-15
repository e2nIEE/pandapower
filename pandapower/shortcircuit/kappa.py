# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import numpy as np
import networkx as nx

from pandapower.shortcircuit.idx_bus import GS, BS, BUS_I, KAPPA, C_MAX, C_MIN, BASE_KV, R_EQUIV, X_EQUIV
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X
from pandapower.shortcircuit.idx_bus import *

#def calc_kappa(net):
#    network_structure = net._options["network_structure"]
#    bus = net._is_elems["bus"]
#    bus["kappa_korr"] = 1.
#    if network_structure == "meshed":
#        bus["kappa_korr"] = 1.15
#    elif network_structure == "auto":
#        mg = nxgraph_from_ppc(net)
#        for bidx in net._is_elems["bus"].index:
#            ppc_index = net._pd2ppc_lookups["bus"][bidx]
#            paths = list(nx.all_simple_paths(mg, ppc_index, "earth"))
#            if len(paths) > 1:
#                bus.kappa_korr.at[bidx] = 1.15
#                for path in paths:
#                    r = sum([mg[b1][b2][0]["r"] for b1, b2 in zip(path, path[1:])])
#                    x = sum([mg[b1][b2][0]["x"] for b1, b2 in zip(path, path[1:])])
#                    if r / x < .3:                                     
#                        bus.kappa_korr.at[bidx] = 1.
#                        break           
#    rx_equiv = np.real(bus.z_equiv) / np.imag(bus.z_equiv)
#    kappa = 1.02 + .98 * np.exp(-3 * rx_equiv)
#    print(kappa, kappa_korr, kapp_max)
#    bus["kappa"] = np.clip(bus.kappa_korr * kappa, 1, bus.kappa_max)
    
def nxgraph_from_ppc(net, ppc):
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
    
def _add_kappa_to_ppc(net, ppc):
    if not net._options["kappa"]:
        return
    network_structure = net._options["network_structure"]
    kappa_max = np.full(ppc["bus"].shape[0], 2.)
    lv_buses = np.where(ppc["bus"][:, BASE_KV] < 1.)
    if len(lv_buses) > 0:
        kappa_max[lv_buses] = 1.8

    if network_structure == "meshed":
        kappa_korr = np.full(ppc["bus"].shape[0], 1.15)
    else:
        kappa_korr = np.full(ppc["bus"].shape[0], 1.)
    if network_structure == "auto":
        kappa_korr = np.full(ppc["bus"].shape[0], 1.)
        mg = nxgraph_from_ppc(net, ppc)
        for bidx in ppc["bus"][:, BUS_I]:
            paths = list(nx.all_simple_paths(mg, bidx, "earth"))
            if len(paths) > 1:
                kappa_korr[bidx] = 1.15
                for path in paths:
                    r = sum([mg[b1][b2][0]["r"] for b1, b2 in zip(path, path[1:])])
                    x = sum([mg[b1][b2][0]["x"] for b1, b2 in zip(path, path[1:])])
                    if r / x < .3:                                     
                        kappa_korr[bidx] = 1.
                        break           
    rx_equiv = ppc["bus"][:, R_EQUIV] / ppc["bus"][:, X_EQUIV]
    kappa = 1.02 + .98 * np.exp(-3 * rx_equiv)
    print(kappa, kappa_korr, kappa_max)
    ppc["bus"][:, KAPPA] = np.clip(kappa_korr * kappa, 1, kappa_max)
    
def _add_c_to_ppc(net, ppc):
    ppc["bus"][:, C_MAX] = 1.1
    ppc["bus"][:, C_MIN] = 1.
    lv_buses = np.where(ppc["bus"][:, BASE_KV] < 1.)
    if len(lv_buses) > 0:
        lv_tol_percent = net["_options"]["lv_tol_percent"]
        if lv_tol_percent==10:
            c_ns = 1.1
        elif lv_tol_percent==6:
            c_ns = 1.05
        else:
            raise ValueError("Voltage tolerance in the low voltage grid has" \
                                        " to be either 6% or 10% according to IEC 60909")
        ppc["bus"][lv_buses, C_MAX] = c_ns
        ppc["bus"][lv_buses, C_MIN] = .95
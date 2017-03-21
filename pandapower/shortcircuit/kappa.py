# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import numpy as np
import networkx as nx

from pandapower.shortcircuit.idx_bus import KAPPA, R_EQUIV, X_EQUIV
from pypower.idx_bus import BUS_I, BASE_KV, GS, BS
from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X

def _add_kappa_to_ppc(net, ppc):
    if not net._options["kappa"]:
        return
    topology = net._options["topology"]
    kappa_max = np.full(ppc["bus"].shape[0], 2.)
    lv_buses = np.where(ppc["bus"][:, BASE_KV] < 1.)
    if len(lv_buses) > 0:
        kappa_max[lv_buses] = 1.8

    if topology == "meshed":
        kappa_korr = np.full(ppc["bus"].shape[0], 1.15)
    else:
        kappa_korr = np.full(ppc["bus"].shape[0], 1.)
    if topology == "auto":
        kappa_korr = np.full(ppc["bus"].shape[0], 1.)
        mg = nxgraph_from_ppc(net, ppc)
        for bidx in ppc["bus"][:, BUS_I].astype(int):
            paths = list(nx.all_simple_paths(mg, bidx, "earth"))
            if len(paths) > 1:
                kappa_korr[bidx] = 1.15
                for path in paths:
                    r = sum([mg[b1][b2][0]["r"] for b1, b2 in zip(path, path[1:])])
                    x = sum([mg[b1][b2][0]["x"] for b1, b2 in zip(path, path[1:])])
                    if r / x < .3:                                     
                        kappa_korr[bidx] = 1.
                        break           
    rx_equiv = ppc["bus_sc"][:, R_EQUIV] / ppc["bus_sc"][:, X_EQUIV]
    kappa = 1.02 + .98 * np.exp(-3 * rx_equiv)
    ppc["bus_sc"][:, KAPPA] = np.clip(kappa_korr * kappa, 1, kappa_max)
   
def nxgraph_from_ppc(net, ppc):
    bus_lookup = net._pd2ppc_lookups["bus"]
    mg = nx.MultiGraph()
    mg.add_nodes_from(ppc["bus"][:, BUS_I].astype(int))
    mg.add_edges_from((int(branch[T_BUS]), int(branch[F_BUS]),
                       {"r": branch[BR_R], "x": branch[BR_X]}) for branch in ppc["branch"].real)
    mg.add_node("earth")
    vs_buses_pp = list(set(net._is_elems["ext_grid"].bus.values) | set(net._is_elems["gen"].bus))
    vs_buses = bus_lookup[vs_buses_pp]
    z = 1 / (ppc["bus"][vs_buses, GS] + ppc["bus"][vs_buses, BS] * 1j)
    mg.add_edges_from(("earth", int(bus), {"r": z.real, "x": z.imag}) 
                        for bus, z in zip(vs_buses, z))
    return mg
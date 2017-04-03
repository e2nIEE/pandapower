# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.
import copy
import networkx as nx
import numpy as np

from pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X
from pypower.idx_bus import BUS_I, GS, BS

from pandapower.shortcircuit.idx_bus import KAPPA, R_EQUIV, X_EQUIV
from pandapower.shortcircuit.impedance import _calc_equiv_sc_impedance

def _add_kappa_to_ppc(net, ppc):
    if not net._options["kappa"]:
        return
    ppc_20 = copy.deepcopy(ppc)
    y = 1 / (ppc_20["branch"][:, BR_R] + ppc_20["branch"][:, BR_X] * 1j)
    z = 1 / (y.real + y.imag * 1j * 2 / 5)
    ppc_20["branch"][:, BR_R] = z.real
    ppc_20["branch"][:, BR_X] = z.imag

#    ppc_20["branch"][:, BR_X] *= (5/2)
    ppc_20["bus"][:, BS] *= (2/5)
    _calc_equiv_sc_impedance(net, ppc_20)
    rx_equiv_20 = ppc_20["bus_sc"][:, R_EQUIV] / ppc_20["bus_sc"][:, X_EQUIV] * 2 / 5
    rx_equiv_50 = ppc["bus_sc"][:, R_EQUIV] / ppc["bus_sc"][:, X_EQUIV]
    print(rx_equiv_20 - rx_equiv_50)
    print(ppc["internal"]["Ybus"] - ppc_20["internal"]["Ybus"])
    print(ppc["internal"]["zbus"] - ppc_20["internal"]["zbus"])

    ppc["bus_sc"][:, KAPPA] = _kappa(rx_equiv_20)

def _kappa(rx):
    return 1.02 + .98 * np.exp(-3 * rx)

def nxgraph_from_ppc(net, ppc):
    bus_lookup = net._pd2ppc_lookups["bus"]
    mg = nx.MultiGraph()
    mg.add_nodes_from(ppc["bus"][:, BUS_I].astype(int))
    mg.add_edges_from((int(branch[T_BUS]), int(branch[F_BUS]),
                       {"r": branch[BR_R], "x": branch[BR_X]}) for branch in ppc["branch"].real)
    mg.add_node("earth")
    vs_buses_pp = list(set(net._is_elements["ext_grid"].bus.values)|set(net._is_elements["gen"].bus))
    vs_buses = bus_lookup[vs_buses_pp]
    z = 1 / (ppc["bus"][vs_buses, GS] + ppc["bus"][vs_buses, BS] * 1j)
    mg.add_edges_from(("earth", int(bus), {"r": z.real, "x": z.imag})
                        for bus, z in zip(vs_buses, z))
    return mg
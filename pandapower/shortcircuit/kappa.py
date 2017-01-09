# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import numpy as np
import networkx as nx

import pandapower.topology as top


def calc_kappa(net, network_structure):
    net.bus["kappa_max"] = 2.
    net.bus["kappa_korr"] = 1.
    if network_structure == "meshed":
        net.bus["kappa_korr"] = 1.15
    elif network_structure == "auto":
        for bus, bustab in net.bus[net.bus.in_service==True].iterrows():
            paths = []
            mg = top.create_nxgraph(net, multi=False)
            for i, eg in net.ext_grid.iterrows():
                for p in nx.all_simple_paths(mg, bus, eg.bus):
                    paths.append((i,p))
            if len(paths) > 1:
                net.bus.kappa_korr.at[bus] = 1.15
                for eg, path in paths:
                    r = net.ext_grid.r.at[eg]
                    x = net.ext_grid.x.at[eg]
                    lines = top.elements_on_path(mg, path, "l", multi=False)
                    r += net.line.r.loc[lines].sum()
                    x += net.line.x.loc[lines].sum()
                    trafos = top.elements_on_path(mg, path, "t", multi=False)
                    r += net.trafo.r.loc[trafos].sum()
                    x += net.trafo.x.loc[trafos].sum()
                    if r / x < .3:                                     
                        net.bus.kappa_korr.at[bus] = 1.
                        break                
    rx_equiv = np.real(net.bus.z_equiv) / np.imag(net.bus.z_equiv)
    kappa = 1.02 + .98 * np.exp(-3 * rx_equiv)
    net.bus["kappa"] = np.clip(net.bus.kappa_korr * kappa, 1, net.bus.kappa_max)
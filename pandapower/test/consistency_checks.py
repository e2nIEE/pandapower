# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

from numpy import allclose
import pandas as pd
import pandapower as pp

def runpp_with_consistency_checks(net, **kwargs):
    pp.runpp(net, **kwargs)
    indices_consistent(net)
    branch_loss_consistent_with_bus_feed_in(net)
    element_power_consistent_with_bus_power(net)
    return True
    
def indices_consistent(net):
    for element in ["bus", "load", "ext_grid", "sgen", "trafo", "trafo3w", "line", "shunt", 
                    "ward", "xward", "impedance", "gen"]:
        e_idx = net[element].index
        res_idx = net["res_" + element].index
        assert len(e_idx) == len(res_idx), "length of %s bus and res_%s indices do not match"%(element, element)
        assert all(e_idx == res_idx), "%s bus and res_%s indices do not match"%(element, element)


def branch_loss_consistent_with_bus_feed_in(net):
    """
    The surpluss of bus feed summed over all buses always has to be equal to the sum of losses in
    all branches.
    """
    # Active Power
    bus_surplus_p = -net.res_bus.p_kw.sum()
    bus_surplus_q = -net.res_bus.q_kvar.sum()

    branch_loss_p = net.res_line.pl_kw.sum() + net.res_trafo.pl_kw.sum() + \
                    net.res_trafo3w.pl_kw.sum() + net.res_impedance.pl_kw.sum() 
    branch_loss_q = net.res_line.ql_kvar.sum() + net.res_trafo.ql_kvar.sum() + \
                    net.res_trafo3w.ql_kvar.sum() + net.res_impedance.ql_kvar.sum()                     
    assert allclose(bus_surplus_p, branch_loss_p)
    assert allclose(bus_surplus_q, branch_loss_q)


def element_power_consistent_with_bus_power(net):
    """
    The bus feed-in at each node has to be equal to the sum of the element feed ins at each node.
    """
    bus_p = pd.Series(data=0, index=net.bus.index, dtype=float)
    bus_q = pd.Series(data=0, index=net.bus.index, dtype=float)

    for idx, tab in net.ext_grid.iterrows():
        if tab.in_service:
            bus_p.at[tab.bus] += net.res_ext_grid.p_kw.at[idx]
            bus_q.at[tab.bus] += net.res_ext_grid.q_kvar.at[idx]

    for idx, tab in net.gen.iterrows():
        if tab.in_service:
            bus_p.at[tab.bus] += net.res_gen.p_kw.at[idx]
            bus_q.at[tab.bus] += net.res_gen.q_kvar.at[idx]

    for idx, tab in net.load.iterrows():
        bus_p.at[tab.bus] += net.res_load.p_kw.at[idx]
        bus_q.at[tab.bus] += net.res_load.q_kvar.at[idx]

    for idx, tab in net.sgen.iterrows():
        bus_p.at[tab.bus] += net.res_sgen.p_kw.at[idx]
        bus_q.at[tab.bus] += net.res_sgen.q_kvar.at[idx]

    for idx, tab in net.shunt.iterrows():
        bus_p.at[tab.bus] += net.res_shunt.p_kw.at[idx]
        bus_q.at[tab.bus] += net.res_shunt.q_kvar.at[idx]

    for idx, tab in net.ward.iterrows():
        bus_p.at[tab.bus] += net.res_ward.p_kw.at[idx]
        bus_q.at[tab.bus] += net.res_ward.q_kvar.at[idx]

    for idx, tab in net.xward.iterrows():
        bus_p.at[tab.bus] += net.res_xward.p_kw.at[idx]
        bus_q.at[tab.bus] += net.res_xward.q_kvar.at[idx]

    assert allclose(net.res_bus.p_kw.values, bus_p.values, equal_nan=True)
    assert allclose(net.res_bus.q_kvar.values, bus_q.values, equal_nan=True)

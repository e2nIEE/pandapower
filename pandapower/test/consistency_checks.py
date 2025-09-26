# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pandas as pd
from numpy import allclose, isclose
import numpy as np

from pandapower import runpp, rundcpp, runpp_pgm
from pandapower.build_branch import _calc_tap_from_dataframe
from pandapower.pf.runpp_3ph import runpp_3ph
from pandapower.results import get_relevant_elements

phases = ["a", "b", "c"]


def runpp_with_consistency_checks(net, **kwargs):
    runpp(net, **kwargs)
    consistency_checks(net)
    return True

def runpp_3ph_with_consistency_checks(net, **kwargs):
    runpp_3ph(net, **kwargs)
    consistency_checks_3ph(net)
    return True

def rundcpp_with_consistency_checks(net, **kwargs):
    rundcpp(net, **kwargs)
    consistency_checks(net, test_q=False)
    return True

def runpp_pgm_with_consistency_checks(net):
    runpp_pgm(net, error_tolerance_vm_pu=1e-11, symmetric=True)
    consistency_checks(net)
    return True

def runpp_pgm_3ph_with_consistency_checks(net):
    runpp_pgm(net, error_tolerance_vm_pu=1e-11, symmetric=False)
    consistency_checks_3ph(net)
    return True


def consistent_b2b_vsc(net, rtol):
    pass


def consistency_checks(net, rtol=1e-3, test_q=True):
    indices_consistent(net)
    branch_loss_consistent_with_bus_feed_in(net, rtol)
    element_power_consistent_with_bus_power(net, rtol, test_q)
    # consistent_b2b_vsc(net, rtol)  # todo


def indices_consistent(net):
    elements = get_relevant_elements()
    for element in elements:
        e_idx = net[element].index
        res_idx = net["res_" + element].index
        assert len(e_idx) == len(res_idx), "length of %s bus and res_%s indices do not match"%(element, element)
        assert all(e_idx == res_idx), "%s bus and res_%s indices do not match"%(element, element)


def branch_loss_consistent_with_bus_feed_in(net, atol=1e-2):
    """
    The surpluss of bus feed summed over all buses always has to be equal to the sum of losses in
    all branches.
    """
    # Active Power
    bus_surplus_p = -net.res_bus.p_mw.sum()
    bus_surplus_q = -net.res_bus.q_mvar.sum()
    bus_dc_surplus_p = -net.res_bus_dc.p_mw.sum()

    branch_loss_p = net.res_line.pl_mw.values.sum() + net.res_trafo.pl_mw.values.sum() + \
                    net.res_trafo3w.pl_mw.values.sum() + net.res_impedance.pl_mw.values.sum() + \
                    net.res_dcline.pl_mw.values.sum() + net.res_tcsc.pl_mw.values.sum()
    branch_loss_q = net.res_line.ql_mvar.values.sum() + net.res_trafo.ql_mvar.values.sum() + \
                    net.res_trafo3w.ql_mvar.values.sum() + net.res_impedance.ql_mvar.values.sum() + \
                    net.res_dcline.q_to_mvar.values.sum() + net.res_dcline.q_from_mvar.values.sum() + \
                    net.res_tcsc.ql_mvar.values.sum()
    branch_dc_loss = net.res_line_dc.pl_mw.values.sum()

    try:
        assert isclose(bus_surplus_p, branch_loss_p, atol=atol)
    except AssertionError:
        raise AssertionError("Branch losses are %.4f MW, but power generation at the buses exceeds the feedin by %.4f MW"%(branch_loss_p, bus_surplus_p))
    try:
        assert isclose(bus_dc_surplus_p, branch_dc_loss, atol=atol)
    except AssertionError:
        raise AssertionError("DC branch losses are %.4f MW, but power generation at the DC buses exceeds the feedin by %.4f MW"%(branch_dc_loss, bus_dc_surplus_p))
    try:
        assert isclose(bus_surplus_q, branch_loss_q, atol=atol)
    except AssertionError:
        raise AssertionError("Branch losses are %.4f MVar, but power generation at the buses exceeds the feedin by %.4f MVar"%(branch_loss_q, bus_surplus_q))


def element_power_consistent_with_bus_power(net, rtol=1e-2, test_q=True):
    """
    The bus feed-in at each node has to be equal to the sum of the element feed ins at each node.
    """
    bus_p = pd.Series(data=0., index=net.bus.index)
    bus_q = pd.Series(data=0., index=net.bus.index)
    bus_p_dc = pd.Series(data=0., index=net.bus_dc.index)

    for idx, tab in net.ext_grid.iterrows():
        if tab.in_service:
            bus_p.at[tab.bus] -= net.res_ext_grid.p_mw.at[idx]
            bus_q.at[tab.bus] -= net.res_ext_grid.q_mvar.at[idx]

    for idx, tab in net.gen.iterrows():
        if tab.in_service:
            bus_p.at[tab.bus] -= net.res_gen.p_mw.at[idx]
            bus_q.at[tab.bus] -= net.res_gen.q_mvar.at[idx]

    for idx, tab in net.load.iterrows():
        bus_p.at[tab.bus] += net.res_load.p_mw.at[idx]
        bus_q.at[tab.bus] += net.res_load.q_mvar.at[idx]

    for idx, tab in net.sgen.iterrows():
        bus_p.at[tab.bus] -= net.res_sgen.p_mw.at[idx]
        bus_q.at[tab.bus] -= net.res_sgen.q_mvar.at[idx]

    for idx, tab in net.asymmetric_load.iterrows():
        bus_p.at[tab.bus] += net.res_asymmetric_load.p_mw.at[idx]
        bus_q.at[tab.bus] += net.res_asymmetric_load.q_mvar.at[idx]

    for idx, tab in net.asymmetric_sgen.iterrows():
        bus_p.at[tab.bus] -= net.res_asymmetric_sgen.p_mw.at[idx]
        bus_q.at[tab.bus] -= net.res_asymmetric_sgen.q_mvar.at[idx]

    for idx, tab in net.storage.iterrows():
        bus_p.at[tab.bus] += net.res_storage.p_mw.at[idx]
        bus_q.at[tab.bus] += net.res_storage.q_mvar.at[idx]

    for idx, tab in net.shunt.iterrows():
        bus_p.at[tab.bus] += net.res_shunt.p_mw.at[idx]
        bus_q.at[tab.bus] += net.res_shunt.q_mvar.at[idx]

    for idx, tab in net.ward.iterrows():
        bus_p.at[tab.bus] += net.res_ward.p_mw.at[idx]
        bus_q.at[tab.bus] += net.res_ward.q_mvar.at[idx]

    for idx, tab in net.xward.iterrows():
        bus_p.at[tab.bus] += net.res_xward.p_mw.at[idx]
        bus_q.at[tab.bus] += net.res_xward.q_mvar.at[idx]

    for idx, tab in net.svc.iterrows():
        bus_q.at[tab.bus] += net.res_svc.q_mvar.at[idx]

    for idx, tab in net.ssc.iterrows():
        bus_q.at[tab.bus] += net.res_ssc.q_mvar.at[idx]

    for idx, tab in net.vsc.iterrows():
        bus_p.at[tab.bus] += net.res_vsc.p_mw.at[idx]
        bus_q.at[tab.bus] += net.res_vsc.q_mvar.at[idx]
        bus_p_dc.at[tab.bus_dc] += net.res_vsc.p_dc_mw.at[idx]

    for idx, tab in net.b2b_vsc.iterrows():
        bus_p.at[tab.bus] += net.res_b2b_vsc.p_mw.at[idx]
        bus_q.at[tab.bus] += net.res_b2b_vsc.q_mvar.at[idx]
        bus_p_dc.at[tab.bus_dc_plus] += net.res_b2b_vsc.p_dc_mw_p.at[idx]
        bus_p_dc.at[tab.bus_dc_minus] += net.res_b2b_vsc.p_dc_mw_m.at[idx]

    assert allclose(net.res_bus.p_mw.values, bus_p.values, equal_nan=True, rtol=rtol)
    assert allclose(net.res_bus_dc.p_mw.values, bus_p_dc.values, equal_nan=True, rtol=rtol)
    if test_q:
        assert allclose(net.res_bus.q_mvar.values, bus_q.values, equal_nan=True, rtol=rtol)


def consistency_checks_3ph(net, rtol=2e-3):
    indices_consistent_3ph(net)
    branch_loss_consistent_with_bus_feed_in_3ph(net, rtol)
    element_power_consistent_with_bus_power_3ph(net, rtol)
    trafo_currents_consistent_3ph(net)

def indices_consistent_3ph(net):
    elements = get_relevant_elements("pf_3ph")
    for element in elements:
        e_idx = net[element].index
        res_idx = net["res_" + element+"_3ph"].index
        assert len(e_idx) == len(res_idx), "length of %s bus and res_%s indices do not match"%(element, element)
        assert all(e_idx == res_idx), "%s bus and res_%s indices do not match"%(element, element)


def branch_loss_consistent_with_bus_feed_in_3ph(net, atol=1e-2):
    """
    The surpluss of bus feed summed over all buses always has to be equal to the sum of losses in
    all branches.
    """
    bus_surplus_p = -net.res_bus_3ph[["p_a_mw", "p_b_mw", "p_c_mw"]].sum().sum()
    bus_surplus_q = -net.res_bus_3ph[["q_a_mvar", "q_b_mvar", "q_c_mvar"]].sum().sum()


    branch_loss_p = net.res_line_3ph.pl_a_mw.sum() + net.res_trafo_3ph.pl_a_mw.sum() + \
                    net.res_line_3ph.pl_b_mw.sum() + net.res_trafo_3ph.pl_b_mw.sum() + \
                    net.res_line_3ph.pl_c_mw.sum() + net.res_trafo_3ph.pl_c_mw.sum()

    branch_loss_q = net.res_line_3ph.ql_a_mvar.sum() + net.res_trafo_3ph.ql_a_mvar.sum() + \
                    net.res_line_3ph.ql_b_mvar.sum() + net.res_trafo_3ph.ql_b_mvar.sum() + \
                    net.res_line_3ph.ql_c_mvar.sum() + net.res_trafo_3ph.ql_c_mvar.sum()

    try:
        assert isclose(bus_surplus_p, branch_loss_p, atol=atol)
    except AssertionError:
        raise AssertionError("Branch losses are %.4f MW, but power generation at the buses exceeds the feedin by %.4f MW"%(branch_loss_p, bus_surplus_p))
    try:
        assert isclose(bus_surplus_q, branch_loss_q, atol=atol)
    except AssertionError:
        raise AssertionError("Branch losses are %.4f MVar, but power generation at the buses exceeds the feedin by %.4f MVar"%(branch_loss_q, bus_surplus_q))


def element_power_consistent_with_bus_power_3ph(net, rtol=1e-2):
    """
    The bus feed-in at each node has to be equal to the sum of the element feed ins at each node.
    """
    bus_p_a = pd.Series(data=0., index=net.bus.index)
    bus_q_a = pd.Series(data=0., index=net.bus.index)
    bus_p_b = pd.Series(data=0., index=net.bus.index)
    bus_q_b = pd.Series(data=0., index=net.bus.index)
    bus_p_c = pd.Series(data=0., index=net.bus.index)
    bus_q_c = pd.Series(data=0., index=net.bus.index)

    for idx, tab in net.ext_grid.iterrows():
        bus_p_a.at[tab.bus] -= net.res_ext_grid_3ph.p_a_mw.at[idx]
        bus_q_a.at[tab.bus] -= net.res_ext_grid_3ph.q_a_mvar.at[idx]
        bus_p_b.at[tab.bus] -= net.res_ext_grid_3ph.p_b_mw.at[idx]
        bus_q_b.at[tab.bus] -= net.res_ext_grid_3ph.q_b_mvar.at[idx]
        bus_p_c.at[tab.bus] -= net.res_ext_grid_3ph.p_c_mw.at[idx]
        bus_q_c.at[tab.bus] -= net.res_ext_grid_3ph.q_c_mvar.at[idx]

    for idx, tab in net.load.iterrows():
        bus_p_a.at[tab.bus] += net.res_load_3ph.p_mw.at[idx]/3
        bus_q_a.at[tab.bus] += net.res_load_3ph.q_mvar.at[idx] /3
        bus_p_b.at[tab.bus] += net.res_load_3ph.p_mw.at[idx]/3
        bus_q_b.at[tab.bus] += net.res_load_3ph.q_mvar.at[idx] /3
        bus_p_c.at[tab.bus] += net.res_load_3ph.p_mw.at[idx]/3
        bus_q_c.at[tab.bus] += net.res_load_3ph.q_mvar.at[idx] /3

    for idx, tab in net.asymmetric_load.iterrows():
        bus_p_a.at[tab.bus] += net.res_asymmetric_load_3ph.p_a_mw.at[idx]
        bus_q_a.at[tab.bus] += net.res_asymmetric_load_3ph.q_a_mvar.at[idx]
        bus_p_b.at[tab.bus] += net.res_asymmetric_load_3ph.p_b_mw.at[idx]
        bus_q_b.at[tab.bus] += net.res_asymmetric_load_3ph.q_b_mvar.at[idx]
        bus_p_c.at[tab.bus] += net.res_asymmetric_load_3ph.p_c_mw.at[idx]
        bus_q_c.at[tab.bus] += net.res_asymmetric_load_3ph.q_c_mvar.at[idx]

    for idx, tab in net.asymmetric_sgen.iterrows():
        bus_p_a.at[tab.bus] -= net.res_asymmetric_sgen_3ph.p_a_mw.at[idx]
        bus_q_a.at[tab.bus] -= net.res_asymmetric_sgen_3ph.q_a_mvar.at[idx]
        bus_p_b.at[tab.bus] -= net.res_asymmetric_sgen_3ph.p_b_mw.at[idx]
        bus_q_b.at[tab.bus] -= net.res_asymmetric_sgen_3ph.q_b_mvar.at[idx]
        bus_p_c.at[tab.bus] -= net.res_asymmetric_sgen_3ph.p_c_mw.at[idx]
        bus_q_c.at[tab.bus] -= net.res_asymmetric_sgen_3ph.q_c_mvar.at[idx]

    for idx, tab in net.sgen.iterrows():
        bus_p_a.at[tab.bus] -= net.res_sgen_3ph.p_mw.at[idx] / 3
        bus_q_a.at[tab.bus] -= net.res_sgen_3ph.q_mvar.at[idx] / 3
        bus_p_b.at[tab.bus] -= net.res_sgen_3ph.p_mw.at[idx] / 3
        bus_q_b.at[tab.bus] -= net.res_sgen_3ph.q_mvar.at[idx] / 3
        bus_p_c.at[tab.bus] -= net.res_sgen_3ph.p_mw.at[idx] / 3
        bus_q_c.at[tab.bus] -= net.res_sgen_3ph.q_mvar.at[idx] / 3

    assert allclose(net.res_bus_3ph.p_a_mw.values, bus_p_a.values, equal_nan=True, rtol=rtol)
    assert allclose(net.res_bus_3ph.q_a_mvar.values, bus_q_a.values, equal_nan=True, rtol=rtol)
    assert allclose(net.res_bus_3ph.p_b_mw.values, bus_p_b.values, equal_nan=True, rtol=rtol)
    assert allclose(net.res_bus_3ph.q_b_mvar.values, bus_q_b.values, equal_nan=True, rtol=rtol)
    assert allclose(net.res_bus_3ph.p_c_mw.values, bus_p_c.values, equal_nan=True, rtol=rtol)
    assert allclose(net.res_bus_3ph.q_c_mvar.values, bus_q_c.values, equal_nan=True, rtol=rtol)

def get_trafo_s_3ph(net, tf_index, side):
    s = np.array([(net.res_trafo_3ph.at[tf_index, "p_"+ph+"_"+side+"_mw"]+1j *
                   net.res_trafo_3ph.at[tf_index, "q_"+ph+"_"+side+"_mvar"])
                   for ph in phases
                ])
    if side == "lv":
        s = -s
    return s

def get_trafo_v_3ph(net, tf_index, side):
    bus_id = net.trafo.at[tf_index, side+"_bus"]
    v = np.array([(net.res_bus_3ph.at[bus_id, "vm_"+ph+"_pu"] *
                   net.bus.vn_kv[bus_id] *
                   np.exp(1j * np.deg2rad(
                       net.res_bus_3ph.at[bus_id, "va_"+ph+"_degree"])
                          )
                   )
                  for ph in phases
                ]) / np.sqrt(3)
    return v

def get_trafo_currents_3ph(net, tf_index, side):
    s = get_trafo_s_3ph(net, tf_index, side)
    v = get_trafo_v_3ph(net, tf_index, side)
    i = np.conjugate(s / v)
    return i

def check_dyn_transformer_currents(i_hv, i_lv, ratio, shift_degree, rtol):
    # HV and LV Currents are related depending on clock (shift)
    # Dyn11
    if shift_degree == -30:
        assert isclose(i_hv[0], (i_lv[0] - i_lv[2])/(ratio * np.sqrt(3)), rtol)
        assert isclose(i_hv[1], (i_lv[1] - i_lv[0])/(ratio * np.sqrt(3)), rtol)
        assert isclose(i_hv[2], (i_lv[2] - i_lv[1])/(ratio * np.sqrt(3)), rtol)
    # Dyn1
    elif shift_degree == 30:
        assert isclose(i_hv[0], (i_lv[0] - i_lv[1])/(ratio * np.sqrt(3)), rtol)
        assert isclose(i_hv[1], (i_lv[1] - i_lv[2])/(ratio * np.sqrt(3)), rtol)
        assert isclose(i_hv[2], (i_lv[2] - i_lv[0])/(ratio * np.sqrt(3)), rtol)
    # Dyn5
    elif shift_degree == 150:
        assert isclose(i_hv[0], (i_lv[2] - i_lv[0])/(ratio * np.sqrt(3)), rtol)
        assert isclose(i_hv[1], (i_lv[0] - i_lv[1])/(ratio * np.sqrt(3)), rtol)
        assert isclose(i_hv[2], (i_lv[1] - i_lv[2])/(ratio * np.sqrt(3)), rtol)
    # Dyn7
    elif (shift_degree == 210) or (shift_degree == -150):
        assert isclose(i_hv[0], (i_lv[1] - i_lv[0])/(ratio * np.sqrt(3)), rtol)
        assert isclose(i_hv[1], (i_lv[2] - i_lv[1])/(ratio * np.sqrt(3)), rtol)
        assert isclose(i_hv[2], (i_lv[0] - i_lv[2])/(ratio * np.sqrt(3)), rtol)

def check_ynyn_traformer_currents(i_hv, i_lv, ratio, shift_degree, rtol):
    # YNyn0
    if shift_degree == 0:
        assert isclose(i_hv[0], i_lv[0] / ratio, rtol)
        assert isclose(i_hv[1], i_lv[1] / ratio, rtol)
        assert isclose(i_hv[2], i_lv[2] / ratio, rtol)
    # YNyn6
    if (shift_degree == 180) or (shift_degree == -180):
        assert isclose(i_hv[0], -i_lv[0] / ratio, rtol)
        assert isclose(i_hv[1], -i_lv[1] / ratio, rtol)
        assert isclose(i_hv[2], -i_lv[2] / ratio, rtol)

def trafo_currents_consistent_3ph(net):
    """
    The HV and LV currents of the transformer has to be related in accordance with trafo vector_group and clock
    """
    rtol = 1e-1
    if "vector_group" not in net.trafo:
        return
    for vector_group, trafo_df in net.trafo.groupby('vector_group'):
        if vector_group not in ["Dyn", "YNd", "YNyn", "Yzn"]:
            continue
        ### Yzn need to be implemented
        if vector_group == "Yzn":
            continue
        ###############################
        vnh, vnl, _ = _calc_tap_from_dataframe(net, trafo_df)
        ratio = vnh / vnl
        for index, trafo in trafo_df.iterrows():
            i_hv = get_trafo_currents_3ph(net, index, "hv")
            i_lv = get_trafo_currents_3ph(net, index, "lv")
            tf_index = net.trafo.index.get_loc(index)
            if vector_group == "Dyn":
                check_dyn_transformer_currents(i_hv, i_lv, ratio[tf_index], trafo["shift_degree"], rtol)

            if vector_group == "YNyn":
                check_ynyn_traformer_currents(i_hv, i_lv, ratio[tf_index], trafo["shift_degree"], rtol)

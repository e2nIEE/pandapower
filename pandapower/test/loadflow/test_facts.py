# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os
import pytest
import copy
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from itertools import product

import pandapower as pp
from pandapower.test import assert_res_equal
from pandapower.test.consistency_checks import runpp_with_consistency_checks
from pandapower.pf.makeYbus_facts import calc_y_svc_pu
from pandapower.auxiliary import _preserve_dtypes

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False


def plot_z(baseZ, xl, xc):
    if not MATPLOTLIB_INSTALLED:
        raise ImportError('matplotlib must be installed to run plot_z()')

    x = np.arange(90, 181)
    y = calc_y_svc_pu(np.deg2rad(x), xl / baseZ, xc / baseZ)
    z = (1 / y) * baseZ
    plt.plot(x, z)


def xtcr(x, x_l):
    return np.pi * x_l / (2 * (np.pi - x) + np.sin(2 * x))


def xtcsc(x, x_l, x_c):
    return np.pi * x_l / (2 * (np.pi - x) + np.sin(2 * x) + np.pi * x_l / x_c)


def copy_with_impedance(net):
    baseMVA = net.sn_mva  # MVA
    baseV = net.bus.vn_kv.values  # kV
    baseI = baseMVA / (baseV * np.sqrt(3))
    baseZ = baseV ** 2 / baseMVA

    net_ref = copy.deepcopy(net)
    for i in net.tcsc.index.values:
        pp.create_impedance(net_ref, net.tcsc.from_bus.at[i], net.tcsc.to_bus.at[i], 0,
                            net.res_tcsc.x_ohm.at[i] / baseZ[net.tcsc.to_bus.at[i]], baseMVA,
                            in_service=net.tcsc.in_service.at[i], name="tcsc")
    net_ref.tcsc.in_service = False

    if len(net.svc) > 0:
        # pp.create_loads(net_ref, net.svc.bus.values, 0, net.res_svc.q_mvar.values, in_service=net.svc.in_service.values)
        # create shunts because of Ybus comparison
        q = np.square(net.bus.loc[net.svc.bus.values, 'vn_kv']) / net.res_svc.x_ohm.values
        pp.create_shunts(net_ref, net.svc.bus.values, q.fillna(0), in_service=net.svc.in_service.values, name="svc")
        net_ref.svc.in_service = False

    if len(net.ssc) > 0:
        # create shunts because of Ybus comparison
        in_service = net.ssc.in_service.values
        ssc_bus = net.ssc.bus.values
        aux_bus = pp.create_buses(net_ref, len(net.ssc), net.bus.loc[ssc_bus, "vn_kv"].values)
        for fb, tb, r, x, i in zip(ssc_bus, aux_bus, net.ssc.r_ohm.values / baseZ[ssc_bus],
                                   net.ssc.x_ohm.values / baseZ[ssc_bus], in_service):
            pp.create_impedance(net_ref, fb, tb, r, x, baseMVA, name="ssc", in_service=i)
        if len(net.res_ssc) > 0:
            vm_pu = net.res_ssc.vm_internal_pu.fillna(1)
        else:
            vm_pu = net.ssc.set_vm_pu.fillna(1)
        pp.create_gens(net_ref, aux_bus, 0, vm_pu, in_service=in_service)
        net_ref.ssc.in_service = False

    if len(net.vsc) > 0:
        # create shunts because of Ybus comparison
        in_service = net.vsc.in_service.values
        vsc_bus = net.vsc.bus.values
        aux_bus = pp.create_buses(net_ref, len(net.vsc), net.bus.loc[vsc_bus, "vn_kv"].values)
        for fb, tb, r, x, i in zip(vsc_bus, aux_bus, net.vsc.r_ohm.values / baseZ[vsc_bus],
                                   net.vsc.x_ohm.values / baseZ[vsc_bus], in_service):
            pp.create_impedance(net_ref, fb, tb, r, x, baseMVA, name="vsc", in_service=i)
        g = pp.create_gens(net_ref, aux_bus, 0, net.res_vsc.vm_internal_pu.fillna(1), in_service=in_service)
        vsc_pv = net.vsc.control_mode_ac.values == "vm_pu"
        net_ref.gen.loc[g[vsc_pv], "vm_pu"] = net.vsc.loc[vsc_pv, "control_value_ac"].values
        net_ref.vsc.in_service = False
        net_ref.bus_dc.loc[net.vsc.bus_dc.values, "in_service"] = False

    return net_ref


def compare_tcsc_impedance(net, net_ref, idx_tcsc, idx_impedance):
    backup_q = net_ref.res_bus.loc[net.ssc.bus.values, "q_mvar"].copy()
    net_ref.res_bus.loc[net.ssc.bus.values, "q_mvar"] += net_ref.res_impedance.loc[
        net_ref.impedance.query("name=='ssc'").index, "q_from_mvar"].values
    bus_idx = net.bus.index.values
    for col in ("vm_pu", "va_degree", "p_mw", "q_mvar"):
        assert np.allclose(net.res_bus[col], net_ref.res_bus.loc[bus_idx, col], rtol=0, atol=1e-6)
    assert np.allclose(net.res_ext_grid.p_mw, net_ref.res_ext_grid.p_mw, rtol=0, atol=1e-6)
    assert np.allclose(net.res_ext_grid.q_mvar, net_ref.res_ext_grid.q_mvar, rtol=0, atol=1e-6)

    for col in "p_from_mw", "q_from_mvar", "p_to_mw", 'q_to_mvar':
        assert np.allclose(net.res_tcsc.loc[idx_tcsc, col], net_ref.res_impedance.loc[idx_impedance, col],
                           rtol=0, atol=1e-6)
    assert np.allclose(net._ppc["internal"]["Ybus"].toarray(), net_ref._ppc["internal"]["Ybus"].toarray(), rtol=0,
                       atol=1e-6)
    net_ref.res_bus.loc[net.ssc.bus.values, "q_mvar"] = backup_q


def compare_ssc_impedance_gen(net, net_ref, element="ssc"):
    backup_q = net_ref.res_bus.loc[net[element].bus.values, "q_mvar"].copy()
    ### comparing the original buses in net and net_ref(witout the auxilary buses)
    net_ref.res_bus.loc[net[element].bus.values, "q_mvar"] += \
        net_ref.res_impedance.loc[net_ref.impedance.query(f"name=='{element}'").index, "q_from_mvar"].values
    assert np.allclose(np.abs(net._ppc["internal"]["V"][net.bus.index]),
                       np.abs(net_ref._ppc["internal"]["V"][net.bus.index]), rtol=0, atol=1e-6)

    for col in net.res_bus.columns:
        assert np.allclose(net.res_bus[col][net.bus.index], net_ref.res_bus[col][net.bus.index], rtol=0, atol=1e-6)

    ### compare the internal bus at ssc and the auxilary bus at net_ref
    in_service = net[element].in_service.values
    for i, j in zip(['vm_internal_pu', 'va_internal_degree'], ['vm_pu', 'va_degree']):
        assert np.allclose(net[f"res_{element}"][i][in_service],
                           net_ref.res_bus[j][len(net.bus):].values[in_service], rtol=0, atol=1e-6)

    assert np.allclose(np.abs(net._ppc["internal"]["V"][len(net.bus):]),
                       net_ref.res_bus.vm_pu[len(net.bus):][in_service], rtol=0, atol=1e-6)
    assert np.allclose(np.angle(net._ppc["internal"]["V"][len(net.bus):], deg=True),
                       net_ref.res_bus.va_degree[len(net.bus):][in_service], rtol=0, atol=1e-6)

    # compare ext_grid_result
    for col in net.res_ext_grid.columns:
        assert np.allclose(net.res_ext_grid[col][net.ext_grid.index], net_ref.res_ext_grid[col][net.ext_grid.index],
                           rtol=0, atol=1e-6)

    # compare line results
    ###
    for col in net.res_line.columns:
        assert np.allclose(net.res_line[col][net.line.index], net_ref.res_line[col][net.line.index], rtol=0, atol=1e-6)

    assert np.allclose(net._ppc["internal"]["Ybus"].toarray(), net_ref._ppc["internal"]["Ybus"].toarray(), rtol=0,
                       atol=1e-6)
    net_ref.res_bus.loc[net[element].bus.values, "q_mvar"] = backup_q


def add_tcsc_to_line(net, xl, xc, set_p_mw, from_bus, line, side="from_bus"):
    aux = pp.create_bus(net, net.bus.at[from_bus, "vn_kv"], "aux")
    net.line.loc[line, side] = aux

    idx = pp.create_tcsc(net, from_bus, aux, xl, xc, set_p_mw, 100, controllable=True)
    return idx


def facts_case_study_grid():
    net = pp.create_empty_network()

    b1 = pp.create_bus(net, name="B1", vn_kv=18)
    b2 = pp.create_bus(net, name="B2", vn_kv=16.5)
    b3 = pp.create_bus(net, name="B3", vn_kv=230)
    b4 = pp.create_bus(net, name="B4", vn_kv=230)
    b5 = pp.create_bus(net, name="B5", vn_kv=230)
    b6 = pp.create_bus(net, name="B6", vn_kv=230)
    b7 = pp.create_bus(net, name="B7", vn_kv=230)
    b8 = pp.create_bus(net, name="B8", vn_kv=230)

    pp.create_ext_grid(net, bus=b1, vm_pu=1, va_degree=0)

    pp.create_line_from_parameters(net, name="L1", from_bus=b3, to_bus=b4, length_km=30, r_ohm_per_km=0.049,
                                   x_ohm_per_km=0.136, g_us_per_km=0, c_nf_per_km=142, max_i_ka=1.5)
    pp.create_line_from_parameters(net, name="L2", from_bus=b3, to_bus=b4, length_km=30, r_ohm_per_km=0.049,
                                   x_ohm_per_km=0.136, g_us_per_km=0, c_nf_per_km=142, max_i_ka=1.5)
    pp.create_line_from_parameters(net, name="L3", from_bus=b4, to_bus=b5, length_km=100, r_ohm_per_km=0.081,
                                   x_ohm_per_km=0.312, g_us_per_km=0, c_nf_per_km=11, max_i_ka=1.5)
    pp.create_line_from_parameters(net, name="L4", from_bus=b4, to_bus=b6, length_km=100, r_ohm_per_km=0.081,
                                   x_ohm_per_km=0.312, g_us_per_km=0, c_nf_per_km=11, max_i_ka=1.5)
    pp.create_line_from_parameters(net, name="L5", from_bus=b5, to_bus=b7, length_km=220, r_ohm_per_km=0.081,
                                   x_ohm_per_km=0.312, g_us_per_km=0, c_nf_per_km=11, max_i_ka=1.5)
    pp.create_line_from_parameters(net, name="L6", from_bus=b6, to_bus=b8, length_km=140, r_ohm_per_km=0.081,
                                   x_ohm_per_km=0.312, g_us_per_km=0, c_nf_per_km=11, max_i_ka=1.5)
    pp.create_line_from_parameters(net, name="L7", from_bus=b5, to_bus=b6, length_km=180, r_ohm_per_km=0.081,
                                   x_ohm_per_km=0.312, g_us_per_km=0, c_nf_per_km=11, max_i_ka=1.5)
    pp.create_line_from_parameters(net, name="L8", from_bus=b7, to_bus=b8, length_km=180, r_ohm_per_km=0.081,
                                   x_ohm_per_km=0.312, g_us_per_km=0, c_nf_per_km=11, max_i_ka=1.5)

    # pp.create_line_from_parameters(net,name="L9",from_bus=3,to_bus=4,length_km=100, r_ohm_per_km=0.312,
    # x_ohm_per_km=0.312,g_us_per_km=0,c_nf_per_km=11)

    pp.create_transformer_from_parameters(net, name="trafo1", hv_bus=b8, lv_bus=b1, sn_mva=192, vn_hv_kv=230,
                                          vn_lv_kv=18, vkr_percent=0, vector_group="Yy0", pfe_kw=0, vk_percent=12,
                                          i0_percent=0)
    pp.create_transformer_from_parameters(net, name="trafo2", hv_bus=b3, lv_bus=b2, sn_mva=500, vn_hv_kv=230,
                                          vn_lv_kv=16.5, vkr_percent=0, vector_group="Yy0", pfe_kw=0, vk_percent=16,
                                          i0_percent=0)

    pp.create_gen(net, bus=b2, p_mw=500, vm_pu=1)
    # pp.create_sgen(net,bus = 2, p_mw=500,name="WT")
    #
    pp.create_load(net, bus=b4, p_mw=130, q_mvar=50)
    pp.create_load(net, bus=b5, p_mw=120, q_mvar=50)
    pp.create_load(net, bus=b6, p_mw=80, q_mvar=25)
    pp.create_load(net, bus=b7, p_mw=50, q_mvar=25)

    return net


@pytest.mark.parametrize("vm_set_pu", [0.96, 1., 1.04])
def test_svc(vm_set_pu):
    net = pp.networks.case9()
    net3 = copy.deepcopy(net)

    pp.create_svc(net, bus=3, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=90)
    net2 = copy.deepcopy(net)
    runpp_with_consistency_checks(net)
    assert 90 <= net.res_svc.at[0, "thyristor_firing_angle_degree"] <= 180
    assert np.isclose(net.res_bus.at[3, 'vm_pu'], net.svc.at[0, 'set_vm_pu'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_svc.q_mvar.at[0],
                      np.square(net.res_svc.vm_pu.at[0] * net.bus.at[net.svc.bus.at[0], 'vn_kv']) /
                      net.res_svc.x_ohm.at[0])

    # try writing the values to a load and see if the effect is the same:
    lidx = pp.create_load(net3, 3, 0, 0)
    net3.load.loc[lidx, "q_mvar"] = net.res_svc.q_mvar.at[0]
    pp.runpp(net3)
    assert np.isclose(net3.res_bus.at[3, 'vm_pu'], net.svc.at[0, 'set_vm_pu'], rtol=0, atol=1e-6)
    assert np.isclose(net3.res_bus.at[3, 'va_degree'], net.res_svc.at[0, 'va_degree'], rtol=0, atol=1e-6)
    assert np.isclose(net3.res_bus.at[3, 'q_mvar'], net.res_bus.at[3, 'q_mvar'], rtol=0, atol=1e-6)

    net3.load.at[3, "in_service"] = False
    baseZ = np.square(net.bus.at[3, 'vn_kv']) / net.sn_mva
    x_pu = net.res_svc.x_ohm.at[0] / baseZ
    pp.create_shunt(net3, 3, 0)
    # net3.shunt.at[0, "q_mvar"] = x_pu * net.sn_mva
    net3.shunt.at[0, "q_mvar"] = net.res_svc.q_mvar.at[0]
    net3.shunt.at[0, "vn_kv"] = net.bus.at[3, 'vn_kv'] * vm_set_pu
    pp.runpp(net3)
    assert np.isclose(net3.res_bus.at[3, 'vm_pu'], net.svc.at[0, 'set_vm_pu'], rtol=0, atol=1e-6)
    assert np.isclose(net3.res_bus.at[3, 'va_degree'], net.res_svc.at[0, 'va_degree'], rtol=0, atol=1e-6)
    assert np.isclose(net3.res_bus.at[3, 'q_mvar'], net.res_bus.at[3, 'q_mvar'], rtol=0, atol=1e-6)

    net2.svc.at[0, "thyristor_firing_angle_degree"] = net.res_svc.thyristor_firing_angle_degree.at[0]
    net2.svc.at[0, "controllable"] = False
    pp.runpp(net2)
    assert np.isclose(net2.res_bus.at[3, 'vm_pu'], net.svc.at[0, 'set_vm_pu'], rtol=0, atol=1e-6)
    assert np.isclose(net2.res_bus.at[3, 'q_mvar'], net.res_bus.at[3, 'q_mvar'], rtol=0, atol=1e-6)
    assert np.isclose(net2.res_svc.at[0, 'vm_pu'], net.res_svc.at[0, 'vm_pu'], rtol=0, atol=1e-6)
    assert np.isclose(net2.res_svc.at[0, 'va_degree'], net.res_svc.at[0, 'va_degree'], rtol=0, atol=1e-6)
    assert np.isclose(net2.res_svc.at[0, 'q_mvar'], net.res_svc.at[0, 'q_mvar'], rtol=0, atol=1e-6)
    assert np.isclose(net2.res_svc.at[0, 'x_ohm'], net.res_svc.at[0, 'x_ohm'], rtol=0, atol=1e-6)

    # pp.create_svc(net, 6, 1, -10, vm_set_pu, net.res_svc.thyristor_firing_angle_degree.at[0], controllable=False)
    # pp.create_svc(net, 7, 1, -10, vm_set_pu, 90, controllable=True)
    # runpp_with_consistency_checks(net)


@pytest.mark.parametrize("vm_set_pu", [0.96, 1., 1.04])
def test_2_svcs(vm_set_pu):
    net = pp.create_empty_network()
    pp.create_buses(net, 3, 110)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 0, 2, 20, 0.0487, 0.13823, 160, 0.664)

    # both not controllable
    net1 = copy.deepcopy(net)
    pp.create_svc(net1, bus=1, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145,
                  controllable=False)
    pp.create_svc(net1, bus=2, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145,
                  controllable=False)
    pp.runpp(net1)
    net2 = copy.deepcopy(net)
    pp.create_loads(net2, [1, 2], 0, net1.res_svc.q_mvar.values)
    pp.runpp(net2)
    assert_frame_equal(net1.res_bus, net2.res_bus)

    # first controllable
    net1 = copy.deepcopy(net)
    pp.create_svc(net1, bus=1, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145)
    pp.create_svc(net1, bus=2, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145,
                  controllable=False)
    pp.runpp(net1)
    net2 = copy.deepcopy(net)
    pp.create_loads(net2, [1, 2], 0, net1.res_svc.q_mvar.values)
    pp.runpp(net2)
    assert_frame_equal(net1.res_bus, net2.res_bus)

    # second controllable
    net1 = copy.deepcopy(net)
    pp.create_svc(net1, bus=1, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145,
                  controllable=False)
    pp.create_svc(net1, bus=2, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145)
    pp.runpp(net1)
    net2 = copy.deepcopy(net)
    pp.create_loads(net2, [1, 2], 0, net1.res_svc.q_mvar.values)
    pp.runpp(net2)
    assert_frame_equal(net1.res_bus, net2.res_bus)

    # both controllable
    net1 = copy.deepcopy(net)
    pp.create_svc(net1, bus=1, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145)
    pp.create_svc(net1, bus=2, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145)
    pp.runpp(net1)
    net2 = copy.deepcopy(net)
    pp.create_loads(net2, [1, 2], 0, net1.res_svc.q_mvar.values)
    pp.runpp(net2)
    assert_frame_equal(net1.res_bus, net2.res_bus)

    # connected at ext_grid_bus - does not work
    # pp.create_svc(net1, bus=0, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=1, thyristor_firing_angle_degree=145, controllable=False)
    # pp.runpp(net1)


def test_tcsc_simple():
    net = pp.create_empty_network()
    pp.create_buses(net, 2, 110)
    pp.create_ext_grid(net, 0)
    # pp.create_line_from_parameters(net, 0, 1, 100, 0.0487, 0.13823, 160, 0.664)
    pp.create_load(net, 1, 100, 25)
    pp.create_tcsc(net, 0, 1, 1, -10, -100, 140, controllable=False)

    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net.impedance.index)

    net.tcsc.controllable = True
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net.impedance.index)


def test_tcsc_simple1():
    net = pp.create_empty_network()
    pp.create_buses(net, 3, 110)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 100, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 2, 1, 100, 0.0487, 0.13823, 160, 0.664)
    pp.create_load(net, 1, 100, 25)
    pp.create_tcsc(net, 0, 2, 1, -10, 6, 144, controllable=False)

    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net.impedance.index)

    net.tcsc.controllable = True
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net.impedance.index)


def test_tcsc_simple2():
    net = pp.create_empty_network()
    pp.create_buses(net, 3, 110)
    pp.create_ext_grid(net, 0)
    # pp.create_line_from_parameters(net, 0, 1, 100, 0.0487, 0.13823, 160, 0.664)
    pp.create_load(net, 1, 40, 25)
    pp.create_load(net, 2, 60, 25)
    pp.create_tcsc(net, 0, 1, 1, -10, -40, 140, controllable=False)
    pp.create_tcsc(net, 0, 2, 1, -10, -60, 140, controllable=False)

    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)

    net.tcsc.at[0, "controllable"] = True
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)

    net.tcsc.at[1, "controllable"] = True
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)


def test_tcsc_simple3():
    baseMVA = 100  # MVA
    baseV = 110  # kV
    baseI = baseMVA / (baseV * np.sqrt(3))
    baseZ = baseV ** 2 / baseMVA
    xl = 0.2
    xc = -20
    # plot_z(baseZ, xl, xc)

    # (0)-------------(1)-----------------(3)->
    #                  |--(TCSC)--(2)------|

    net = pp.create_empty_network(sn_mva=baseMVA)
    pp.create_buses(net, 4, baseV)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 3, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 2, 3, 20, 0.0487, 0.13823, 160, 0.664)

    pp.create_load(net, 3, 100, 40)

    pp.create_tcsc(net, 1, 2, xl, xc, 5, 170, "Test", controllable=True, min_angle_degree=90, max_angle_degree=180)

    runpp_with_consistency_checks(net, init="dc")

    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)

    # todo:
    #  test with distributed slack
    #  test results by comparing impedance result to formula; p, q, i by comparing to line results; vm, va by comparing to bus results


def test_tcsc_simple3_slack():
    baseMVA = 100  # MVA
    baseV = 110  # kV
    baseI = baseMVA / (baseV * np.sqrt(3))
    baseZ = baseV ** 2 / baseMVA
    xl = 0.2
    xc = -20
    # plot_z(baseZ, xl, xc)

    # (0)-------------(1)-----------------(3)->
    #  |-----(TCSC)--(2)-------------------|

    net = pp.create_empty_network(sn_mva=baseMVA)
    pp.create_buses(net, 4, baseV)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 3, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 2, 3, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 20, 0.0487, 0.13823, 160, 0.664)

    pp.create_load(net, 2, 100, 40)

    # pp.plotting.simple_plot(net)
    pp.create_tcsc(net, 2,0, xl, xc, 5, 170, "Test", controllable=True, min_angle_degree=90, max_angle_degree=180)

    runpp_with_consistency_checks(net, init="dc")

    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)

    # todo:
    #  test with distributed slack
    #  test results by comparing impedance result to formula; p, q, i by comparing to line results; vm, va by comparing to bus results


def test_compare_to_impedance():
    baseMVA = 100  # MVA
    baseV = 110  # kV
    baseI = baseMVA / (baseV * np.sqrt(3))
    baseZ = baseV ** 2 / baseMVA
    xl = 0.2
    xc = -20
    # plot_z(baseZ, xl, xc)

    # (0)-------------(1)-----------------(3)->
    #                  |--(TCSC)--(2)------|

    net = pp.create_empty_network(sn_mva=baseMVA)
    pp.create_buses(net, 4, baseV)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 3, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 2, 3, 20, 0.0487, 0.13823, 160, 0.664)

    pp.create_load(net, 3, 100, 40)

    net_ref = copy.deepcopy(net)

    pp.create_tcsc(net, 1, 2, xl, xc, -20, 170, "Test", controllable=True, min_angle_degree=90, max_angle_degree=180)

    runpp_with_consistency_checks(net, init="dc")

    pp.create_impedance(net_ref, 1, 2, 0, net.res_tcsc.x_ohm.at[0] / baseZ, baseMVA)

    pp.runpp(net_ref)

    # compare when controllable
    compare_tcsc_impedance(net, net_ref, 0, 0)
    assert np.allclose(net._ppc["internal"]["J"].toarray()[:-1, :-1], net_ref._ppc["internal"]["J"].toarray(), rtol=0,
                       atol=5e-5)
    assert np.allclose(net._ppc["internal"]["Ybus"].toarray(), net_ref._ppc["internal"]["Ybus"].toarray(), rtol=0,
                       atol=1e-6)

    # compare when not controllable
    net.tcsc.thyristor_firing_angle_degree = net.res_tcsc.thyristor_firing_angle_degree
    net.tcsc.controllable = False
    runpp_with_consistency_checks(net, init="dc")

    compare_tcsc_impedance(net, net_ref, 0, 0)
    assert np.allclose(net._ppc["internal"]["J"].toarray(), net_ref._ppc["internal"]["J"].toarray(), rtol=0, atol=5e-5)
    assert np.allclose(net._ppc["internal"]["Ybus"].toarray(), net_ref._ppc["internal"]["Ybus"].toarray(), rtol=0,
                       atol=1e-6)


def test_tcsc_case_study():
    net = facts_case_study_grid()
    baseMVA = net.sn_mva
    baseV = 230
    baseZ = baseV ** 2 / baseMVA
    xl = 0.2
    xc = -20
    # plot_z(baseZ, xl, xc)
    f = net.bus.loc[net.bus.name == "B4"].index.values[0]
    t = net.bus.loc[net.bus.name == "B6"].index.values[0]
    aux = pp.create_bus(net, 230, "aux")
    l = net.line.loc[(net.line.from_bus == f) & (net.line.to_bus == t)].index.values[0]
    net.line.loc[l, "from_bus"] = aux

    net_ref = copy.deepcopy(net)

    pp.create_tcsc(net, f, aux, xl, xc, -100, 100, controllable=True)
    pp.runpp(net, init="dc")

    pp.create_impedance(net_ref, f, aux, 0, net.res_tcsc.at[0, "x_ohm"] / baseZ, baseMVA)
    pp.runpp(net_ref)

    compare_tcsc_impedance(net, net_ref, 0, 0)
    assert np.allclose(net._ppc["internal"]["Ybus"].toarray(),
                       net_ref._ppc["internal"]["Ybus"].toarray(), rtol=0, atol=1e-6)


def test_multiple_facts():
    #                  |--(TCSC)--(4)------|
    # (0)-------------(1)-----------------(3)--------(6)
    #                  |-(5)-(TCSC)--(2)---|#
    # unfortunately, TCSC is not implemented for the case when multiple TCSC elements
    # have the same from_bus or to_bus
    baseMVA = 100
    baseV = 110
    baseZ = baseV ** 2 / baseMVA
    xl = 0.2
    xc = -15

    net = pp.create_empty_network(sn_mva=baseMVA)
    pp.create_buses(net, 7, baseV)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 3, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 2, 3, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 4, 3, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 5, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 3, 6, 20, 0.0487, 0.13823, 160, 0.664)

    pp.create_load(net, 3, 100, 40)

    pp.create_tcsc(net, 5, 2, xl, xc, -10, 140, "Test", controllable=True, min_angle_degree=90, max_angle_degree=180)
    pp.create_tcsc(net, 1, 4, xl, xc, -5, 140, "Test", controllable=True, min_angle_degree=90, max_angle_degree=180)

    runpp_with_consistency_checks(net)

    # net = _many_tcsc_test_net()

    net.tcsc.loc[1, "thyristor_firing_angle_degree"] = net.res_tcsc.loc[1, "thyristor_firing_angle_degree"]
    net.tcsc.loc[1, "controllable"] = False
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.query("name=='tcsc'").index)

    pp.create_svc(net, 3, 1, -10, 1.01, 90)
    runpp_with_consistency_checks(net)

    net.svc.at[0, "thyristor_firing_angle_degree"] = net.res_svc.loc[0, "thyristor_firing_angle_degree"]
    net.svc.controllable = False
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.query("name=='tcsc'").index)
    compare_tcsc_impedance(net, net_ref, net.svc.index, net_ref.impedance.query("name=='svc'").index)


def _many_tcsc_test_net():
    #                  |--(TCSC)--(4)------|
    # (0)-------------(1)-----------------(3)--------(6)
    #                  |-(5)-(TCSC)--(2)---|#
    # unfortunately, TCSC is not implemented for the case when multiple TCSC elements
    # have the same from_bus or to_bus
    baseMVA = 100
    baseV = 110
    baseZ = baseV ** 2 / baseMVA
    xl = 0.2
    xc = -15

    net = pp.create_empty_network(sn_mva=baseMVA)
    pp.create_buses(net, 7, baseV)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 3, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 2, 3, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 4, 3, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 5, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 3, 6, 20, 0.0487, 0.13823, 160, 0.664)

    pp.create_load(net, 3, 100, 40)

    pp.create_tcsc(net, 5, 2, xl, xc, -10, 160, "Test", controllable=True, min_angle_degree=90,
                   max_angle_degree=180)
    pp.create_tcsc(net, 1, 4, xl, xc, -5, 160, "Test", controllable=True, min_angle_degree=90,
                   max_angle_degree=180)

    pp.create_svc(net, 3, 1, -10, 1.01, 144)
    pp.create_svc(net, 2, 1, -10, 1., 144)

    pp.create_ssc(net, 6, 0, 5, 1, controllable=True, in_service=True)
    return net


@pytest.mark.parametrize("svc_status", list(product([True, False], repeat=2)))
@pytest.mark.parametrize("tcsc_status", list(product([True, False], repeat=2)))
@pytest.mark.parametrize("ssc_status", list(product([True, False], repeat=1)))
def test_multiple_facts_combinations(svc_status, tcsc_status, ssc_status):
    net = _many_tcsc_test_net()

    net.svc.controllable = svc_status
    net.tcsc.controllable = tcsc_status
    net.ssc.in_service = ssc_status
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.query("name=='tcsc'").index)
    compare_ssc_impedance_gen(net, net_ref)

    net.svc.controllable = True
    net.tcsc.controllable = True
    net.svc.in_service = svc_status
    net.tcsc.in_service = tcsc_status

    # pp.create_ssc(net, 6, 0, 5, 1,controllable=True,in_service=True)

    runpp_with_consistency_checks(net)

    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.query("name=='tcsc'").index)
    compare_ssc_impedance_gen(net, net_ref)


def test_svc_tcsc_case_study():
    net = facts_case_study_grid()
    baseMVA = net.sn_mva
    baseV = 230
    baseZ = baseV ** 2 / baseMVA
    xl = 0.2
    xc = -20
    # plot_z(baseZ, xl, xc)
    f = net.bus.loc[net.bus.name == "B4"].index.values[0]
    t = net.bus.loc[net.bus.name == "B6"].index.values[0]
    aux = pp.create_bus(net, 230, "aux")
    l = net.line.loc[(net.line.from_bus == f) & (net.line.to_bus == t)].index.values[0]
    net.line.loc[l, "from_bus"] = aux

    pp.create_tcsc(net, f, aux, xl, xc, -100, 100, controllable=True)

    pp.create_svc(net, net.bus.loc[net.bus.name == "B7"].index.values[0], 1, -10, 1., 90)

    pp.runpp(net, init="dc")

    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)

    net.gen.slack_weight = 1
    pp.runpp(net, distributed_slack=True, init="dc")
    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref, distributed_slack=True)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)


#
# def test_tcsc_firing_angle_formula():
#     net = pp.create_empty_network()
#     pp.create_buses(net, 2, 110)
#     pp.create_ext_grid(net, 0)
#     pp.create_line_from_parameters(net, 0, 1, 100, 0.0487, 0.13823, 160, 0.664)
#     pp.create_impedance(net, 0, 1, 0, 0.001, 1)
#     #pp.create_line_from_parameters(net, 1, 2, 100, 0.0487, 0.13823, 160, 0.664)
#     pp.create_load(net, 1, 100, 25)
#
#     z_base_ohm = np.square(110) / 1
#     #y_pu = calc_y_svc_pu(np.deg2rad(134.438395), 0.5 / z_base_ohm, -2 / z_base_ohm)
#      y_pu = calc_y_svc_pu(np.deg2rad(135.401298), 0.5 / z_base_ohm, -2 / z_base_ohm)
#     y_pu = 1/(-18.9/z_base_ohm)
#     print((1/y_pu) * z_base_ohm)
#     #    net.impedance.rft_pu
#     net.impedance.xft_pu = 1 / y_pu
#     net.impedance.xtf_pu = 1 / y_pu
#     runpp_with_consistency_checks(net)
#     print(net.res_line.loc[0])
#     print(net.res_impedance.loc[0])

#
# def test_tcsc_firing_angle_formula():
#     net = pp.create_empty_network()
#     pp.create_buses(net, 2, 110)
#     pp.create_ext_grid(net, 0)
#     pp.create_line_from_parameters(net, 0, 1, 100, 0.0487, 0.13823, 160, 0.664)
#     pp.create_line_from_parameters(net, 0, 1, 1, 0, 0.01, 0, 0.664)
#     pp.create_load(net, 1, 100, 25)
#
#     z_base_ohm = np.square(110) / 1
#     #y_pu = calc_y_svc_pu(np.deg2rad(134.438395), 0.5 / z_base_ohm, -2 / z_base_ohm)
#     y_pu = calc_y_svc_pu(np.deg2rad(135.401298), 0.5 / z_base_ohm, -2 / z_base_ohm)
#     #y_pu = calc_y_svc_pu(np.deg2rad(135.401298), 0.5 / z_base_ohm, -2 / z_base_ohm)
#     print((1/y_pu) * z_base_ohm)
#     #    net.impedance.rft_pu
#     #net.line.loc[1, "x_ohm_per_km"] = (1 / y_pu) * z_base_ohm
#     net.line.loc[1, "x_ohm_per_km"] = -18.9
#     pp.runpp(net, max_iteration=100)
#     print(net.res_line)
#
#
# def test_tcsc_firing_angle_formula():
#     net = pp.create_empty_network()
#     pp.create_buses(net, 2, 110)
#     pp.create_ext_grid(net, 0)
#     pp.create_line_from_parameters(net, 0, 1, 100, 0.0487, 0.13823, 160, 0.664)
#     pp.create_load(net, 1, 100, 25)
#
#     z_base_ohm = np.square(110) / 1
#     #y_pu = calc_y_svc_pu(np.deg2rad(134.438395), 0.5 / z_base_ohm, -2 / z_base_ohm)
#     y_pu = calc_y_svc_pu(np.deg2rad(141), 1 / z_base_ohm, -10 / z_base_ohm)
#     #y_pu = calc_y_svc_pu(np.deg2rad(135.401298), 0.5 / z_base_ohm, -2 / z_base_ohm)
#     print((1/y_pu) )
#     print((1/y_pu) * z_base_ohm)
#     print(xtcsc(np.deg2rad(141), 1, -10) / z_base_ohm)
#     #    net.impedance.rft_pu
#     pp.create_shunt(net, 1, -y_pu, y_pu)
#     runpp_with_consistency_checks(net)
#     print(net.res_line.loc[0])


def test_tcsc_simple5():
    net = pp.create_empty_network(sn_mva=100)
    pp.create_buses(net, 4, 110)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 3, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_load(net, 3, 100, 25)

    pp.create_tcsc(net, 2, 3, 1, -10, -20, 90)
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, 0, 0)


def test_ssc_simple():
    net = pp.create_empty_network()
    pp.create_buses(net, 2, 110)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_load(net, 1, 100, 25)
    pp.create_ssc(net, 1, 0, 5, 1)
    runpp_with_consistency_checks(net)

    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)

    ### compare (ssc) to bus 1(net)

    assert np.isclose(net.res_bus.at[1, "vm_pu"], net.ssc.set_vm_pu.at[0], rtol=0, atol=1e-6)
    assert np.isclose(np.abs(net._ppc["internal"]["V"][-1]), net.res_ssc.vm_internal_pu.at[0], rtol=0, atol=1e-6)

    assert np.isclose(net.res_ssc.vm_pu[0], net.res_bus.vm_pu.at[1], rtol=0, atol=1e-6)
    assert np.isclose(net.res_ssc.va_degree[0], net.res_bus.va_degree.at[1], rtol=0, atol=1e-6)

    compare_ssc_impedance_gen(net, net_ref)

    assert np.isclose(net.res_bus.q_mvar[0], net_ref.res_bus.q_mvar.at[0], rtol=0, atol=1e-6)
    assert np.isclose(net.res_ssc.q_mvar[0], net.res_bus.q_mvar.at[1] - net.load.q_mvar.at[0], rtol=0, atol=1e-6)
    assert np.isclose(net.res_ssc.q_mvar[0], net_ref.res_impedance.q_from_mvar.at[0], rtol=0, atol=1e-6)


def test_ssc_minimal():
    net = pp.create_empty_network()
    pp.create_bus(net, 110)
    pp.create_ext_grid(net, 0)
    pp.create_ssc(net, 0, 0, 5, 1)
    runpp_with_consistency_checks(net)

    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)

    ### compare (ssc) to bus 1(net)

    assert np.isclose(net.res_bus.at[0, "vm_pu"], net.ssc.set_vm_pu.at[0], rtol=0, atol=1e-6)
    assert np.isclose(np.abs(net._ppc["internal"]["V"][-1]), net.res_ssc.vm_internal_pu.at[0], rtol=0, atol=1e-6)

    assert np.isclose(net.res_ssc.vm_pu[0], net.res_bus.vm_pu.at[0], rtol=0, atol=1e-6)
    assert np.isclose(net.res_ssc.va_degree[0], net.res_bus.va_degree.at[0], rtol=0, atol=1e-6)

    compare_ssc_impedance_gen(net, net_ref)

    assert np.isclose(net.res_bus.q_mvar[0], net_ref.res_bus.q_mvar.at[0], rtol=0, atol=1e-6)
    assert np.isclose(net.res_ssc.q_mvar[0], net_ref.res_impedance.q_from_mvar.at[0], rtol=0, atol=1e-6)


def test_ssc_controllable():
    net = pp.create_empty_network()
    pp.create_buses(net, 3, 110)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)

    z_base = np.square(110) / net.sn_mva
    x = 5
    # both not controllable
    net1 = copy.deepcopy(net)
    pp.create_ssc(net1, 1, 0, x, 1, controllable=False)
    pp.create_ssc(net1, 2, 0, x, 1)
    runpp_with_consistency_checks(net1)
    assert np.isclose(net1.res_ssc.vm_internal_pu.at[0], 1, rtol=0, atol=1e-6)
    assert np.isclose(net1.res_ssc.vm_pu.at[1], 1, rtol=0, atol=1e-6)

    net2 = copy.deepcopy(net)
    pp.create_ssc(net2, 1, 0, x, 1, controllable=False, vm_internal_pu=1.02, va_internal_degree=150)
    runpp_with_consistency_checks(net2)
    assert np.isclose(net2.res_ssc.vm_internal_pu, 1.02, rtol=0, atol=1e-6)


def test_ssc_case_study():
    net = facts_case_study_grid()

    pp.create_ssc(net, bus=6, r_ohm=0, x_ohm=5, set_vm_pu=1, controllable=True)
    # pp.create_svc(net, 6, 1, -10, 1., 90,controllable=True)
    # net.res_ssc.q_mvar = -9.139709

    runpp_with_consistency_checks(net)


def test_2_sscs():
    net = pp.create_empty_network()
    pp.create_buses(net, 3, 110)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)

    z_base = np.square(110) / net.sn_mva
    x = 5
    # both not controllable
    net1 = copy.deepcopy(net)
    pp.create_ssc(net1, 1, 0, x, 1, controllable=True)
    pp.create_ssc(net1, 2, 0, x, 1, controllable=True)
    runpp_with_consistency_checks(net1)

    net2 = copy_with_impedance(net1)
    pp.runpp(net2)
    compare_ssc_impedance_gen(net1, net2)

    # first controllable
    net1 = copy.deepcopy(net)
    pp.create_ssc(net1, 1, 0, x, 1, in_service=False, controllable=False)
    pp.create_ssc(net1, 2, 0, x, 1, in_service=False, controllable=False)
    pp.runpp(net1)
    net2 = copy_with_impedance(net1)
    pp.runpp(net2)
    compare_ssc_impedance_gen(net1, net2)

    return
    # todo:
    # # second controllable
    # net1 = copy.deepcopy(net)
    # pp.create_ssc(net1, 1, 0, 121/z_base, 1, controllable=False)
    # pp.create_ssc(net1, 2, 0, 121/z_base, 1, controllable=True)
    #
    # pp.runpp(net1)
    # net2 = copy.deepcopy(net)
    # pp.create_load(net2, [1, 2], 100, 25)
    # pp.runpp(net2)
    # assert_frame_equal(net1.res_bus, net2.res_bus)
    #
    # # both controllable
    # net1 = copy.deepcopy(net)
    # pp.create_ssc(net1, 1, 0, 121/z_base, 1, controllable=True)
    # pp.create_ssc(net1, 2, 0, 121/z_base, 1, controllable=True)
    # pp.runpp(net1)
    # net2 = copy.deepcopy(net)
    # pp.create_load(net2, [1, 2], 100, 25)
    # pp.runpp(net2)
    # assert_frame_equal(net1.res_bus, net2.res_bus)


def test_vsc_hvdc():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc_from_parameters(net, 0, 1, 100, 0.1, 1)

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1.,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1.,
                  control_mode_dc="p_mw", control_value_dc=5)

    pp.runpp(net)
    runpp_with_consistency_checks(net)


def test_vsc_hvdc_control_q():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc_from_parameters(net, 0, 1, 100, 0.1, 1)

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="q_mvar", control_value_ac=7.5,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1.,
                  control_mode_dc="p_mw", control_value_dc=5)

    runpp_with_consistency_checks(net)


def test_vsc_multiterminal_hvdc():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 5, 110, geodata=[(0, 50), (50, 100), (200, 100), (50, 0), (200, 0)])
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 0, 3, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 3, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 3, 4, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 5)

    # DC part
    pp.create_bus_dc(net, 320, 'A', geodata=(50, 100))
    pp.create_bus_dc(net, 320, 'B', geodata=(200, 50))
    pp.create_bus_dc(net, 320, 'C', geodata=(200, 100))
    pp.create_bus_dc(net, 320, 'D', geodata=(200, 0))
    pp.create_bus_dc(net, 320, 'E', geodata=(50, 0))

    pp.create_line_dc_from_parameters(net, 0, 1, 100, 0.1, 1)
    pp.create_line_dc_from_parameters(net, 1, 2, 100, 0.1, 1)
    pp.create_line_dc_from_parameters(net, 1, 3, 100, 0.1, 1)
    pp.create_line_dc_from_parameters(net, 1, 4, 100, 0.1, 1)

    # pp.create_vsc(net, 1, 0, 0.1, 5, control_mode_dc="vm_pu", control_value_dc=1.02)
    # pp.create_vsc(net, 2, 2, 0.1, 5, control_value_dc=5)
    # pp.create_vsc(net, 4, 3, 0.1, 5, control_value_dc=15)
    # pp.create_vsc(net, 3, 4, 0.1, 5, control_mode_dc="vm_pu", control_value_dc=1.02)

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="q_mvar", control_value_ac=3,
                  control_mode_dc="vm_pu", control_value_dc=1)
    pp.create_vsc(net, 2, 2, 0.1, 5, 0.15,
                  control_mode_ac="q_mvar", control_value_ac=10,
                  control_mode_dc="p_mw", control_value_dc=5)
    pp.create_vsc(net, 4, 3, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1.05,
                  control_mode_dc="p_mw", control_value_dc=15)
    pp.create_vsc(net, 3, 4, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1.03,
                  control_mode_dc="vm_pu", control_value_dc=1.02)

    runpp_with_consistency_checks(net)


def test_line_dc_bus_dc_structures():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)

    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 5)

    pp.runpp(net)

    # DC part
    ## bus structure 1
    # pp.create_bus_dc(net, 110, 'A')
    # pp.create_bus_dc(net, 110, 'B')
    ## bus structure 2
    pp.create_buses_dc(net, 2, 110)

    ## line structure 1
    # pp.create_line_dc_from_parameters(net, 0, 1, 100, 0.1, 1)

    ## line structure 2
    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1.,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1.,
                  control_mode_dc="p_mw", control_value_dc=5)

    runpp_with_consistency_checks(net)


def test_line_dc_bus_dc_structures2():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 5, 110, geodata=[(0, 50), (50, 100), (200, 100), (50, 0), (200, 0)])
    pp.create_lines_from_parameters(net, [0, 1, 0, 1, 3], [1, 2, 3, 3, 4], 30, 0.0487, 0.13823, 160, 0.664)

    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 5)

    # DC part
    pp.create_bus_dc(net, 320, 'A', geodata=(50, 100))
    pp.create_bus_dc(net, 320, 'B', geodata=(200, 50))
    pp.create_bus_dc(net, 320, 'C', geodata=(200, 100))
    pp.create_bus_dc(net, 320, 'D', geodata=(200, 0))
    pp.create_bus_dc(net, 320, 'E', geodata=(50, 0))

    ## line structure 3
    # pp.create_lines_dc_from_parameters(net,[0,1,1,1],[1,2,3,4],100,0.01,1)
    # pp.create_lines_from_parameters()
    ## line structure 4
    pp.create_lines_dc(net, [0, 1, 1, 1], [1, 2, 3, 4], 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1.,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 2, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1.,
                  control_mode_dc="p_mw", control_value_dc=5)
    pp.create_vsc(net, 4, 3, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1.,
                  control_mode_dc="p_mw", control_value_dc=15)
    pp.create_vsc(net, 3, 4, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1.,
                  control_mode_dc="vm_pu", control_value_dc=1.02)

    runpp_with_consistency_checks(net)


def test_vsc_hvdc_structure1():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=5)
    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)

    runpp_with_consistency_checks(net)


def test_vsc_hvdc_structure1_alternate():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=5)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)

    runpp_with_consistency_checks(net)


def test_setting_of_dc_out_of_service():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')
    pp.create_bus_dc(net, 110, 'T', in_service=False)  # todo results for this dc bus must be NaN

    pp.create_vsc(net, 1, 0, 0, 5, 0.15, control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 1, 0, 5, 0.15, control_value_dc=5)
    pp.create_vsc(net, 2, 2, 0, 5, 0.15, control_value_dc=5)

    runpp_with_consistency_checks(net)  ## does the not in_service dc bus set the vsc out of service?


def test_setting_of_dc_vsc_out_of_service():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')
    pp.create_bus_dc(net, 110, 'T', in_service=False)  # todo results for this dc bus must be NaN

    pp.create_vsc(net, 1, 0, 0, 5, 0.15, control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 1, 0, 5, 0.15, control_value_dc=5)
    pp.create_vsc(net, 2, 2, 0, 5, 0.15, control_value_dc=5, in_service=False)

    runpp_with_consistency_checks(net)  ## does the not in_service dc bus set the vsc out of service?


def test_vsc_hvdc_structure2():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 5, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 2, 3, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 3, 4, 30, 0.0487, 0.13823, 160, 0.664)

    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 3)
    pp.create_load(net, 4, 10, 4)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")
    pp.create_line_dc(net, 1, 2, 100, std_type="2400-CU")
    pp.create_line_dc(net, 2, 3, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=5)

    pp.create_vsc(net, 3, 2, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 4, 3, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=5)

    runpp_with_consistency_checks(net)


def test_vsc_hvdc_mode0_without_dc_line():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1.02,
                  control_mode_dc="vm_pu", control_value_dc=1.02)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'vm_pu'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'vm_pu'], net.vsc.at[1, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_dc_line():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 2, 110)

    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, 10, 5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 0, 0, 0.1, 5, 0.15,
                  control_mode_ac="q_mvar", control_value_ac=0,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 1, 1, 0.1, 5, 0.15,
                  control_mode_ac="slack", control_value_ac=1.02,
                  control_mode_dc="p_mw", control_value_dc=0)

    runpp_with_consistency_checks(net)


def test_vsc_hvdc_mode1():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15, control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15, control_mode_ac="vm_pu", control_value_ac=1.02,
                  control_mode_dc="vm_pu", control_value_dc=1)

    pp.runpp(net)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'vm_pu'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'vm_pu'], net.vsc.at[1, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode2():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=5)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1.02,
                  control_mode_dc="vm_pu", control_value_dc=1)

    # pp.runpp(net)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'vm_pu'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'vm_pu'], net.vsc.at[1, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'p_mw'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode2_without_dc_line():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15, control_mode_ac="vm_pu", control_value_ac=1, control_mode_dc="p_mw",
                  control_value_dc=5)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15, control_mode_ac="vm_pu", control_value_ac=1.02, control_mode_dc="vm_pu",
                  control_value_dc=1)

    runpp_with_consistency_checks(net)

    # todo one of the vsc is out of service, so only 1 dc bus will have correct result

    # assert np.isclose(net.res_bus.at[1, 'vm_pu'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'vm_pu'], net.vsc.at[1, 'control_value_ac'], rtol=0, atol=1e-6)
    # assert np.isclose(net.res_bus_dc.at[0, 'p_mw'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode3():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1.02,
                  control_mode_dc="p_mw", control_value_dc=5)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'vm_pu'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'vm_pu'], net.vsc.at[1, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'p_mw'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode4():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15, control_mode_ac="vm_pu", control_value_ac=1, control_mode_dc="p_mw",
                  control_value_dc=5)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15, control_mode_ac="vm_pu", control_value_ac=1.02, control_mode_dc="p_mw",
                  control_value_dc=5)

    # all DC buses are set out of service, so there is no error in this case and bus_dc results are NaN
    # with pytest.raises(UserWarning, match="reference bus for the dc grid"):
    #     pp.runpp(net)
    runpp_with_consistency_checks(net)


def test_vsc_hvdc_mode5():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=4)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15,
                  control_mode_ac="q_mvar", control_value_ac=3,
                  control_mode_dc="vm_pu", control_value_dc=1)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'vm_pu'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'q_mvar'], net.vsc.at[1, 'control_value_ac'] + net.load.at[0, "q_mvar"], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'p_mw'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode6():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15,
                  control_mode_ac="q_mvar", control_value_ac=3,
                  control_mode_dc="vm_pu", control_value_dc=1)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'vm_pu'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'q_mvar'], net.vsc.at[1, 'control_value_ac'] + net.load.at[0, "q_mvar"], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode7():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15,
                  control_mode_ac="q_mvar", control_value_ac=3,
                  control_mode_dc="p_mw", control_value_dc=4)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'vm_pu'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'q_mvar'], net.vsc.at[1, 'control_value_ac'] + net.load.at[0, "q_mvar"], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'p_mw'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode9():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 4)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="q_mvar", control_value_ac=5,
                  control_mode_dc="p_mw", control_value_dc=4)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'q_mvar'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'vm_pu'], net.vsc.at[1, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'p_mw'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode10():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="q_mvar", control_value_ac=3,
                  control_mode_dc="vm_pu", control_value_dc=1)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'q_mvar'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'vm_pu'], net.vsc.at[1, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode11():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="q_mvar", control_value_ac=3,
                  control_mode_dc="vm_pu", control_value_dc=1)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=4)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'q_mvar'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'vm_pu'], net.vsc.at[1, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'p_mw'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode13():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 4)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="q_mvar", control_value_ac=5,
                  control_mode_dc="p_mw", control_value_dc=4)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15,
                  control_mode_ac="q_mvar", control_value_ac=5,
                  control_mode_dc="vm_pu", control_value_dc=1)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'q_mvar'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'q_mvar'], net.vsc.at[1, 'control_value_ac'] + net.load.at[0, "q_mvar"], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'p_mw'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode14():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15, control_mode_ac="q_mvar", control_value_ac=3, control_mode_dc="vm_pu",
                  control_value_dc=1)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15, control_mode_ac="q_mvar", control_value_ac=3, control_mode_dc="vm_pu",
                  control_value_dc=1)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'q_mvar'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'q_mvar'], net.vsc.at[1, 'control_value_ac'] + net.load.at[0, "q_mvar"], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode15():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 5)

    # DC part
    pp.create_bus_dc(net, 110, 'A', index=1)
    pp.create_bus_dc(net, 110, 'B', index=0)

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU", index=6)

    pp.create_vsc(net, 1, 1, 0.1, 5, 0.15,
                  control_mode_ac="q_mvar", control_value_ac=3,
                  control_mode_dc="vm_pu", control_value_dc=1, index=3)
    pp.create_vsc(net, 2, 0, 0.1, 5, 0.15,
                  control_mode_ac="q_mvar", control_value_ac=4,
                  control_mode_dc="p_mw", control_value_dc=4, index=2)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'q_mvar'], net.vsc.at[3, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'q_mvar'], net.vsc.at[2, 'control_value_ac'] + net.load.at[0, "q_mvar"], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[3, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'p_mw'], net.vsc.at[2, 'control_value_dc'], rtol=0, atol=1e-6)


# TODO : indexing tests for VSC and bus_dc


def test_minimal_ac():
    net = pp.create_empty_network()
    # AC part
    pp.create_bus(net, 110)
    pp.create_ext_grid(net, 0, vm_pu=1.02)

    # DC part
    pp.create_bus_dc(net, 110, 'A')

    pp.create_vsc(net, 0, 0, 0.1, 6, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1.01,
                  control_mode_dc="vm_pu", control_value_dc=1.02)

    with pytest.raises(UserWarning, match="Voltage controlling elements"):
        pp.runpp(net)

    net.vsc.at[0, "control_value_ac"] = 1.02

    net_copy = copy_with_impedance(net)
    pp.runpp(net_copy)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)

    compare_ssc_impedance_gen(net, net_copy, "vsc")


def test_minimal_vsc_hvdc():
    net = pp.create_empty_network()
    # AC part
    pp.create_bus(net, 110)
    pp.create_ext_grid(net, 0)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_vsc(net, 0, 0, 0.1, 5, 0.15, control_mode_dc="vm_pu", control_value_dc=1.02)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    # assert pd.isnull(net.res_bus_dc.at[1, 'vm_pu'])  #todo
    # assert pd.isnull(net.res_bus_dc.at[1, 'p_mw'])   #todo

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_line_dc.at[0, 'vm_from_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_line_dc.at[0, 'vm_to_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_line_dc.at[0, 'loading_percent'], 0, rtol=0, atol=1e-6)

    #pp.create_bus(net, 110)
    #pp.create_load(net, 2, 10)
    #pp.create_vsc(net, 1, 1, 0.1, 5, control_mode_ac="slack", control_value_ac=1.03, control_mode_dc="p_mw", control_value_dc=10)  #todo


def test_simple_vsc_hvdc():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, q_mvar=5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)

    runpp_with_consistency_checks(net)

    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=1)
    runpp_with_consistency_checks(net)


def test_simple_2vsc_hvdc1():
    # np.set_printoptions(linewidth=1000, suppress=True, precision=3)
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110, geodata=[(0, 0), (100, 0), (200, 0)])
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, 10, q_mvar=5)

    # DC part
    pp.create_bus_dc(net, 110, 'A', geodata=(100, 10))
    pp.create_bus_dc(net, 110, 'B', geodata=(200, 10))

    pp.create_bus_dc(net, 110, 'C', geodata=(100, -10))
    pp.create_bus_dc(net, 110, 'D', geodata=(200, -10))

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")
    pp.create_line_dc(net, 2, 3, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac='vm_pu', control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=10)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15,
                  control_mode_ac='vm_pu', control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)

    pp.create_vsc(net, 1, 2, 0.1, 5, 0.15,
                  control_mode_ac='vm_pu', control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=10)
    pp.create_vsc(net, 2, 3, 0.1, 5, 0.15,
                  control_mode_ac='vm_pu', control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)

    runpp_with_consistency_checks(net)


def test_simple_2vsc_hvdc2():
    # np.set_printoptions(linewidth=1000, suppress=True, precision=3)
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110, geodata=[(0, 0), (100, 0), (200, 0)])
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, 10, q_mvar=5)

    # DC part
    pp.create_bus_dc(net, 110, 'A', geodata=(100, 10))
    pp.create_bus_dc(net, 110, 'B', geodata=(200, 10))

    pp.create_bus_dc(net, 110, 'C', geodata=(100, -10))
    pp.create_bus_dc(net, 110, 'D', geodata=(200, -10))

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")
    pp.create_line_dc(net, 2, 3, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac='vm_pu', control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=10)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15,
                  control_mode_ac='vm_pu', control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)

    #pp.create_vsc(net, 1, 2, 0.1, 5, control_mode_ac='vm_pu', control_value_ac=1, control_mode_dc="p_mw", control_value_dc=10)
    #pp.create_vsc(net, 2, 3, 0.1, 5, control_mode_ac='vm_pu', control_value_ac=1, control_mode_dc="vm_pu", control_value_dc=1.02)

    pp.create_buses(net, 2, 110, geodata=[(100, -5), (200, -5)])
    pp.create_line_from_parameters(net, 1, 3, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 2, 4, 30, 0.0487, 0.13823, 160, 0.664)

    pp.create_vsc(net, 3, 2, 0.1, 5, 0.15,
                  control_mode_ac='vm_pu', control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=10)
    pp.create_vsc(net, 4, 3, 0.1, 5, 0.15,
                  control_mode_ac='vm_pu', control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)

    runpp_with_consistency_checks(net)


def test_b2b_vsc_1():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, q_mvar=5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=2)

    runpp_with_consistency_checks(net)
    assert np.isclose(net.res_bus_dc.at[0, 'p_mw'], 0, rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[0, 'p_dc_mw'], -net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[1, 'p_dc_mw'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_multiple_b2b_vsc_1():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 5, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 3, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 2, 4, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, q_mvar=5)

    # first B2B converter
    pp.create_bus_dc(net, 110, 'A')

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=2)

    # second B2B converter
    pp.create_bus_dc(net, 110, 'B')

    pp.create_vsc(net, 3, 1, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 4, 1, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=4)

    runpp_with_consistency_checks(net)
    assert np.isclose(net.res_bus_dc.at[0, 'p_mw'], 0, rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[0, 'p_dc_mw'], -net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[1, 'p_dc_mw'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)

    assert np.isclose(net.res_bus_dc.at[1, 'p_mw'], 0, rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[2, 'p_dc_mw'], -net.vsc.at[3, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[3, 'p_dc_mw'], net.vsc.at[3, 'control_value_dc'], rtol=0, atol=1e-6)


def test_tres_amigas_b2b_vsc_1():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 5, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 0, 3, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 4, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, q_mvar=5)

    # DC system
    pp.create_bus_dc(net, 110, 'A')

    pp.create_vsc(net, 4, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=2)
    pp.create_vsc(net, 3, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=4)

    runpp_with_consistency_checks(net)
    assert net.res_ext_grid.p_mw.at[0] > 10
    assert np.isclose(net.res_bus_dc.at[0, 'p_mw'], 0, rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[0, 'p_dc_mw'],
                      -net.vsc.at[1, 'control_value_dc'] - net.vsc.at[2, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[1, 'p_dc_mw'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[2, 'p_dc_mw'], net.vsc.at[2, 'control_value_dc'], rtol=0, atol=1e-6)


def test_tres_amigas_b2b_vsc_2():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 5, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 0, 3, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 4, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, q_mvar=5)

    # DC system
    pp.create_bus_dc(net, 110, 'A')

    pp.create_vsc(net, 4, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 3, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=4)

    runpp_with_consistency_checks(net)
    assert net.res_ext_grid.p_mw.at[0] > 10
    assert np.isclose(net.res_bus_dc.at[0, 'p_mw'], 0, rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[0, 'p_dc_mw'], -net.vsc.at[2, 'control_value_dc'] / 2, rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[1, 'p_dc_mw'], -net.vsc.at[2, 'control_value_dc'] / 2, rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[2, 'p_dc_mw'], net.vsc.at[2, 'control_value_dc'], rtol=0, atol=1e-6)


def test_b2b_vsc_2():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 4, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 2, 3, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 0, 3, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 3, 10, q_mvar=5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=2)

    runpp_with_consistency_checks(net)
    assert net.res_ext_grid.p_mw.at[0] > 10


def test_b2b_vsc_2a():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 4, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 2, 3, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 3, 10, q_mvar=5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 0, 0.1, 5, 0.15,
                  control_mode_ac="slack", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=0)

    runpp_with_consistency_checks(net)
    assert net.res_ext_grid.p_mw.at[0] > 10


def test_b2b_vsc_3():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 4, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 2, 3, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 3, 10, q_mvar=5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.01)
    pp.create_vsc(net, 2, 0, 0.1, 5, 0.15,
                  control_mode_ac="slack", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.01)
    # whatever we put in control_mode_dc should actually be ignored

    runpp_with_consistency_checks(net)
    assert net.res_ext_grid.p_mw.at[0] > 10


def test_b2b_vsc_4():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 4, 110)
    pp.create_line_from_parameters(net, 1, 3, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_load(net, 0, 40, 10)
    pp.create_ext_grid(net, 3)
    pp.create_load(net, 2, 80, 20)


    # DC part
    pp.create_bus_dc(net, 150, 'A')

    pp.create_vsc(net, 0, 0, 0.1, 5, 0.15,
                  control_mode_ac="slack", control_value_ac=1.,
                  control_mode_dc="p_mw", control_value_dc=0.)
    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="q_mvar", control_value_ac=10.,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 0, 0.1, 5, 0.15,
                  control_mode_ac="slack", control_value_ac=1.,
                  control_mode_dc="p_mw", control_value_dc=0.)

    runpp_with_consistency_checks(net)


def test_b2b_vsc_5():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 5, 110)
    pp.create_line_from_parameters(net, 1, 3, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 2, 4, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 3)
    pp.create_ext_grid(net, 4)
    pp.create_load(net, 0, 40, 10)
    pp.create_load(net, 2, 80, 20)


    # DC part
    pp.create_bus_dc(net, 150, 'A')

    pp.create_vsc(net, 0, 0, 0.1, 5, 0.15,
                  control_mode_ac="slack", control_value_ac=1.,
                  control_mode_dc="p_mw", control_value_dc=0.)
    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="q_mvar", control_value_ac=10.,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 0, 0.1, 5, 0.15,
                  control_mode_ac="q_mvar", control_value_ac=4.,
                  control_mode_dc="p_mw", control_value_dc=5.)
    runpp_with_consistency_checks(net)


def test_b2b_vsc_6():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 2)
    pp.create_load(net, 0, 20, 5)

    # DC part
    pp.create_bus_dc(net, 150, 'A')
    pp.create_vsc(net, 0, 0, 0.1, 5, 0.15,
                  control_mode_ac="slack", control_value_ac=1.,
                  control_mode_dc="p_mw", control_value_dc=0.)
    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="q_mvar", control_value_ac=0.,
                  control_mode_dc="vm_pu", control_value_dc=1.)

    # with pytest.raises(NotImplementedError):
    #     pp.runpp(net)
    runpp_with_consistency_checks(net)


def test_b2b_vsc_7():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 2, 110)
    pp.create_ext_grid(net, 1)  # todo: why is it not working when ext_grid is connected to the VSC AC bus?
    pp.create_load(net, 0, 20, 5)

    # DC part
    pp.create_bus_dc(net, 150, 'A')
    pp.create_vsc(net, 0, 0, 0.1, 5, 0.15,
                  control_mode_ac="slack", control_value_ac=1.,
                  control_mode_dc="p_mw", control_value_dc=0.)
    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="q_mvar", control_value_ac=0.,
                  control_mode_dc="vm_pu", control_value_dc=1.)

    runpp_with_consistency_checks(net)


def test_b2b_line_dc_raise():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 4, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 2, 3, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, q_mvar=5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=2)
    pp.create_vsc(net, 3, 1, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=2)

    runpp_with_consistency_checks(net)

    # with pytest.raises(NotImplementedError, match="Back-To-Back"):
    #     pp.runpp(net)


def test_line_dc_and_2_vsc1():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 4, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 2, 3, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, q_mvar=5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 3, 1, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=2)

    runpp_with_consistency_checks(net)


def test_line_dc_and_2_vsc2():
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 4, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 2, 3, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 3, 10, q_mvar=5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=-2)
    pp.create_vsc(net, 2, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=-2)
    pp.create_vsc(net, 3, 1, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)

    runpp_with_consistency_checks(net)


@pytest.mark.parametrize("control_mode_ac", list(product(['vm_pu', 'q_mvar'], repeat=2)))
@pytest.mark.parametrize("control_mode_dc", list(product(['vm_pu', 'p_mw'], repeat=2)))
def test_2vsc_1ac_2dc(control_mode_ac, control_mode_dc):
    if control_mode_dc[0] == control_mode_dc[1] == 'p_mw':
        pytest.skip("Skipping test with two 'p_mw' in control_mode_dc")
    print(f"{control_mode_dc=}, {control_mode_ac=}")

    val_ac = {"vm_pu": 1, "q_mvar": -5}
    val_dc = {"vm_pu": 1, "p_mw": 10}

    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 2, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, 10, 3)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac=control_mode_ac[0], control_value_ac=val_ac[control_mode_ac[0]],
                  control_mode_dc=control_mode_dc[0], control_value_dc=val_dc[control_mode_dc[0]])
    pp.create_vsc(net, 1, 1, 0.1, 5, 0.15,
                  control_mode_ac=control_mode_ac[1], control_value_ac=val_ac[control_mode_ac[1]],
                  control_mode_dc=control_mode_dc[1], control_value_dc=val_dc[control_mode_dc[1]])

    if control_mode_ac[0] == control_mode_ac[1]:
        runpp_with_consistency_checks(net)
    else:
        with pytest.raises(NotImplementedError, match="share the same AC bus"):
            pp.runpp(net)


@pytest.mark.skip(reason="DC line connected to D2B VSC configuration not implemented")
@pytest.mark.parametrize("control_mode_ac", list(product(['vm_pu', 'q_mvar'], repeat=2)))
@pytest.mark.parametrize("control_mode_dc", list(product(['vm_pu', 'p_mw'], repeat=2)))
def test_2vsc_2ac_1dc(control_mode_ac, control_mode_dc):
    if control_mode_dc[0] == control_mode_dc[1] == 'p_mw':
        pytest.skip("Skipping test with two 'p_mw' in control_mode_dc")

    val_ac = {"vm_pu": 1, "q_mvar": -5}
    val_dc = {"vm_pu": 1, "p_mw": 10}

    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)

    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, 10, 3)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'A')

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15, control_mode_ac=control_mode_ac[0],
                  control_value_ac=val_ac[control_mode_ac[0]], control_mode_dc=control_mode_dc[0],
                  control_value_dc=val_dc[control_mode_dc[0]])
    pp.create_vsc(net, 2, 0, 0.1, 5, 0.15, control_mode_ac=control_mode_ac[1],
                  control_value_ac=val_ac[control_mode_ac[1]], control_mode_dc=control_mode_dc[1],
                  control_value_dc=val_dc[control_mode_dc[1]])

    runpp_with_consistency_checks(net)


@pytest.mark.skip(reason="DC line connected to D2B VSC configuration not implemented")
@pytest.mark.parametrize("control_mode_ac", list(product(['vm_pu', 'q_mvar'], repeat=2)))
@pytest.mark.parametrize("control_mode_dc", list(product(['vm_pu', 'p_mw'], repeat=2)))
def test_2vsc_1ac_1dc(control_mode_ac, control_mode_dc):
    if control_mode_dc[0] == control_mode_dc[1] == 'p_mw':
        pytest.skip("Skipping test with two 'p_mw' in control_mode_dc")

    val_ac = {"vm_pu": 1, "q_mvar": -5}
    val_dc = {"vm_pu": 1, "p_mw": 10}

    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 2, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)

    pp.create_ext_grid(net, 0)
    pp.create_load(net, 1, 10, q_mvar=5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'A')

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15, control_mode_ac=control_mode_ac[0],
                  control_value_ac=val_ac[control_mode_ac[0]], control_mode_dc=control_mode_dc[0],
                  control_value_dc=val_dc[control_mode_dc[0]])
    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15, control_mode_ac=control_mode_ac[1],
                  control_value_ac=val_ac[control_mode_ac[1]], control_mode_dc=control_mode_dc[1],
                  control_value_dc=val_dc[control_mode_dc[1]])

    if control_mode_ac[0] == control_mode_ac[1]:
        runpp_with_consistency_checks(net)
    else:
        with pytest.raises(NotImplementedError, match="share the same AC bus"):
            pp.runpp(net)


def test_vsc_slack_minimal_wrong():
    # np.set_printoptions(linewidth=1000, suppress=True, precision=2)
    # from pandapower.test.loadflow.test_facts import *
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 2, 110, geodata=[(200, 0), (400, 0)])
    pp.create_load(net, 1, 10, 4)
    pp.create_gen(net, 0, 0)

    # DC part
    pp.create_bus_dc(net, 110, 'A', geodata=(210, 0))
    pp.create_bus_dc(net, 110, 'B', geodata=(390, 0))

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 0, 0, 0.1, 5, 0.15,
                  control_mode_ac="slack", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 1, 1, 0.1, 5, 0.15,
                  control_mode_ac="slack", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=1)

    # VSC as slack cannot have DC bus as vm_pu, therefore must be excluded from DC slacks
    # Then the DC buses are set out of service, and the corresponding VSC are also set out of service
    # Then the corresponding AC buses are changed from type REF to type PQ, which is valid because type PV is set later
    # Then runpp raises "no slacks" error:
    with pytest.raises(UserWarning, match="No reference bus is available"):
        pp.runpp(net)


def test_vsc_slack_minimal_wrong2():
    # np.set_printoptions(linewidth=1000, suppress=True, precision=2)
    # from pandapower.test.loadflow.test_facts import *
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 2, 110, geodata=[(200, 0), (400, 0)])
    pp.create_load(net, 1, 10, 4)

    # DC part
    pp.create_bus_dc(net, 110, 'A', geodata=(210, 0))
    pp.create_bus_dc(net, 110, 'B', geodata=(390, 0))

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 0, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 1, 1, 0.1, 5, 0.15,
                  control_mode_ac="slack", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=1)

    # VSC that defines AC slack cannot define DC slack at the same time
    # DC slack buses that are only connected to VSC AC slacks are converted to type P buses
    # Then runpp raises "no DC slacks" error:
    with pytest.raises(UserWarning, match="No reference bus for the dc grid is available"):
        pp.runpp(net)


@pytest.mark.xfail(reason="AC bus same as ext_grid bus not implemented")
def test_vsc_slack_minimal():  # todo: fix that FACTS elements can be connected to ext_grid buses
    # np.set_printoptions(linewidth=1000, suppress=True, precision=2)
    # from pandapower.test.loadflow.test_facts import *
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 2, 110, geodata=[(200, 0), (400, 0)])
    pp.create_load(net, 1, 10, 4)
    pp.create_ext_grid(net, 0)

    # DC part
    pp.create_bus_dc(net, 110, 'A', geodata=(210, 0))
    pp.create_bus_dc(net, 110, 'B', geodata=(390, 0))

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 0, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 1, 1, 0.1, 5, 0.15,
                  control_mode_ac="slack", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=1)

    # pp.runpp(net)

    runpp_with_consistency_checks(net)

    # pp.plotting.simple_plot(net, plot_loads=True, load_size=5)


def test_vsc_slack():
    # np.set_printoptions(linewidth=1000, suppress=True, precision=3)
    # from pandapower.test.loadflow.test_facts import *
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 4, 110, geodata=[(0, 0), (200, 0), (400, 0), (600, 0)])
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 2, 3, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 3, 10, 4)

    # DC part
    pp.create_bus_dc(net, 110, 'A', geodata=(210, 0))
    pp.create_bus_dc(net, 110, 'B', geodata=(390, 0))

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15,
                  control_mode_ac="slack", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=0.)

    runpp_with_consistency_checks(net)

    # pp.plotting.simple_plot(net, plot_loads=True)


def test_vsc_slack2():
    # np.set_printoptions(linewidth=1000, suppress=True, precision=3)
    # from pandapower.test.loadflow.test_facts import *
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110, geodata=[(0, 0), (200, 0), (400, 0)])
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 4)

    # DC part
    pp.create_bus_dc(net, 110, 'A', geodata=(210, 0))
    pp.create_bus_dc(net, 110, 'B', geodata=(390, 0))

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15,
                  control_mode_ac="slack", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=10)

    runpp_with_consistency_checks(net)

    # pp.plotting.simple_plot(net, plot_loads=True)


def test_vsc_slack_oos():
    # np.set_printoptions(linewidth=1000, suppress=True, precision=3)
    # from pandapower.test.loadflow.test_facts import *
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 4, 110, geodata=[(0, 0), (200, 0), (400, 0), (600, 0)])
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 2, 3, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 3, 10, 4)
    net.bus.loc[[2, 3], "in_service"] = False

    # DC part
    pp.create_bus_dc(net, 110, 'A', geodata=(210, 0))
    pp.create_bus_dc(net, 110, 'B', geodata=(390, 0))

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15,
                  control_mode_ac="slack", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=0.)

    runpp_with_consistency_checks(net)


def test_vsc_dc_r():
    # np.set_printoptions(linewidth=1000, suppress=True, precision=3)
    # from pandapower.test.loadflow.test_facts import *
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110, geodata=[(0, 0), (200, 0), (400, 0)])
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 4)

    # DC part
    pp.create_bus_dc(net, 110, 'A', geodata=(210, 0))
    pp.create_bus_dc(net, 110, 'B', geodata=(390, 0))

    pp.create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.15,
                  control_mode_ac="vm_pu", control_value_ac=1,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.15,
                  control_mode_ac="slack", control_value_ac=1,
                  control_mode_dc="p_mw", control_value_dc=10)

    runpp_with_consistency_checks(net)


def test_vsc_hvdc_dc_rl():
    # from pandapower.test.loadflow.test_facts import *
    net = pp.create_empty_network()
    # AC part
    pp.create_buses(net, 3, 110)
    pp.create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    pp.create_ext_grid(net, 0)
    pp.create_load(net, 2, 10, 5)

    # DC part
    pp.create_bus_dc(net, 110, 'A')
    pp.create_bus_dc(net, 110, 'B')

    pp.create_line_dc_from_parameters(net, 0, 1, 100, 0.1, 1)

    pp.create_vsc(net, 1, 0, 0.1, 5, 0.5, pl_dc_mw=0.5,
                  control_mode_ac="vm_pu", control_value_ac=1.,
                  control_mode_dc="vm_pu", control_value_dc=1.02)
    pp.create_vsc(net, 2, 1, 0.1, 5, 0.5, pl_dc_mw=0.75,
                  control_mode_ac="vm_pu", control_value_ac=1.,
                  control_mode_dc="p_mw", control_value_dc=5)

    runpp_with_consistency_checks(net)


def test_results_pf_grid():
    # todo: improve accuracy of DC losses and reduce tolerances
    path = os.path.join(pp.pp_dir, "test", "test_files", "test_ac_dc.json")
    net = pp.from_json(path)
    res = pp.converter.validate_pf_conversion(net)
    assert np.max(np.abs(res['diff_vm']['diff'])) < 1e-6
    assert np.max(np.abs(res['diff_va']['diff'])) < 1e-3
    assert np.max(np.abs(res['bus_dc_diff'])) < 5e-6
    assert np.max(np.abs(res['line_diff'])) < 0.02
    assert np.max(np.abs(res['line_dc_diff'])) < 1e-3
    assert np.max(np.abs(res['vsc_p_diff_is'])) < 0.02
    assert np.max(np.abs(res['vsc_q_diff_is'])) < 0.01
    assert np.max(np.abs(res['vsc_p_dc_diff_is'])) < 1e-3
    assert np.max(np.abs(res['ext_grid_p_diff'])) < 0.05
    assert np.max(np.abs(res['ext_grid_q_diff'])) < 0.01


# TODO test for when the VSC, SSC, TCSC, connect to same buses

if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

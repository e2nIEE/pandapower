# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest
import numpy as np
from pandas.testing import assert_frame_equal

import pandapower as pp
from pandapower.test.consistency_checks import runpp_with_consistency_checks
from pandapower.pf.makeYbus_facts import calc_y_svc_pu

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

    net_ref = net.deepcopy()
    for i in net.tcsc.index.values:
        pp.create_impedance(net_ref, net.tcsc.from_bus.at[i], net.tcsc.to_bus.at[i], 0,
                            net.res_tcsc.x_ohm.at[i] / baseZ[net.tcsc.to_bus.at[i]], baseMVA,
                            in_service=net.tcsc.in_service.at[i])
    net_ref.tcsc.in_service = False

    if len(net.svc) > 0:
        # pp.create_loads(net_ref, net.svc.bus.values, 0, net.res_svc.q_mvar.values, in_service=net.svc.in_service.values)
        # create shunts because of Ybus comparison
        q = np.square(net.bus.loc[net.svc.bus.values, 'vn_kv']) / net.res_svc.x_ohm.values
        pp.create_shunts(net_ref, net.svc.bus.values, q.fillna(0), in_service=net.svc.in_service.values)
        net_ref.svc.in_service = False
    return net_ref


def compare_tcsc_impedance(net, net_ref, idx_tcsc, idx_impedance):
    for col in ("vm_pu", "va_degree", "p_mw", "q_mvar"):
        assert np.allclose(net.res_bus[col], net_ref.res_bus[col], rtol=0, atol=1e-6)
    assert np.allclose(net.res_ext_grid.p_mw, net_ref.res_ext_grid.p_mw, rtol=0, atol=1e-6)
    assert np.allclose(net.res_ext_grid.q_mvar, net_ref.res_ext_grid.q_mvar, rtol=0, atol=1e-6)

    for col in "p_from_mw", "q_from_mvar", "p_to_mw", 'q_to_mvar':
        assert np.allclose(net.res_tcsc.loc[idx_tcsc, col], net_ref.res_impedance.loc[idx_impedance, col],
                           rtol=0, atol=1e-6)
    assert np.allclose(net._ppc["internal"]["Ybus"].toarray(), net_ref._ppc["internal"]["Ybus"].toarray(), rtol=0,
                       atol=1e-6)


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
    net3 = net.deepcopy()

    pp.create_svc(net, bus=3, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=90)
    net2 = net.deepcopy()
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

    net2.svc.thyristor_firing_angle_degree.at[0] = net.res_svc.thyristor_firing_angle_degree.at[0]
    net2.svc.controllable.at[0] = False
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
    net1 = net.deepcopy()
    pp.create_svc(net1, bus=1, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145,
                  controllable=False)
    pp.create_svc(net1, bus=2, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145,
                  controllable=False)
    pp.runpp(net1)
    net2 = net.deepcopy()
    pp.create_loads(net2, [1, 2], 0, net1.res_svc.q_mvar.values)
    pp.runpp(net2)
    assert_frame_equal(net1.res_bus, net2.res_bus)

    # first controllable
    net1 = net.deepcopy()
    pp.create_svc(net1, bus=1, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145)
    pp.create_svc(net1, bus=2, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145,
                  controllable=False)
    pp.runpp(net1)
    net2 = net.deepcopy()
    pp.create_loads(net2, [1, 2], 0, net1.res_svc.q_mvar.values)
    pp.runpp(net2)
    assert_frame_equal(net1.res_bus, net2.res_bus)

    # second controllable
    net1 = net.deepcopy()
    pp.create_svc(net1, bus=1, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145,
                  controllable=False)
    pp.create_svc(net1, bus=2, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145)
    pp.runpp(net1)
    net2 = net.deepcopy()
    pp.create_loads(net2, [1, 2], 0, net1.res_svc.q_mvar.values)
    pp.runpp(net2)
    assert_frame_equal(net1.res_bus, net2.res_bus)

    # both controllable
    net1 = net.deepcopy()
    pp.create_svc(net1, bus=1, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145)
    pp.create_svc(net1, bus=2, x_l_ohm=1, x_cvar_ohm=-10, set_vm_pu=vm_set_pu, thyristor_firing_angle_degree=145)
    pp.runpp(net1)
    net2 = net.deepcopy()
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

    net.tcsc.controllable.at[0] = True
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)

    net.tcsc.controllable.at[1] = True
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

    runpp_with_consistency_checks(net)

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

    net_ref = net.deepcopy()

    pp.create_tcsc(net, 1, 2, xl, xc, -20, 170, "Test", controllable=True, min_angle_degree=90, max_angle_degree=180)

    runpp_with_consistency_checks(net)

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
    runpp_with_consistency_checks(net)

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

    net_ref = net.deepcopy()

    pp.create_tcsc(net, f, aux, xl, xc, -100, 100, controllable=True)
    pp.runpp(net)

    pp.create_impedance(net_ref, f, aux, 0, net.res_tcsc.at[0, "x_ohm"] / baseZ, baseMVA)
    pp.runpp(net_ref)

    compare_tcsc_impedance(net, net_ref, 0, 0)
    assert np.allclose(net._ppc["internal"]["Ybus"].toarray(),
                       net_ref._ppc["internal"]["Ybus"].toarray(), rtol=0, atol=1e-6)


def test_multiple_facts():
    #                  |--(TCSC)--(4)------|
    # (0)-------------(1)-----------------(3)->
    #                  |-(5)-(TCSC)--(2)---|#
    # unfortunately, TCSC is not implemented for the case when multiple TCSC elements
    # have the same from_bus or to_bus
    baseMVA = 100
    baseV = 110
    baseZ = baseV ** 2 / baseMVA
    xl = 0.2
    xc = -20

    net = pp.create_empty_network(sn_mva=baseMVA)
    pp.create_buses(net, 6, baseV)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 3, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 2, 3, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 4, 3, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 5, 20, 0.0487, 0.13823, 160, 0.664)

    pp.create_load(net, 3, 100, 40)

    pp.create_tcsc(net, 5, 2, xl, xc, -10, 140, "Test", controllable=True, min_angle_degree=90, max_angle_degree=180)
    pp.create_tcsc(net, 1, 4, xl, xc, -5, 140, "Test", controllable=True, min_angle_degree=90, max_angle_degree=180)

    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)

    net.tcsc.loc[1, "thyristor_firing_angle_degree"] = net.res_tcsc.loc[1, "thyristor_firing_angle_degree"]
    net.tcsc.loc[1, "controllable"] = False
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)

    pp.create_svc(net, 3, 1, -10, 1.01, 90)
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)

    pp.create_svc(net, 2, 1, -10, 1., 144, controllable=False)
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)

    net.svc.thyristor_firing_angle_degree.at[0] = net.res_svc.loc[0, "thyristor_firing_angle_degree"]
    net.svc.controllable = [False, True]
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)

    net.tcsc.controllable = True
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)

    net.tcsc.controllable = False
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)

    net.svc.in_service.at[0] = False
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)

    net.tcsc.in_service.at[1] = False
    runpp_with_consistency_checks(net)
    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)


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

    pp.runpp(net)

    net_ref = copy_with_impedance(net)
    pp.runpp(net_ref)
    compare_tcsc_impedance(net, net_ref, net.tcsc.index, net_ref.impedance.index)

    net.gen.slack_weight = 1
    pp.runpp(net, distributed_slack=True)
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
#     y_pu = calc_y_svc_pu(np.deg2rad(135.401298), 0.5 / z_base_ohm, -2 / z_base_ohm)
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


if __name__ == "__main__":
    pytest.main([__file__])

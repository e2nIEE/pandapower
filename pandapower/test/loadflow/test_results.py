# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import pandas as pd
import pytest
from numpy import in1d, isnan, isclose

import pandapower as pp
import pandapower.control
from pandapower.test.consistency_checks import runpp_with_consistency_checks
from pandapower.test.loadflow.result_test_network_generator import add_test_enforce_qlims, \
    add_test_gen
from pandapower.test.helper_functions import assert_res_equal
from pandapower.test.conftest import result_test_network

# simple example grid for tap dependent impedance tests:

def add_trafo_connection(net, hv_bus, trafotype="2W"):
    cb = pp.create_bus(net, vn_kv=0.4)
    pp.create_load(net, cb, 0.2, 0.05)

    if trafotype == "3W":
        cbm = pp.create_bus(net, vn_kv=0.9)
        pp.create_load(net, cbm, 0.1, 0.03)
        pp.create_transformer3w_from_parameters(net, hv_bus=hv_bus, mv_bus=cbm, lv_bus=cb,
                                                vn_hv_kv=20., vn_mv_kv=0.9, vn_lv_kv=0.45, sn_hv_mva=0.6, sn_mv_mva=0.5,
                                                sn_lv_mva=0.4, vk_hv_percent=1., vk_mv_percent=1., vk_lv_percent=1.,
                                                vkr_hv_percent=0.3, vkr_mv_percent=0.3, vkr_lv_percent=0.3,
                                                pfe_kw=0.2, i0_percent=0.3, tap_neutral=0.,
                                                tap_pos=2, tap_step_percent=1., tap_min=-2, tap_max=2)
    else:
        pp.create_transformer(net, hv_bus=hv_bus, lv_bus=cb, std_type="0.25 MVA 20/0.4 kV", tap_pos=2)


def create_net():
    net = pp.create_empty_network()
    vn_kv = 20
    b1 = pp.create_bus(net, vn_kv=vn_kv)
    pp.create_ext_grid(net, b1, vm_pu=1.01)
    b2 = pp.create_bus(net, vn_kv=vn_kv)
    l1 = pp.create_line_from_parameters(net, b1, b2, 12.2, r_ohm_per_km=0.08, x_ohm_per_km=0.12,
                                        c_nf_per_km=300, max_i_ka=.2, df=.8)
    for i in range(2):
        add_trafo_connection(net, b2)

    return net


def test_line(result_test_network, v_tol=1e-6, i_tol=1e-6, s_tol=5e-3, l_tol=1e-3):
    net = result_test_network
    buses = net.bus[net.bus.zone == "test_line"]
    lines = [x for x in net.line.index if net.line.from_bus[x] in buses.index]
    l1 = lines[0]
    l2 = lines[1]
    l3 = lines[2]
    b2 = buses.index[1]

    # result values from powerfactory
    load1 = 14.578
    load2 = 8.385

    ika1 = 0.0466482
    ika2 = 0.0134161

    p_from1 = 1.212158
    p_from2 = 0.00511

    q_from1 = 0.167416
    q_from2 = -0.469371

    p_to1 = -1.20000
    p_to2 = 0.000

    q_to1 = -1.100000
    q_to2 = 0.0000

    v = 1.007389386


    # line 1
    assert abs(net.res_line.loading_percent.at[l1] - load1) < l_tol
    assert abs(net.res_line.i_ka.at[l1] - ika1) < i_tol
    assert abs(net.res_line.p_from_mw.at[l1] - p_from1) < s_tol
    assert abs(net.res_line.q_from_mvar.at[l1] - q_from1) < s_tol
    assert abs(net.res_line.p_to_mw.at[l1] - p_to1) < s_tol
    assert abs(net.res_line.q_to_mvar.at[l1] - q_to1) < s_tol

    # line2 (open switch line)
    assert abs(net.res_line.loading_percent.at[l2] - load2) < l_tol
    assert abs(net.res_line.i_ka.at[l2] - ika2) < i_tol
    assert abs(net.res_line.p_from_mw.at[l2] - p_from2) < s_tol
    assert abs(net.res_line.q_from_mvar.at[l2] - q_from2) < s_tol
    assert abs(net.res_line.p_to_mw.at[l2] - p_to2) < s_tol
    assert abs(net.res_line.q_to_mvar.at[l2] - q_to2) < s_tol

    assert abs(net.res_bus.vm_pu.at[b2] - v) < v_tol

    # line3 (of out of service line)
    assert abs(net.res_line.loading_percent.at[l3] - 0) < l_tol
    assert abs(net.res_line.i_ka.at[l3] - 0) < i_tol
    assert abs(net.res_line.p_from_mw.at[l3] - 0) < s_tol
    assert abs(net.res_line.q_from_mvar.at[l3] - 0) < s_tol
    assert abs(net.res_line.p_to_mw.at[l3] - 0) < s_tol
    assert abs(net.res_line.q_to_mvar.at[l3] - 0) < s_tol


def test_load_sgen(result_test_network, v_tol=1e-6, i_tol=1e-6, s_tol=5e-3, l_tol=1e-3):
    net = result_test_network
    buses = net.bus[net.bus.zone == "test_load_sgen"]
    loads = [x for x in net.load.index if net.load.bus[x] in buses.index]
    sgens = [x for x in net.sgen.index if net.sgen.bus[x] in buses.index]
    l1 = loads[0]
    sg1 = sgens[0]
    b2 = buses.index[1]
    # result values from powerfactory
    pl1 = 1.200000
    ql1 = 1.100000

    qs1 = -0.1000
    ps1 = 0.500

    u = 1.00477465

    assert abs(net.res_load.p_mw.at[l1] - pl1) < s_tol
    assert abs(net.res_load.q_mvar.at[l1] - ql1) < s_tol
    # pf uses generator system
    assert abs(net.res_sgen.p_mw.at[sg1] - ps1) < s_tol
    # pf uses generator system
    assert abs(net.res_sgen.q_mvar.at[sg1] - qs1) < s_tol
    assert abs(net.res_bus.vm_pu.at[b2] - u) < v_tol


def test_load_sgen_split(result_test_network, v_tol=1e-6, i_tol=1e-6, s_tol=5e-3, l_tol=1e-3):
    # splitting up the load/sgen should not change the result
    net = result_test_network
    buses = net.bus[net.bus.zone == "test_load_sgen_split"]
    b2 = buses.index[1]

    u = 1.00477465

    assert abs(net.res_bus.vm_pu.at[b2] - u) < v_tol


def test_trafo(result_test_network, v_tol=1e-6, i_tol=1e-6, s_tol=1e-2, l_tol=1e-3, va_tol=1e-2):
    net = result_test_network
    buses = net.bus[net.bus.zone == "test_trafo"]
    trafos = [x for x in net.trafo.index if net.trafo.hv_bus[x] in buses.index]
    t1 = trafos[0]
    t2 = trafos[1]
    t3 = trafos[2]
    b2 = buses.index[1]
    b3 = buses.index[2]
    # powerfactory results to check t-equivalent circuit model
    runpp_with_consistency_checks(net, trafo_model="t", trafo_loading="current", init="dc",
                                  calculate_voltage_angles=True)

    load1 = 28.7842
    load2 = 0.4830

    ph1 = 0.204756
    ph2 = 0.0017741

    qh1 = 0.052848
    qh2 = 0.000038

    pl1 = -0.2000000
    pl2 = 0

    ql1 = -0.050
    ql2 = 0.0

    ih1 = 0.006043
    ih2 = 0.000051

    il1 = 0.303631
    il2 = 0

    v2 = 1.010159155
    v3 = 0.980003098

    va2 = -0.06736233
    va3 = -150.73914408

    assert abs(net.res_trafo.loading_percent.at[t1] - load1) < l_tol
    assert abs(net.res_trafo.p_hv_mw.at[t1] - ph1) < s_tol
    assert abs(net.res_trafo.q_hv_mvar.at[t1] - qh1) < s_tol
    assert abs(net.res_trafo.p_lv_mw.at[t1] - pl1) < s_tol
    assert abs(net.res_trafo.q_lv_mvar.at[t1] - ql1) < s_tol
    assert abs(net.res_trafo.i_hv_ka.at[t1] - ih1) < i_tol
    assert abs(net.res_trafo.i_lv_ka.at[t1] - il1) < i_tol

    assert abs(net.res_trafo.loading_percent.at[t2] - load2) < l_tol
    assert abs(net.res_trafo.p_hv_mw.at[t2] - ph2) < s_tol
    assert abs(net.res_trafo.q_hv_mvar.at[t2] - qh2) < s_tol
    assert abs(net.res_trafo.p_lv_mw.at[t2] - pl2) < s_tol
    assert abs(net.res_trafo.q_lv_mvar.at[t2] - ql2) < s_tol
    assert abs(net.res_trafo.i_hv_ka.at[t2] - ih2) < i_tol
    assert abs(net.res_trafo.i_lv_ka.at[t2] - il2) < i_tol

    assert abs(net.res_trafo.loading_percent.at[t3] - 0) < l_tol
    assert abs(net.res_trafo.p_hv_mw.at[t3] - 0) < s_tol
    assert abs(net.res_trafo.q_hv_mvar.at[t3] - 0) < s_tol
    assert abs(net.res_trafo.p_lv_mw.at[t3] - 0) < s_tol
    assert abs(net.res_trafo.q_lv_mvar.at[t3] - 0) < s_tol
    assert abs(net.res_trafo.i_hv_ka.at[t3] - 0) < i_tol
    assert abs(net.res_trafo.i_lv_ka.at[t3] - 0) < i_tol

    assert abs(net.res_bus.vm_pu.at[b2] - v2) < v_tol
    assert abs(net.res_bus.vm_pu.at[b3] - v3) < v_tol

    assert abs(net.res_bus.va_degree.at[b2] - va2) < va_tol
    assert abs(net.res_bus.va_degree.at[b3] - va3) < va_tol

    # sincal results to check pi-equivalent circuit model
    net.trafo.loc[trafos, "parallel"] = 1  # sincal is tested without parallel transformers
    runpp_with_consistency_checks(net, trafo_model="pi", trafo_loading="current")

    load1 = 57.637
    load2 = 0.483
    v2 = 1.01014991616
    v3 = 0.97077261471

    assert abs(net.res_trafo.loading_percent.at[t1] - load1) < l_tol
    assert abs(net.res_trafo.loading_percent.at[t2] - load2) < l_tol

    assert abs(net.res_bus.vm_pu.at[b2] - v2) < v_tol
    assert abs(net.res_bus.vm_pu.at[b3] - v3) < v_tol

    runpp_with_consistency_checks(net, trafo_model="pi", trafo_loading="power")

    load1 = 52.929
    load2 = 0.444

    assert abs(net.res_trafo.loading_percent.at[t1] - load1) < l_tol
    assert abs(net.res_trafo.loading_percent.at[t2] - load2) < l_tol


def test_trafo_2_taps(v_tol=1e-6, i_tol=1e-6, s_tol=1e-2, l_tol=1e-3, va_tol=1e-2):
    # from pandapower.test.loadflow.test_results import *

    net = pp.create_empty_network()
    pp.create_bus(net, 110)
    pp.create_bus(net, 20)
    pp.create_ext_grid(net, 0)
    pp.create_transformer_from_parameters(net, 0, 1, 100, 110, 20, 0.5, 12, 14, 0.5,
                                          tap_side="hv", tap_neutral=0, tap_max=10,
                                          tap_min=-10, tap_step_percent=2, tap_step_degree=0,
                                          tap_pos=0, tap_phase_shifter=False,
                                          tap2_side="hv", tap2_neutral=0, tap2_max=10,
                                          tap2_min=-10, tap2_step_percent=2, tap2_step_degree=0,
                                          tap2_pos=0, tap2_phase_shifter=False)

    pp.create_load(net, 1, 10)

    pp.runpp(net)
    net.res_bus


def test_tap_dependent_impedance(result_test_network):
    net = result_test_network

    # first, basic example with piecewise linear characteristic
    characteristic_vk = pp.control.Characteristic.from_points(net, ((net.trafo.at[0, 'tap_min'], 0.9 * net.trafo.at[0, 'vk_percent']),
                                                                    (net.trafo.at[0, 'tap_neutral'], net.trafo.at[0, 'vk_percent']),
                                                                    (net.trafo.at[0, 'tap_max'], 1.1 * net.trafo.at[0, 'vk_percent'])))
    characteristic_vkr = pp.control.Characteristic.from_points(net, ((net.trafo.at[0, 'tap_min'], 0.9 * net.trafo.at[0, 'vkr_percent']),
                                                                     (net.trafo.at[0, 'tap_neutral'], net.trafo.at[0, 'vkr_percent']),
                                                                     (net.trafo.at[0, 'tap_max'], 1.1 * net.trafo.at[0, 'vkr_percent'])))
    idx_vk = characteristic_vk.index
    idx_vkr = characteristic_vkr.index

    # we use for reference
    net0 = net.deepcopy()

    net.trafo["tap_dependent_impedance"] = pd.Series(index=net.trafo.index, dtype=bool, data=False)
    net.trafo.loc[0, 'tap_dependent_impedance'] = True
    net.trafo.loc[0, ['vk_percent_characteristic', 'vkr_percent_characteristic']] = idx_vk, idx_vkr

    # first, make sure there is no change for neutral
    net.trafo.at[0, "tap_pos"] = net.trafo.tap_neutral.at[0]
    net0.trafo.at[0, "tap_pos"] = net.trafo.tap_neutral.at[0]
    pp.runpp(net)
    pp.runpp(net0)
    assert_res_equal(net, net0)


    # now check the min and max positions
    for pos, factor in (("tap_min", 0.9), ("tap_max", 1.1)):
        assert isclose(characteristic_vk(net.trafo[pos].at[0]), net.trafo.vk_percent.at[0]*factor, rtol=0, atol=1e-12)
        assert isclose(characteristic_vkr(net.trafo[pos].at[0]), net.trafo.vkr_percent.at[0]*factor, rtol=0, atol=1e-12)
        net0.trafo.at[0, "vk_percent"] = net.trafo.vk_percent.at[0]*factor
        net0.trafo.at[0, "vkr_percent"] = net.trafo.vkr_percent.at[0]*factor
        net0.trafo.at[0, "tap_pos"] = net.trafo[pos].at[0]
        pp.runpp(net0)
        net.trafo.at[0, "tap_pos"] = net.trafo[pos].at[0]
        pp.runpp(net)
        assert_res_equal(net, net0)


def test_tap_dependent_impedance_controller_comparison():
    net1 = create_net()
    net2 = create_net()

    pp.control.create_trafo_characteristics(net1, 'trafo', [0], 'vk_percent', [[-2, -1, 0, 1, 2]], [[5, 5.2, 6, 6.8, 7]])
    pp.control.create_trafo_characteristics(net1, 'trafo', [0], 'vkr_percent', [[-2, -1, 0, 1, 2]], [[1.3, 1.4, 1.44, 1.5, 1.6]])

    pp.control.SplineCharacteristic(net2, [-2, -1, 0, 1, 2], [5, 5.2, 6, 6.8, 7])
    pp.control.SplineCharacteristic(net2, [-2, -1, 0, 1, 2], [1.3, 1.4, 1.44, 1.5, 1.6])
    pp.control.TapDependentImpedance(net2, [0], 0, output_variable="vk_percent")
    pp.control.TapDependentImpedance(net2, [0], 1, output_variable="vkr_percent")

    pp.runpp(net1)
    pp.runpp(net2, run_control=True)

    assert_res_equal(net1, net2)


def test_tap_dependent_impedance_controller_comparison_3w():
    net1 = create_net()
    net2 = create_net()
    for i in range(2):
        add_trafo_connection(net1, net1.trafo.at[0, 'hv_bus'], "3W")
        add_trafo_connection(net2, net2.trafo.at[0, 'hv_bus'], "3W")

    pp.control.create_trafo_characteristics(net1, 'trafo3w', [0], 'vk_hv_percent', [[-2, -1, 0, 1, 2]], [[0.85, 0.9, 1, 1.1, 1.15]])
    pp.control.create_trafo_characteristics(net1, 'trafo3w', [0], 'vkr_hv_percent', [[-2, -1, 0, 1, 2]], [[0.27, 0.28, 0.3, 0.32, 0.33]])
    pp.control.create_trafo_characteristics(net1, 'trafo3w', [0], 'vk_mv_percent', [[-2, -1, 0, 1, 2]], [[0.85, 0.9, 1, 1.1, 1.15]])
    pp.control.create_trafo_characteristics(net1, 'trafo3w', [0], 'vkr_mv_percent', [[-2, -1, 0, 1, 2]], [[0.27, 0.28, 0.3, 0.32, 0.33]])
    pp.control.create_trafo_characteristics(net1, 'trafo3w', [0], 'vk_lv_percent', [[-2, -1, 0, 1, 2]], [[0.85, 0.9, 1, 1.1, 1.15]])
    pp.control.create_trafo_characteristics(net1, 'trafo3w', [0], 'vkr_lv_percent', [[-2, -1, 0, 1, 2]], [[0.27, 0.28, 0.3, 0.32, 0.33]])

    pp.control.SplineCharacteristic(net2, [-2, -1, 0, 1, 2], [0.85, 0.9, 1, 1.1, 1.15])
    pp.control.SplineCharacteristic(net2, [-2, -1, 0, 1, 2], [0.27, 0.28, 0.3, 0.32, 0.33])

    pp.control.TapDependentImpedance(net2, [0], 0, trafotable="trafo3w", output_variable="vk_hv_percent")
    pp.control.TapDependentImpedance(net2, [0], 1, trafotable="trafo3w", output_variable="vkr_hv_percent")
    pp.control.TapDependentImpedance(net2, [0], 0, trafotable="trafo3w", output_variable="vk_mv_percent")
    pp.control.TapDependentImpedance(net2, [0], 1, trafotable="trafo3w", output_variable="vkr_mv_percent")
    pp.control.TapDependentImpedance(net2, [0], 0, trafotable="trafo3w", output_variable="vk_lv_percent")
    pp.control.TapDependentImpedance(net2, [0], 1, trafotable="trafo3w", output_variable="vkr_lv_percent")

    pp.runpp(net1)
    pp.runpp(net2, run_control=True)

    assert_res_equal(net1, net2)


def test_undefined_tap_dependent_impedance_characteristics():
    # if some characteristic per 1 trafo are undefined, but at least 1 is defined -> OK
    # if all characteristic per 1 trafo are undefined -> raise error
    net = create_net()
    pp.control.create_trafo_characteristics(net, 'trafo', [0], 'vk_percent', [[-2, -1, 0, 1, 2]], [[5, 5.2, 6, 6.8, 7]])
    pp.control.create_trafo_characteristics(net, 'trafo', [0], 'vkr_percent', [[-2, -1, 0, 1, 2]], [[1.3, 1.4, 1.44, 1.5, 1.6]])
    pp.control.create_trafo_characteristics(net, 'trafo', [1], 'vk_percent', [[-2, -1, 0, 1, 2]], [[5, 5.2, 6, 6.8, 7]])

    # does not raise error
    pp.runpp(net)

    # this will raise error
    net.trafo.at[1, "vk_percent_characteristic"] = None
    with pytest.raises(UserWarning):
        pp.runpp(net)


def test_undefined_tap_dependent_impedance_characteristics_trafo3w():
    # if some characteristic per 1 trafo are undefined, but at least 1 is defined -> OK
    # if all characteristic per 1 trafo are undefined -> raise error
    net = create_net()
    add_trafo_connection(net, 1, "3W")
    add_trafo_connection(net, 1, "3W")
    net2 = create_net()
    add_trafo_connection(net2, 1, "3W")
    add_trafo_connection(net2, 1, "3W")

    pp.control.create_trafo_characteristics(net, 'trafo3w', [0, 1], 'vk_mv_percent', [[-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]], [[0.7, 0.9, 1, 1.1, 1.3], [0.7, 0.9, 1, 1.1, 1.3]])
    pp.control.create_trafo_characteristics(net, 'trafo3w', [0, 1], 'vkr_mv_percent', [[-2, -1, 0, 1, 2], [-2, -1, 0, 1, 2]], [[0.3, 0.45, 0.5, 0.55, 0.7], [0.3, 0.45, 0.5, 0.55, 0.7]])

    pp.control.Characteristic(net2, [-2, -1, 0, 1, 2], [0.7, 0.9, 1, 1.1, 1.3])
    pp.control.Characteristic(net2, [-2, -1, 0, 1, 2], [0.3, 0.45, 0.5, 0.55, 0.7])

    pp.control.TapDependentImpedance(net2, [0], 0, trafotable="trafo3w", output_variable="vk_mv_percent")
    pp.control.TapDependentImpedance(net2, [0], 1, trafotable="trafo3w", output_variable="vkr_mv_percent")
    pp.control.TapDependentImpedance(net2, [1], 0, trafotable="trafo3w", output_variable="vk_mv_percent")
    pp.control.TapDependentImpedance(net2, [1], 1, trafotable="trafo3w", output_variable="vkr_mv_percent")

    pp.runpp(net)
    pp.runpp(net2, run_control=True)
    assert_res_equal(net, net2)

    net.trafo3w.at[0, "vk_mv_percent_characteristic"] = None
    pp.runpp(net)
    net2.controller.at[0, "in_service"] = False
    pp.runpp(net2, run_control=True)
    assert_res_equal(net, net2)

    net.trafo3w.at[0, "vkr_mv_percent_characteristic"] = None
    net2.controller.at[1, "in_service"] = False
    with pytest.raises(UserWarning):
        pp.runpp(net)

    net.trafo3w.at[0, "tap_dependent_impedance"] = False
    pp.runpp(net)
    pp.runpp(net2, run_control=True)
    assert_res_equal(net, net2)


def test_ext_grid(result_test_network, v_tol=1e-6, va_tol=1e-2, i_tol=1e-6, s_tol=5e-3, l_tol=1e-3):
    net = result_test_network
    runpp_with_consistency_checks(net, calculate_voltage_angles=True)
    buses = net.bus[net.bus.zone == "test_ext_grid"]
    b2 = buses.index[1]
    ext_grids = [
        x for x in net.ext_grid.index if net.ext_grid.bus[x] in buses.index]
    eg1 = ext_grids[0]
    eg2 = ext_grids[1]
    # results from powerfactory
    p1 = 5.6531650
    q1 = -2.1074499

    v2 = 1.015506741
    va2 = 1.47521433

    p2 = 5.8377758
    q2 = -2.7786795

    assert abs(net.res_ext_grid.p_mw.at[eg1] - (-p1))
    assert abs(net.res_ext_grid.q_mvar.at[eg1] - (-q1))

    assert abs(net.res_ext_grid.p_mw.at[eg2] - (-p2))
    assert abs(net.res_ext_grid.q_mvar.at[eg2] - (-q2))

    assert abs(net.res_bus.vm_pu.at[b2] - v2) < v_tol
    assert abs(net.res_bus.va_degree.at[b2] - va2) < va_tol


def test_ward(result_test_network, v_tol=1e-6, i_tol=1e-6, s_tol=5e-3, l_tol=1e-3):
    net = result_test_network
    buses = net.bus[net.bus.zone == "test_ward"]
    wards = [x for x in net.ward.index if net.ward.bus[x] in buses.index]
    b2 = buses.index[1]
    w1 = wards[0]
    # powerfactory results
    pw = -1.7046146
    qw = -1.3042294
    u = 1.00192121

    assert abs(net.res_bus.vm_pu.loc[b2] - u) < v_tol
    assert abs(net.res_ward.p_mw.loc[w1] - (-pw)) < s_tol
    assert abs(net.res_ward.q_mvar.loc[w1] - (-qw)) < s_tol


def test_ward_split(result_test_network, v_tol=1e-6, i_tol=1e-6, s_tol=5e-3, l_tol=1e-3):
    net = result_test_network
    buses = net.bus[net.bus.zone == "test_ward_split"]
    wards = [x for x in net.ward.index if net.ward.bus[x] in buses.index]
    b2 = buses.index[1]
    w1 = wards[0]
    w2 = wards[1]
    # powerfactory results
    pw = -1.7046146
    qw = -1.3042294
    u = 1.00192121

    assert abs(net.res_bus.vm_pu.at[b2] - u)
    assert abs(net.res_ward.p_mw.loc[[w1, w2]].sum() - (-pw))
    assert abs(net.res_ward.q_mvar.loc[[w1, w2]].sum() - (-qw))
    #


def test_xward(result_test_network, v_tol=1e-6, i_tol=1e-6, s_tol=5e-3, l_tol=1e-3):
    net = result_test_network
    buses = net.bus[net.bus.zone == "test_xward"]
    xwards = [x for x in net.xward.index if net.xward.bus[x] in buses.index]
    b2 = buses.index[1]
    xw1 = xwards[0]
    xw2 = xwards[1]  # Out of servic xward
    #    powerfactory result for 1 xward
    u = 1.00308684
    pxw = -1.7210380
    qxw = -0.9759919
    #
    assert abs(net.res_bus.vm_pu.at[b2] - u) < v_tol
    assert abs(net.res_xward.p_mw.at[xw1] - (-pxw)) < s_tol
    assert abs(net.res_xward.q_mvar.at[xw1] - (-qxw)) < s_tol

    assert abs(net.res_bus.vm_pu.at[b2] - u) < 1e-2
    assert abs(net.res_xward.p_mw.loc[[xw1, xw2]].sum() - (-pxw)) < s_tol
    assert abs(net.res_xward.q_mvar.loc[[xw1, xw2]].sum() - (-qxw)) < s_tol


def test_xward_combination(result_test_network, v_tol=1e-6, i_tol=1e-6, s_tol=5e-3, l_tol=1e-3):
    net = result_test_network
    buses = net.bus[net.bus.zone == "test_xward_combination"]
    xwards = [x for x in net.xward.index if net.xward.bus[x] in buses.index]
    b2 = buses.index[1]
    xw1 = xwards[0]
    xw3 = xwards[2]

    # powerfactory result for 2 active xwards
    u = 0.99568034
    pxw1 = -1.7071216
    pxw3 = -1.7071216

    qxw1 = -0.9187316
    qxw3 = -0.9187316

    assert abs(net.res_bus.vm_pu.at[b2] - u) < 1e-2
    assert abs(net.res_xward.p_mw.at[xw1] - (-pxw1)) < s_tol
    assert abs(net.res_xward.q_mvar.at[xw1] - (-qxw1)) < s_tol

    assert abs(net.res_xward.p_mw.at[xw3] - (-pxw3)) < s_tol
    assert abs(net.res_xward.q_mvar.at[xw3] - (-qxw3)) < s_tol


def test_gen(result_test_network, v_tol=1e-6, i_tol=1e-6, s_tol=5e-3, l_tol=1e-3):
    net = result_test_network
    buses = net.bus[net.bus.zone == "test_gen"]
    gens = [x for x in net.gen.index if net.gen.bus[x] in buses.index]
    b2 = buses.index[1]
    b3 = buses.index[2]
    g1 = gens[0]
    # powerfactory results
    q = 0.260660
    u2 = 1.00584636
    vm_set_pu = 1.0

    assert abs(net.res_bus.vm_pu.at[b2] - u2) < v_tol
    assert abs(net.res_bus.vm_pu.at[b3] - vm_set_pu) < v_tol
    assert abs(net.res_gen.q_mvar.at[g1] - (-q)) < s_tol


def test_enforce_qlims(result_test_network, v_tol=1e-6, i_tol=1e-6, s_tol=5e-3, l_tol=1e-3):
    net = result_test_network
    buses = net.bus[net.bus.zone == "test_enforce_qlims"]
    gens = [x for x in net.gen.index if net.gen.bus[x] in buses.index]
    b2 = buses.index[1]
    b3 = buses.index[2]
    g1 = gens[0]

    # enforce reactive power limits
    runpp_with_consistency_checks(net, enforce_q_lims=True)

    # powerfactory results
    u2 = 1.00607194
    u3 = 1.00045091

    assert abs(net.res_bus.vm_pu.at[b2] - u2) < v_tol
    assert abs(net.res_bus.vm_pu.at[b3] - u3) < v_tol
    assert abs(net.res_gen.q_mvar.at[g1] - net.gen.min_q_mvar.at[g1]) < s_tol


def test_trafo3w(result_test_network, v_tol=1e-6, i_tol=1e-6, s_tol=2e-2, l_tol=1e-3):
    net = result_test_network
    buses = net.bus[net.bus.zone == "test_trafo3w"]
    trafos = [x for x in net.trafo3w.index if net.trafo3w.hv_bus[
        x] in buses.index]
    runpp_with_consistency_checks(net, trafo_model="pi")
    b2 = buses.index[1]
    b3 = buses.index[2]
    b4 = buses.index[3]
    t3 = trafos[0]

    uhv = 1.010117166
    umv = 0.955501331
    ulv = 0.940630980

    load = 37.21
    qhv = 0.00164375
    qmv = 0
    qlv = 0

    ihv = 0.00858590198
    imv = 0.20141269123
    ilv = 0.15344761586

    phv = 0.30043
    pmv = -0.200
    plv = -0.100


    assert abs((net.res_bus.vm_pu.at[b2] - uhv)) < v_tol
    assert abs((net.res_bus.vm_pu.at[b3] - umv)) < v_tol
    assert abs((net.res_bus.vm_pu.at[b4] - ulv)) < v_tol

    assert abs((net.res_trafo3w.loading_percent.at[t3] - load)) < l_tol

    assert abs((net.res_trafo3w.p_hv_mw.at[t3] - phv)) < s_tol
    assert abs((net.res_trafo3w.p_mv_mw.at[t3] - pmv)) < s_tol
    assert abs((net.res_trafo3w.p_lv_mw.at[t3] - plv)) < s_tol

    assert abs((net.res_trafo3w.q_hv_mvar.at[t3] - qhv)) < s_tol
    assert abs((net.res_trafo3w.q_mv_mvar.at[t3] - qmv)) < s_tol
    assert abs((net.res_trafo3w.q_lv_mvar.at[t3] - qlv)) < s_tol

    assert abs((net.res_trafo3w.i_hv_ka.at[t3] - ihv)) < i_tol
    assert abs((net.res_trafo3w.i_mv_ka.at[t3] - imv)) < i_tol
    assert abs((net.res_trafo3w.i_lv_ka.at[t3] - ilv)) < i_tol

    runpp_with_consistency_checks(net, trafo_model="pi",trafo3w_losses='star')

    #Test results Integral:
    uhv = 1.01011711678
    umv = 0.95550024145
    ulv = 0.94062989256

    load = 37.209
    qhv = 0.001660
    qmv = 0
    qlv = 0

    ihv = 0.00858591110
    imv = 0.20141290445
    ilv = 0.15344776975

    phv = 0.30043
    pmv = -0.200
    plv = -0.100

    assert abs((net.res_bus.vm_pu.at[b2] - uhv)) < v_tol
    assert abs((net.res_bus.vm_pu.at[b3] - umv)) < v_tol
    assert abs((net.res_bus.vm_pu.at[b4] - ulv)) < v_tol

    assert abs((net.res_trafo3w.loading_percent.at[t3] - load)) < l_tol

    assert abs((net.res_trafo3w.p_hv_mw.at[t3] - phv)) < s_tol
    assert abs((net.res_trafo3w.p_mv_mw.at[t3] - pmv)) < s_tol
    assert abs((net.res_trafo3w.p_lv_mw.at[t3] - plv)) < s_tol

    assert abs((net.res_trafo3w.q_hv_mvar.at[t3] - qhv)) < s_tol
    assert abs((net.res_trafo3w.q_mv_mvar.at[t3] - qmv)) < s_tol
    assert abs((net.res_trafo3w.q_lv_mvar.at[t3] - qlv)) < s_tol

    assert abs((net.res_trafo3w.i_hv_ka.at[t3] - ihv)) < i_tol
    assert abs((net.res_trafo3w.i_mv_ka.at[t3] - imv)) < i_tol
    assert abs((net.res_trafo3w.i_lv_ka.at[t3] - ilv)) < i_tol


def test_impedance(result_test_network, v_tol=1e-6, i_tol=1e-6, s_tol=5e-3, l_tol=1e-3):
    net = result_test_network
    buses = net.bus[net.bus.zone == "test_impedance"]
    impedances = [
        x for x in net.impedance.index if net.impedance.from_bus[x] in buses.index]
    runpp_with_consistency_checks(net)
    buses = net.bus[net.bus.zone == "test_impedance"]
    impedances = [x for x in net.impedance.index if net.impedance.from_bus[x] in buses.index]
    runpp_with_consistency_checks(net, trafo_model="t", numba=True)
    b2 = buses.index[1]
    b3 = buses.index[2]
    imp1 = impedances[0]

    # powerfactory results
    ifrom = 0.0444417
    ito = 0.0029704

    pfrom = 1.1237008
    qfrom = 1.0618504

    pto = -1.000
    qto = -0.500

    u2 = 1.004242894
    u3 = 0.987779091

    assert abs(net.res_impedance.p_from_mw.at[imp1] - pfrom) < s_tol
    assert abs(net.res_impedance.p_to_mw.at[imp1] - pto) < s_tol
    assert abs(net.res_impedance.q_from_mvar.at[imp1] - qfrom) < s_tol
    assert abs(net.res_impedance.q_to_mvar.at[imp1] - qto) < s_tol
    assert abs(net.res_impedance.i_from_ka.at[imp1] - ifrom) < i_tol
    assert abs(net.res_impedance.i_to_ka.at[imp1] - ito) < i_tol

    assert abs(net.res_bus.vm_pu.at[b2] - u2) < v_tol
    assert abs(net.res_bus.vm_pu.at[b3] - u3) < v_tol


def test_bus_bus_switch(result_test_network, v_tol=1e-6, i_tol=1e-6, s_tol=5e-3, l_tol=1e-3):
    net = result_test_network
    buses = net.bus[net.bus.zone == "test_bus_bus_switch"]
    b2 = buses.index[1]
    b3 = buses.index[2]

    # powerfactory voltage
    v2 = 0.982264132
    assert abs(net.res_bus.vm_pu.at[b2] - v2) < v_tol
    assert abs(net.res_bus.vm_pu.at[b3] - v2) < v_tol
    assert abs(net.res_bus.vm_pu.at[b2] == net.res_bus.vm_pu.at[b2])


def test_enforce_q_lims(v_tol=1e-6, i_tol=1e-6, s_tol=5e-3, l_tol=1e-3):
    """ Test for enforce_q_lims loadflow option
    """
    net = pp.create_empty_network()
    net = add_test_gen(net)
    pp.runpp(net)
    buses = net.bus[net.bus.zone == "test_gen"]
    gens = [x for x in net.gen.index if net.gen.bus[x] in buses.index]
    #    b1=buses.index[0]
    b2 = buses.index[1]
    b3 = buses.index[2]
    g1 = gens[0]
    q = -0.260660
    u2 = 1.00584636
    vm_set_pu = 1.0
    assert abs(net.res_bus.vm_pu.at[b2] - u2) < v_tol
    assert abs(net.res_bus.vm_pu.at[b3] - vm_set_pu) < v_tol
    assert abs(net.res_gen.q_mvar.at[g1] - q) < s_tol

    # test_enforce_qlims
    net = add_test_enforce_qlims(net)

    pp.runpp(net, enforce_q_lims=True)
    buses = net.bus[net.bus.zone == "test_enforce_qlims"]
    gens = [x for x in net.gen.index if net.gen.bus[x] in buses.index]
    b2 = buses.index[1]
    b3 = buses.index[2]
    g1 = gens[0]
    u2 = 1.00607194
    u3 = 1.00045091
    assert abs(net.res_bus.vm_pu.at[b2] - u2) < 1e-2
    assert abs(net.res_bus.vm_pu.at[b3] - u3) < 1e-2
    assert abs(net.res_gen.q_mvar.at[g1] - net.gen.min_q_mvar.at[g1]) < 1e-2


def test_shunt(result_test_network, v_tol=1e-6, i_tol=1e-6, s_tol=5e-3, l_tol=1e-3):
    net = result_test_network
    buses = net.bus[net.bus.zone == "test_shunt"]
    b2 = buses.index[1]
    shunts = [x for x in net.shunt.index if net.shunt.bus[x] in buses.index]
    s1 = shunts[0]

    u = 1.0177330269
    p = 0.20544
    q = -2.05444

    assert abs(net.res_bus.vm_pu.loc[b2] - u) < v_tol
    assert abs(net.res_shunt.p_mw.loc[s1] - p) < s_tol
    assert abs(net.res_shunt.q_mvar.loc[s1] - q) < s_tol


def test_shunt_split(result_test_network, v_tol=1e-6, i_tol=1e-6, s_tol=5e-3, l_tol=1e-3):
    net = result_test_network
    buses = net.bus[net.bus.zone == "test_shunt_split"]
    b2 = buses.index[1]
    shunts = [x for x in net.shunt.index if net.shunt.bus[x] in buses.index]
    s1 = shunts[0]

    u = 1.015007138
    p = 0.123628741
    q = -1.236287413

    assert abs(net.res_bus.vm_pu.loc[b2] - u) < v_tol
    assert abs(net.res_shunt.p_mw.loc[s1] - p / 2) < s_tol
    assert abs(net.res_shunt.q_mvar.loc[s1] - q / 2) < s_tol


def test_open(result_test_network):
    net = result_test_network
    buses = net.bus[net.bus.zone == "two_open_switches_on_deactive_line"]
    lines = net['line'][in1d(net['line'].from_bus, buses.index) | in1d(net['line'].to_bus, buses.index)]

    assert isnan(net['res_line'].at[lines.index[1], "i_ka"])

if __name__ == "__main__":
    pytest.main(["-xs"])

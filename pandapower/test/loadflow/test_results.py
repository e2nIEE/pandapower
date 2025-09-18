# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import os

import numpy as np
import pandas as pd
import pytest
from numpy import isin, isnan, isclose

from pandapower import pp_dir
from pandapower.create import create_bus, create_load, create_transformer3w_from_parameters, create_transformer, \
    create_empty_network, create_ext_grid, create_line_from_parameters, create_transformer_from_parameters, \
    create_impedance
from pandapower.file_io import from_json
from pandapower.run import runpp
from pandapower.test.conftest import result_test_network
from pandapower.test.consistency_checks import runpp_with_consistency_checks
from pandapower.test.loadflow.result_test_network_generator import add_test_enforce_qlims, \
    add_test_gen

# simple example grid for tap dependent impedance tests:

def add_trafo_connection(net, hv_bus, trafotype="2W"):
    cb = create_bus(net, vn_kv=0.4)
    create_load(net, cb, 0.2, 0.05)

    if trafotype == "3W":
        cbm = create_bus(net, vn_kv=0.9)
        create_load(net, cbm, 0.1, 0.03)
        create_transformer3w_from_parameters(
            net, hv_bus=hv_bus, mv_bus=cbm, lv_bus=cb,
            vn_hv_kv=20., vn_mv_kv=0.9, vn_lv_kv=0.45, sn_hv_mva=0.6, sn_mv_mva=0.5,
            sn_lv_mva=0.4, vk_hv_percent=1., vk_mv_percent=1., vk_lv_percent=1.,
            vkr_hv_percent=0.3, vkr_mv_percent=0.3, vkr_lv_percent=0.3,
            pfe_kw=0.2, i0_percent=0.3, tap_neutral=0., tap_changer_type="Ratio",
            tap_pos=2, tap_step_percent=1., tap_min=-2, tap_max=2)
    else:
        create_transformer(net, hv_bus=hv_bus, lv_bus=cb, std_type="0.25 MVA 20/0.4 kV", tap_pos=2)


def create_net():
    net = create_empty_network()
    vn_kv = 20
    b1 = create_bus(net, vn_kv=vn_kv)
    create_ext_grid(net, b1, vm_pu=1.01)
    b2 = create_bus(net, vn_kv=vn_kv)
    l1 = create_line_from_parameters(net, b1, b2, 12.2, r_ohm_per_km=0.08, x_ohm_per_km=0.12,
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

    net = create_empty_network()
    create_bus(net, 110)
    create_bus(net, 20)
    create_ext_grid(net, 0)
    create_transformer_from_parameters(net, 0, 1, 100, 110, 20, 0.5, 12, 14, 0.5,
                                       tap_side="hv", tap_neutral=0, tap_max=10,
                                       tap_min=-10, tap_step_percent=2, tap_step_degree=0,
                                       tap_pos=0, tap_changer_type="Ratio",
                                       tap2_side="hv", tap2_neutral=0, tap2_max=10,
                                       tap2_min=-10, tap2_step_percent=2, tap2_step_degree=0,
                                       tap2_pos=0, tap2_changer_type="Ratio")

    create_load(net, 1, 10)

    runpp(net)
    net.res_bus


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

    runpp_with_consistency_checks(net, trafo_model="pi", trafo3w_losses='star')

    # Test results Integral:
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


@pytest.mark.parametrize("tap_pos", (-1, 2))
@pytest.mark.parametrize("tap_side", ('hv', 'mv', 'lv'))
@pytest.mark.parametrize("tap_step_degree", (0, 15, 30))
def test_trafo3w_tap(tap_pos, tap_side, tap_step_degree):
    results = pd.read_csv(os.path.join(pp_dir, "test", "test_files", "test_results_files", "trafo_3w_tap_results.csv"),
                          sep=";", decimal=",")

    if results.query("tap_side == @tap_side & tap_pos == @tap_pos & tap_step_degree == @tap_step_degree").empty:
        pytest.skip(f"Skipping combination: tap_side={tap_side}, tap_pos={tap_pos}, tap_step_degree={tap_step_degree}")

    net = from_json(os.path.join(pp_dir, "test", "test_files", "test_trafo3w_tap.json"))  #
    net.trafo3w.loc[0, 'tap_at_star_point'] = False
    net.trafo3w.loc[1, 'tap_at_star_point'] = True

    net.trafo3w.loc[0, "tap_side"] = tap_side
    net.trafo3w.loc[1, "tap_side"] = tap_side
    net.trafo3w.loc[0, "tap_pos"] = tap_pos
    net.trafo3w.loc[1, "tap_pos"] = tap_pos
    net.trafo3w.loc[0, "tap_step_degree"] = tap_step_degree
    net.trafo3w.loc[1, "tap_step_degree"] = tap_step_degree
    runpp(net)

    for index in range(8):
        for variable, tol in zip(("vm_pu", "va_degree"), (1e-6, 1e-3)):
            assert np.isclose(net.res_bus.at[index, variable], results.query(
                "tap_side==@tap_side & tap_pos==@tap_pos & tap_step_degree==@tap_step_degree &"
                "index==@index & element=='bus' & variable==@variable").value,
                              rtol=0,
                              atol=tol), f"failed for bus {index=}, {variable}, value {net.res_bus.at[index, variable]}"


@pytest.mark.parametrize("tap_pos", (2, 5))
@pytest.mark.parametrize("tap_side", ('hv', 'mv', 'lv'))
@pytest.mark.parametrize("tap_step_degree", (0, 15, 30))
def test_trafo3w_tap_neutral_not_zero(tap_pos, tap_side, tap_step_degree):
    results = pd.read_csv(
        os.path.join(pp_dir, "test", "test_files", "test_results_files", "trafo_3w_tap_results_neutral_not_zero.csv"),
        sep=";", decimal=",")

    if results.query("tap_side == @tap_side & tap_pos == @tap_pos & tap_step_degree == @tap_step_degree").empty:
        pytest.skip(f"Skipping combination: tap_side={tap_side}, tap_pos={tap_pos}, tap_step_degree={tap_step_degree}")

    net = from_json(os.path.join(pp_dir, "test", "test_files", "test_trafo3w_tap.json"))  #

    net.trafo3w.loc[[0, 1], 'tap_min'] += 3
    net.trafo3w.loc[[0, 1], 'tap_max'] += 3
    net.trafo3w.loc[[0, 1], 'tap_neutral'] += 3

    net.trafo3w.loc[0, 'tap_at_star_point'] = False
    net.trafo3w.loc[1, 'tap_at_star_point'] = True

    net.trafo3w.loc[0, "tap_side"] = tap_side
    net.trafo3w.loc[1, "tap_side"] = tap_side
    net.trafo3w.loc[0, "tap_pos"] = tap_pos
    net.trafo3w.loc[1, "tap_pos"] = tap_pos
    net.trafo3w.loc[0, "tap_step_degree"] = tap_step_degree
    net.trafo3w.loc[1, "tap_step_degree"] = tap_step_degree
    runpp(net)

    for index in range(8):
        for variable, tol in zip(("vm_pu", "va_degree"), (1e-6, 1e-3)):
            assert np.isclose(net.res_bus.at[index, variable], results.query(
                "tap_side==@tap_side & tap_pos==@tap_pos & tap_step_degree==@tap_step_degree &"
                "index==@index & element=='bus' & variable==@variable").value,
                              rtol=0,
                              atol=tol), f"failed for bus {index=}, {variable}, value {net.res_bus.at[index, variable]}"


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

    for col in ("p_from_mw", "p_to_mw", "q_from_mvar", "q_to_mvar", "i_ka", "loading_percent"):
        assert col in net.res_switch

    assert isnan(net.res_switch.p_from_mw).all()

    # now test with some switches that have non-zero impedance:
    net.switch.loc[[2, 3], "z_ohm"] = 1e-3
    runpp(net)

    p_from_ref = net.res_ext_grid.p_mw[12]
    p_to_ref = - net.res_line.p_from_mw[28]
    q_from_ref = net.res_ext_grid.q_mvar[12]
    q_to_ref = - net.res_line.q_from_mvar[28]

    assert isclose(net.res_switch.p_from_mw.at[2], p_from_ref, rtol=0, atol=1e-6)
    assert isclose(net.res_switch.p_to_mw.at[2], p_to_ref, rtol=0, atol=1e-6)
    assert isclose(net.res_switch.q_from_mvar.at[2], q_from_ref, rtol=0, atol=1e-6)
    assert isclose(net.res_switch.q_to_mvar.at[2], q_to_ref, rtol=0, atol=1e-6)

    # also test with open switch that have non-zero impedance:
    net.switch.loc[[2], "closed"] = False
    runpp(net)

    assert isnan(net.res_switch.p_from_mw[2])


def test_enforce_q_lims(v_tol=1e-6, i_tol=1e-6, s_tol=5e-3, l_tol=1e-3):
    """ Test for enforce_q_lims loadflow option
    """
    net = create_empty_network()
    net = add_test_gen(net)
    runpp(net)
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

    runpp(net, enforce_q_lims=True)
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
    lines = net['line'][isin(net['line'].from_bus, buses.index) | isin(net['line'].to_bus, buses.index)]

    assert isnan(net['res_line'].at[lines.index[1], "i_ka"])


def test_impedance_g_b():
    net = create_empty_network(sn_mva=100)
    create_bus(net, 110)
    create_bus(net, 20)
    create_ext_grid(net, 0)
    create_impedance(net, 0, 1, rft_pu=0.002, xft_pu=0.004, rtf_pu=0.005, xtf_pu=0.008,
                     gf_pu=0.001, bf_pu=0.003, gt_pu=0.006, bt_pu=0.007, sn_mva=10)
    create_load(net, 1, 20, 4)

    runpp_with_consistency_checks(net, tolerance_mva=1e-12)

    assert np.allclose(net.res_bus.vm_pu, [1.000000, 0.986544], rtol=0, atol=1e-6)
    assert np.allclose(net.res_bus.va_degree, [0.000000, -0.817801], rtol=0, atol=1e-6)
    assert np.allclose(net.res_impedance.p_from_mw, [41.728311], rtol=0, atol=1e-5)
    assert np.allclose(net.res_impedance.q_from_mvar, [13.002714], rtol=0, atol=1e-5)
    assert np.allclose(net.res_impedance.p_to_mw, [-20.0], rtol=0, atol=1e-5)
    assert np.allclose(net.res_impedance.q_to_mvar, [-4.0], rtol=0, atol=1e-5)


def test_trafo_unequal_r_x_hv_lv():
    net = create_empty_network(sn_mva=10)
    create_bus(net, 110)
    create_bus(net, 20)
    create_ext_grid(net, 0)
    create_transformer_from_parameters(net, 0, 1, 150, 120, 19,
                                       1, 3, 20, 0.12, tap_side="hv",
                                       tap_neutral=0, tap_max=2, tap_min=-2, tap_step_percent=5, tap_step_degree=30,
                                       tap_pos=-2, tap_changer_type="Ratio", leakage_resistance_ratio_hv=0.6,
                                       leakage_reactance_ratio_hv=0.2)
    create_load(net, 1, 100, 20)

    runpp_with_consistency_checks(net)
    assert np.allclose(net.res_bus.vm_pu, [1.000000, 0.941818], rtol=0, atol=1e-6)
    assert np.allclose(net.res_bus.va_degree, [0.000000, 2.122982], rtol=0, atol=1e-6)
    assert np.allclose(net.res_trafo.p_hv_mw, [100.725785], rtol=0, atol=1e-5)
    assert np.allclose(net.res_trafo.q_hv_mvar, [22.173582], rtol=0, atol=1e-4)
    assert np.allclose(net.res_trafo.p_lv_mw, [-99.999992], rtol=0, atol=1e-5)
    assert np.allclose(net.res_trafo.q_lv_mvar, [-19.999978], rtol=0, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import os

import numpy as np
import pytest

import pandapower as pp
import pandapower.networks as nw
from pandapower.estimation import chi2_analysis, remove_bad_data, estimate


def test_2bus():
    # 1. Create network
    net = pp.create_empty_network()
    pp.create_bus(net, name="bus1", vn_kv=1.)
    pp.create_bus(net, name="bus2", vn_kv=1.)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=1,x_ohm_per_km=0.5, c_nf_per_km=0,
                                   max_i_ka=1)

    pp.create_measurement(net, "p", "line", 0.0111e3, 0.05e3, 0, element=0)  # p12
    pp.create_measurement(net, "q", "line", 0.06e3, 0.05e3, 0, element=0)  # q12

    pp.create_measurement(net, "v", "bus", 1.019, 0.01, bus=0)  # u1
    pp.create_measurement(net, "v", "bus", 1.04, 0.01, bus=1)   # u2

    # 2. Do state estimation
    success = estimate(net, init='flat')

    v_result = net.res_bus_est.vm_pu.values
    delta_result = net.res_bus_est.va_degree.values

    target_v = np.array([[1.02083378, 1.03812899]])
    diff_v = target_v - v_result
    target_delta = np.array([[0.0, 3.11356604]])
    diff_delta = target_delta - delta_result

    assert success
    assert (np.nanmax(abs(diff_v)) < 1e-6)
    assert (np.nanmax(abs(diff_delta)) < 1e-6)


def test_3bus():
    # 1. Create network
    net = pp.create_empty_network()
    pp.create_bus(net, name="bus1", vn_kv=1.)
    pp.create_bus(net, name="bus2", vn_kv=1.)
    pp.create_bus(net, name="bus3", vn_kv=1.)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=0.7, x_ohm_per_km=0.2, c_nf_per_km=0,
                                   max_i_ka=1)
    pp.create_line_from_parameters(net, 0, 2, 1, r_ohm_per_km=0.8, x_ohm_per_km=0.8, c_nf_per_km=0,
                                   max_i_ka=1)
    pp.create_line_from_parameters(net, 1, 2, 1, r_ohm_per_km=1, x_ohm_per_km=0.6, c_nf_per_km=0,
                                   max_i_ka=1)

    pp.create_measurement(net, "p", "line", -0.0011e3, 0.01e3, bus=0, element=0)  # p12
    pp.create_measurement(net, "q", "line", 0.024e3, 0.01e3, bus=0, element=0)    # q12

    pp.create_measurement(net, "p", "bus", 0.018e3, 0.01e3, bus=2)  # p3
    pp.create_measurement(net, "q", "bus", -0.1e3, 0.01e3, bus=2)   # q3

    pp.create_measurement(net, "v", "bus", 1.08, 0.05, 0)   # u1
    pp.create_measurement(net, "v", "bus", 1.015, 0.05, 2)  # u3

    # 2. Do state estimation
    success = estimate(net, init='flat')
    v_result = net.res_bus_est.vm_pu.values
    delta_result = net.res_bus_est.va_degree.values

    target_v = np.array([1.0627, 1.0589, 1.0317])
    diff_v = target_v - v_result
    target_delta = np.array([0., 0.8677, 3.1381])
    diff_delta = target_delta - delta_result

    assert success
    assert (np.nanmax(abs(diff_v)) < 1e-4)
    assert (np.nanmax(abs(diff_delta)) < 1e-4)


def test_3bus_with_bad_data():
    # 1. Create 3-bus-network
    net = pp.create_empty_network()
    pp.create_bus(net, name="bus1", vn_kv=1.)
    pp.create_bus(net, name="bus2", vn_kv=1.)
    pp.create_bus(net, name="bus3", vn_kv=1.)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=0.7, x_ohm_per_km=0.2, c_nf_per_km=0,
                                   max_i_ka=1)
    pp.create_line_from_parameters(net, 0, 2, 1, r_ohm_per_km=0.8, x_ohm_per_km=0.8, c_nf_per_km=0,
                                   max_i_ka=1)
    pp.create_line_from_parameters(net, 1, 2, 1, r_ohm_per_km=1, x_ohm_per_km=0.6, c_nf_per_km=0,
                                   max_i_ka=1)

    pp.create_measurement(net, "p", "line", -0.0011e3, 0.01e3, bus=0, element=0)  # Pline (bus 1 -> bus 2) at bus 1
    pp.create_measurement(net, "q", "line", 0.024e3, 0.01e3, bus=0, element=0)    # Qline (bus 1 -> bus 2) at bus 1

    pp.create_measurement(net, "p", "bus", 0.018e3, 0.01e3, bus=2)  # P at bus 3
    pp.create_measurement(net, "q", "bus", -0.1e3, 0.01e3, bus=2)   # Q at bus 3

    pp.create_measurement(net, "v", "bus", 1.08, 0.05, bus=0)   # V at bus 1
    pp.create_measurement(net, "v", "bus", 1.015, 0.05, bus=2)  # V at bus 3

    # create false voltage measurement for testing bad data detection (-> should be removed)
    pp.create_measurement(net, "v", "bus", 1.3, 0.05, bus=1)   # V at bus 2

    # 2. Do chi2-test
    bad_data_detected = chi2_analysis(net, init='flat')

    # 3. Perform rn_max_test
    success_rn_max = remove_bad_data(net, init='flat')
    v_est_rn_max = net.res_bus_est.vm_pu.values
    delta_est_rn_max = net.res_bus_est.va_degree.values

    target_v = np.array([1.0627, 1.0589, 1.0317])
    diff_v = target_v - v_est_rn_max
    target_delta = np.array([0., 0.8677, 3.1381])
    diff_delta = target_delta - delta_est_rn_max

    assert bad_data_detected
    assert success_rn_max
    assert (np.nanmax(abs(diff_v)) < 1e-4)
    assert (np.nanmax(abs(diff_delta)) < 1e-4)


def test_3bus_with_out_of_service_bus():
    # Test case from book "Power System State Estimation", A. Abur, A. G. Exposito, p. 20ff.
    # S_ref = 1 MVA (PP standard)
    # V_ref = 1 kV
    # Z_ref = 1 Ohm

    # The example only had per unit values, but pandapower expects kV, MVA, kW, kVar
    # Measurements should be in kW/kVar/A - Voltage in p.u.

    # 1. Create network
    net = pp.create_empty_network()
    pp.create_bus(net, name="bus1", vn_kv=1.)
    pp.create_bus(net, name="bus2", vn_kv=1.)
    pp.create_bus(net, name="bus3", vn_kv=1.)
    pp.create_bus(net, name="bus4", vn_kv=1., in_service=0)  # out-of-service bus test
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=.01, x_ohm_per_km=.03, c_nf_per_km=0.,
                                   max_i_ka=1)
    pp.create_line_from_parameters(net, 0, 2, 1, r_ohm_per_km=.02, x_ohm_per_km=.05, c_nf_per_km=0.,
                                   max_i_ka=1)
    pp.create_line_from_parameters(net, 1, 2, 1, r_ohm_per_km=.03, x_ohm_per_km=.08, c_nf_per_km=0.,
                                   max_i_ka=1)

    pp.create_measurement(net, "v", "bus", 1.006, .004, bus=0)  # V at bus 1
    pp.create_measurement(net, "v", "bus", .968, .004, bus=1)   # V at bus 2

    pp.create_measurement(net, "p", "bus", -501, 10, 1)  # P at bus 2
    pp.create_measurement(net, "q", "bus", -286, 10, 1)  # Q at bus 2

    pp.create_measurement(net, "p", "line", 888, 8, 0, 0)   # Pline (bus 1 -> bus 2) at bus 1
    pp.create_measurement(net, "p", "line", 1173, 8, 0, 1)  # Pline (bus 1 -> bus 3) at bus 1
    pp.create_measurement(net, "q", "line", 568, 8, 0, 0)   # Qline (bus 1 -> bus 2) at bus 1
    pp.create_measurement(net, "q", "line", 663, 8, 0, 1)   # Qline (bus 1 -> bus 3) at bus 1

    # 2. Do state estimation
    success = estimate(net, init='flat')
    v_result = net.res_bus_est.vm_pu.values
    delta_result = net.res_bus_est.va_degree.values

    target_v = np.array([[0.9996, 0.9741, 0.9438, np.nan]])
    diff_v = target_v - v_result
    target_delta = np.array([[0., -1.2475, -2.7457, np.nan]])
    diff_delta = target_delta - delta_result

    assert success
    assert (np.nanmax(abs(diff_v)) < 1e-4)
    assert (np.nanmax(abs(diff_delta)) < 1e-4)


def test_3bus_with_transformer():
    # 1. Create network
    net = pp.create_empty_network()
    pp.create_bus(net, name="bus1", vn_kv=10.)
    pp.create_bus(net, name="bus2", vn_kv=10.)
    pp.create_bus(net, name="bus3", vn_kv=10.)
    pp.create_bus(net, name="bus4", vn_kv=110.)
    pp.create_ext_grid(net, bus=3, vm_pu = 1.01)
    pp.create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=.01, x_ohm_per_km=.03, c_nf_per_km=0.,
                                   max_i_ka=1)
    pp.create_line_from_parameters(net, 0, 2, 1, r_ohm_per_km=.02, x_ohm_per_km=.05, c_nf_per_km=0.,
                                   max_i_ka=1)
    pp.create_line_from_parameters(net, 1, 2, 1, r_ohm_per_km=.03, x_ohm_per_km=.08, c_nf_per_km=0.,
                                   max_i_ka=1)
    pp.create_transformer(net, 3, 0, std_type="25 MVA 110/10 kV")

    pp.create_load(net, 1, 450, 300)
    pp.create_load(net, 2, 350, 200)
    pp.runpp(net, calculate_voltage_angles=True)

    # net.trafo.shift_degree = 0.

    pp.create_measurement(net, "v", "bus", 1.009, .004, bus=0)  # V at bus 1
    pp.create_measurement(net, "v", "bus", 1.006, .004, bus=1)   # V at bus 2

    pp.create_measurement(net, "p", "bus", -691, 10, bus=0)  # P at bus 1
    pp.create_measurement(net, "q", "bus", -402, 10, bus=0)  # Q at bus 1

    pp.create_measurement(net, "p", "bus", 0., 1.0, bus=2)  # P at bus 3
    pp.create_measurement(net, "q", "bus", 0., 1.0, bus=2)  # Q at bus 3

    pp.create_measurement(net, "p", "line", 400, 8, 0, 0)   # Pline (bus 1 -> bus 2) at bus 1
    pp.create_measurement(net, "p", "line", 100, 8, 0, 1)  # Pline (bus 1 -> bus 3) at bus 1
    pp.create_measurement(net, "q", "line", 250, 8, 0, 0)   # Qline (bus 1 -> bus 2) at bus 1
    pp.create_measurement(net, "q", "line", 50, 8, 0, 1)   # Qline (bus 1 -> bus 3) at bus 1

    pp.create_measurement(net, "p", "transformer", 1233, 10, bus=3, element=0)  # transformer meas.
    pp.create_measurement(net, "q", "transformer", 700, 10, bus=3, element=0)  # at hv side

    # alterantive:
    # pp.create_measurement(net, "p", "bus", 1233, 10, bus=3)  # P at transformer bus
    # pp.create_measurement(net, "q", "bus", 700, 10, bus=3)  # Q at transformer bus

    # 2. Do state estimation
    success = estimate(net, init='slack', tolerance=2e-5)  # higher tolerance required
    v_result = net.res_bus_est.vm_pu.values
    delta_result = net.res_bus_est.va_degree.values

    diff_v = net.res_bus.vm_pu.values - v_result
    diff_delta = net.res_bus.va_degree.values - delta_result

    assert success
    assert (np.nanmax(abs(diff_v)) < 1e-3)
    assert (np.nanmax(abs(diff_delta)) < 0.09)


def test_3bus_with_2_slacks():
    # load the net which already contains 3 buses
    net = load_3bus_network()
    # add the same net with different slack (no galvanic connection)
    # skip bus index 4 as further stability test
    pp.create_bus(net, name="bus5", vn_kv=1., index=5)
    pp.create_bus(net, name="bus6", vn_kv=1., index=6)
    pp.create_bus(net, name="bus7", vn_kv=1., index=7)
    pp.create_ext_grid(net, 5)
    pp.create_line_from_parameters(net, 5, 6, 1, r_ohm_per_km=.01, x_ohm_per_km=.03, c_nf_per_km=0.,
                                   max_i_ka=1)
    pp.create_line_from_parameters(net, 5, 7, 1, r_ohm_per_km=.02, x_ohm_per_km=.05, c_nf_per_km=0.,
                                   max_i_ka=1)
    pp.create_line_from_parameters(net, 6, 7, 1, r_ohm_per_km=.03, x_ohm_per_km=.08, c_nf_per_km=0.,
                                   max_i_ka=1)

    pp.create_measurement(net, "v", "bus", 1.006, .004, bus=5)  # V at bus 5
    pp.create_measurement(net, "v", "bus", .968, .004, bus=6)   # V at bus 6

    pp.create_measurement(net, "p", "bus", -501, 10, 6)  # P at bus 6
    pp.create_measurement(net, "q", "bus", -286, 10, 6)  # Q at bus 6

    pp.create_measurement(net, "p", "line", 888, 8, 5, 3)   # Pline (bus 1 -> bus 2) at bus 5
    pp.create_measurement(net, "p", "line", 1173, 8, 5, 4)  # Pline (bus 1 -> bus 3) at bus 5
    pp.create_measurement(net, "q", "line", 568, 8, 5, 3)   # Qline (bus 1 -> bus 2) at bus 5
    pp.create_measurement(net, "q", "line", 663, 8, 5, 4)   # Qline (bus 1 -> bus 3) at bus 5

    # 2. Do state estimation
    success = estimate(net, init='flat', maximum_iterations=10)
    v_result = net.res_bus_est.vm_pu.values
    delta_result = net.res_bus_est.va_degree.values

    target_v = np.array([0.9996, 0.9741, 0.9438, np.nan, 0.9996, 0.9741, 0.9438])
    diff_v = target_v - v_result
    target_delta = np.array([0.0, -1.2475469989322963, -2.7457167371166862, np.nan, 0.0,
                             -1.2475469989322963, -2.7457167371166862])
    diff_delta = target_delta - delta_result

    assert success
    assert (np.nanmax(abs(diff_v)) < 1e-4)
    assert (np.nanmax(abs(diff_delta)) < 1e-4)


def test_3bus_with_i_line_measurements():
    np.random.seed(1)
    net = load_3bus_network()
    net.measurement.drop(net.measurement.index, inplace=True)
    pp.create_load(net, 1, p_kw=495.974966, q_kvar=297.749528)
    pp.create_load(net, 2, p_kw=1514.220983, q_kvar=787.528929)
    pp.runpp(net)
    pp.create_measurement(net, "v", "bus", net.res_bus.vm_pu[0] * r(0.01), 0.01, 0)
    pp.create_measurement(net, "v", "bus", net.res_bus.vm_pu[2] * r(0.01), 0.01, 1)
    pp.create_measurement(net, "p", "bus", -net.res_bus.p_kw[0] * r(),
                          max(1.0, abs(0.03 * net.res_bus.p_kw[0])), 0)
    pp.create_measurement(net, "q", "bus", -net.res_bus.q_kvar[0] * r(),
                          max(1.0, abs(0.03 * net.res_bus.q_kvar[0])), 0)
    pp.create_measurement(net, "p", "bus", -net.res_bus.p_kw[2] * r(),
                          max(1.0, abs(0.03 * net.res_bus.p_kw[2])), 2)
    pp.create_measurement(net, "q", "bus", -net.res_bus.q_kvar[2] * r(),
                          max(1.0, abs(0.03 * net.res_bus.q_kvar[2])), 2)
    pp.create_measurement(net, "p", "line", net.res_line.p_from_kw[0] * r(),
                          max(1.0, abs(0.03 * net.res_line.p_from_kw[0])), element=0, bus=0)
    pp.create_measurement(net, "q", "line", net.res_line.q_from_kvar[0] * r(),
                          max(1.0, abs(0.03 * net.res_line.q_from_kvar[0])), element=0, bus=0)
    pp.create_measurement(net, "i", "line", net.res_line.i_from_ka[0] * 1e3 * r(),
                          max(1.0, abs(30 * net.res_line.i_from_ka[0])), element=0, bus=0)
    pp.create_measurement(net, "i", "line", net.res_line.i_from_ka[1] * 1e3 * r(),
                          max(1.0, abs(30 * net.res_line.i_from_ka[1])), element=1, bus=0)
    success = estimate(net, init='flat')

    assert success
    assert (np.nanmax(abs(net.res_bus_est.vm_pu.values - net.res_bus.vm_pu.values)) < 0.045)
    assert (np.nanmax(abs(net.res_bus_est.va_degree.values - net.res_bus.va_degree.values)) < 0.9)


def test_3bus_with_pq_line_from_to_measurements():
    np.random.seed(2017)
    net = load_3bus_network()
    net.measurement.drop(net.measurement.index, inplace=True)
    pp.create_load(net, 1, p_kw=495.974966, q_kvar=297.749528)
    pp.create_load(net, 2, p_kw=1514.220983, q_kvar=787.528929)
    pp.runpp(net)
    pp.create_measurement(net, "v", "bus", net.res_bus.vm_pu[0] * r(0.01), 0.01, 0)
    pp.create_measurement(net, "v", "bus", net.res_bus.vm_pu[2] * r(0.01), 0.01, 1)
    pp.create_measurement(net, "p", "bus", -net.res_bus.p_kw[0] * r(),
                          max(1.0, abs(0.03 * net.res_bus.p_kw[0])), 0)
    pp.create_measurement(net, "q", "bus", -net.res_bus.q_kvar[0] * r(),
                          max(1.0, abs(0.03 * net.res_bus.q_kvar[0])), 0)
    pp.create_measurement(net, "p", "bus", -net.res_bus.p_kw[2] * r(),
                          max(1.0, abs(0.03 * net.res_bus.p_kw[2])), 2)
    pp.create_measurement(net, "q", "bus", -net.res_bus.q_kvar[2] * r(),
                          max(1.0, abs(0.03 * net.res_bus.q_kvar[2])), 2)
    pp.create_measurement(net, "p", "line", net.res_line.p_from_kw[0] * r(),
                          max(1.0, abs(0.03 * net.res_line.p_from_kw[0])), element=0, bus=0)
    pp.create_measurement(net, "q", "line", net.res_line.q_from_kvar[0] * r(),
                          max(1.0, abs(0.03 * net.res_line.q_from_kvar[0])), element=0, bus=0)
    pp.create_measurement(net, "p", "line", net.res_line.p_to_kw[0] * r(),
                          max(1.0, abs(0.03 * net.res_line.p_to_kw[0])), element=0, bus=1)
    pp.create_measurement(net, "q", "line", net.res_line.q_to_kvar[0] * r(),
                          max(1.0, abs(0.03 * net.res_line.q_to_kvar[0])), element=0, bus=1)

    success = estimate(net, init='flat')

    assert success
    assert (np.nanmax(abs(net.res_bus_est.vm_pu.values - net.res_bus.vm_pu.values)) < 0.023)
    assert (np.nanmax(abs(net.res_bus_est.va_degree.values - net.res_bus.va_degree.values)) < 0.12)


def test_cigre_network(init='flat'):
    # 1. create network
    # test the mv ring network with all available voltage measurements and bus powers
    # test if switches and transformer will work correctly with the state estimation
    np.random.seed(123456)
    net = nw.create_cigre_network_mv(with_der=False)
    pp.runpp(net)

    for bus, row in net.res_bus.iterrows():
        pp.create_measurement(net, "v", "bus", row.vm_pu * r(0.01), 0.01, bus)
        # if np.random.randint(0, 4) == 0:
        #    continue
        pp.create_measurement(net, "p", "bus", -row.p_kw * r(), max(1.0, abs(0.03 * row.p_kw)),
                              bus)
        pp.create_measurement(net, "q", "bus", -row.q_kvar * r(), max(1.0, abs(0.03 * row.q_kvar)),
                              bus)

    # 2. Do state estimation
    success = estimate(net, init=init, calculate_voltage_angles=False)
    v_result = net.res_bus_est.vm_pu.values
    delta_result = net.res_bus_est.va_degree.values

    target_v = net.res_bus.vm_pu.values
    diff_v = target_v - v_result
    target_delta = net.res_bus.va_degree.values
    diff_delta = target_delta - delta_result

    assert success
    assert (np.nanmax(abs(diff_v)) < 0.0043)
    assert (np.nanmax(abs(diff_delta)) < 0.17)


def test_cigre_network_with_slack_init():
    test_cigre_network(init='slack')


def test_IEEE_case_9_with_bad_data():
    # 1. Create network
    # test grid: IEEE case 9 (HV)
    # overall: 9 buses, 8 lines
    # choosen standard deviations for different types of measurements:
    # - V: 0.01 p.u.
    # - P: 1000 kW (1kW for no load)
    # - Q: 1000 kVA (1kVA for no load)
    net = pp.networks.case9()

    pp.create_measurement(net, "v", "bus", 0.995, 0.01, bus=1)   # V at bus 1
    pp.create_measurement(net, "v", "bus", 0.992, 0.01, bus=2)   # V at bus 2
    pp.create_measurement(net, "v", "bus", 0.988, 0.01, bus=3)   # V at bus 3
    pp.create_measurement(net, "v", "bus", 0.969, 0.01, bus=4)   # V at bus 4
    pp.create_measurement(net, "v", "bus", 1.019, 0.01, bus=5)   # V at bus 5
    pp.create_measurement(net, "v", "bus", 0.999, 0.01, bus=7)   # V at bus 7
    pp.create_measurement(net, "v", "bus", 0.963, 0.01, bus=8)   # V at bus 8

    pp.create_measurement(net, "p", "bus", 73137., 100., bus=0)   # P at bus 0
    pp.create_measurement(net, "p", "bus", 85133., 100., bus=2)   # P at bus 2
    pp.create_measurement(net, "p", "bus", 0., 1., bus=3)   # P at bus 3
    pp.create_measurement(net, "p", "bus", 0., 1., bus=5)   # P at bus 5
    pp.create_measurement(net, "p", "bus", -99884., 100., bus=6)   # P at bus 6
    pp.create_measurement(net, "p", "bus", 0., 10., bus=7)   # P at bus 7

    pp.create_measurement(net, "q", "bus", 24272., 100., bus=0)   # P at bus 0
    pp.create_measurement(net, "q", "bus", 13969., 100., bus=1)   # P at bus 1
    pp.create_measurement(net, "q", "bus", 4235., 100., bus=2)   # P at bus 2
    pp.create_measurement(net, "q", "bus", 0., 1., bus=3)   # Q at bus 3
    pp.create_measurement(net, "q", "bus", -30177., 100., bus=4)   # Q at bus 4
    pp.create_measurement(net, "q", "bus", 0., 10., bus=5)   # Q at bus 5
    pp.create_measurement(net, "q", "bus", -36856., 100., bus=6)   # Q at bus 6
    pp.create_measurement(net, "q", "bus", 0., 1., bus=7)   # Q at bus 7
    pp.create_measurement(net, "q", "bus", -49673., 100., bus=8)   # Q at bus 8

    # 2. Do state estimation
    success_SE = estimate(net, init='flat')
    v_est_SE = net.res_bus_est.vm_pu.values
    delta_SE = net.res_bus_est.va_degree.values

    # 3. Create false measurement (very close to useful values)
    pp.create_measurement(net, "v", "bus", 0.9, 0.01, bus=0)  # V at bus 0
    # pp.create_measurement(net, "p", "bus", -1, 10., bus=4)

    # 4. Do chi2-test
    bad_data_detected = chi2_analysis(net, init='flat')

    # 5. Perform rn_max_test
    success_rn_max = remove_bad_data(net, init='flat')
    v_est_rn_max = net.res_bus_est.vm_pu.values
    delta_est_rn_max = net.res_bus_est.va_degree.values

    diff_v = v_est_SE - v_est_rn_max
    diff_delta = delta_SE - delta_est_rn_max

    assert success_SE
    assert bad_data_detected
    assert success_rn_max
    assert (np.nanmax(abs(diff_v)) < 1e-5)
    assert (np.nanmax(abs(diff_delta)) < 1e-5)


def test_init_slack_with_multiple_transformers(angles=True):
    np.random.seed(123)
    net = pp.create_empty_network()
    pp.create_bus(net, 220, index=0)
    pp.create_bus(net, 110, index=1)
    pp.create_bus(net, 110, index=2)
    pp.create_bus(net, 110, index=3)
    pp.create_bus(net, 10, index=4)
    pp.create_bus(net, 10, index=5)
    pp.create_bus(net, 10, index=6)
    pp.create_bus(net, 10, index=7, in_service=False)
    pp.create_transformer(net, 3, 7, std_type="63 MVA 110/10 kV", in_service=False)
    pp.create_transformer(net, 3, 4, std_type="63 MVA 110/10 kV")
    pp.create_transformer(net, 0, 1, std_type="100 MVA 220/110 kV")
    pp.create_line(net, 1, 2, 2.0, std_type="N2XS(FL)2Y 1x120 RM/35 64/110 kV")
    pp.create_line(net, 1, 3, 2.0, std_type="N2XS(FL)2Y 1x120 RM/35 64/110 kV")
    pp.create_line(net, 4, 5, 2.0, std_type="NA2XS2Y 1x95 RM/25 12/20 kV")
    pp.create_line(net, 5, 6, 2.0, std_type="NA2XS2Y 1x95 RM/25 12/20 kV")
    pp.create_load(net, 2, 5000, 3300)
    pp.create_load(net, 5, 900, 500)
    pp.create_load(net, 6, 700, 300)
    pp.create_ext_grid(net, bus=0, vm_pu=1.04, va_degree=10., name="Slack 220 kV")
    pp.runpp(net, calculate_voltage_angles=angles)
    for bus, row in net.res_bus[net.bus.in_service == True].iterrows():
        pp.create_measurement(net, "v", "bus", row.vm_pu * r(0.01), 0.01, bus)
        if row.p_kw != 0.:
            continue
        pp.create_measurement(net, "p", "bus", -row.p_kw * r(), max(1.0, abs(0.03 * row.p_kw)),
                              bus)
        pp.create_measurement(net, "q", "bus", -row.q_kvar * r(), max(1.0, abs(0.03 * row.q_kvar)),
                              bus)
    pp.create_measurement(net, "p", "line", net.res_line.p_from_kw[0], 10., bus=1, element=0)
    pp.create_measurement(net, "q", "line", net.res_line.q_from_kvar[0], 10., bus=1, element=0)
    pp.create_measurement(net, "p", "line", net.res_line.p_from_kw[2], 10., bus=4, element=2)
    pp.create_measurement(net, "q", "line", net.res_line.q_from_kvar[2], 10., bus=4, element=2)
    pp.create_measurement(net, "p", "line", net.res_line.p_from_kw[3], 10., bus=5, element=3)
    pp.create_measurement(net, "q", "line", net.res_line.q_from_kvar[3], 10., bus=5, element=3)
    success = estimate(net, init='slack', calculate_voltage_angles=angles)

    # pretty high error for vm_pu (half percent!)
    assert success
    assert (np.nanmax(np.abs(net.res_bus.vm_pu.values - net.res_bus_est.vm_pu.values)) < 0.006)
    assert (np.nanmax(np.abs(net.res_bus.va_degree.values
                             - net.res_bus_est.va_degree.values)) < 0.006)


def test_init_slack_with_multiple_transformers_angles_off():
    test_init_slack_with_multiple_transformers(False)


def test_check_existing_measurements():
    np.random.seed(2017)
    net = pp.create_empty_network()
    pp.create_bus(net, 10.)
    pp.create_bus(net, 10.)
    pp.create_line(net, 0, 1, 0.5, std_type="149-AL1/24-ST1A 10.0")
    m1 = pp.create_measurement(net, "v", "bus", 1.006, .004, 0)
    m2 = pp.create_measurement(net, "v", "bus", 1.006, .004, 0)

    assert m1 == m2
    assert len(net.measurement) == 1
    m3 = pp.create_measurement(net, "v", "bus", 1.006, .004, 0, check_existing=False)
    assert m3 != m2
    assert len(net.measurement) == 2

    m4 = pp.create_measurement(net, "p", "line", -0.0011e3, 0.01e3, bus=0, element=0,
                               check_existing=True)
    m5 = pp.create_measurement(net, "p", "line", -0.0011e3, 0.01e3, bus=0, element=0,
                               check_existing=True)
    assert m4 == m5

    m6 = pp.create_measurement(net, "p", "line", -0.0011e3, 0.01e3, bus=0, element=0,
                               check_existing=False)
    assert m5 != m6


def load_3bus_network():
    folder = os.path.abspath(os.path.dirname(pp.__file__))
    return pp.from_pickle(os.path.join(folder, "test", "estimation", "3bus_wls.p"))


def r(v=0.03):
    return np.random.normal(1.0, v)


if __name__ == '__main__':
    pytest.main(['-xs', __file__])

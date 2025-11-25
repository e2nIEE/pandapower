# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os
from copy import deepcopy

import numpy as np
import pytest

from pandapower import pp_dir
from pandapower.create import create_empty_network, create_bus, create_ext_grid, create_line_from_parameters, \
    create_measurement, create_load, create_transformer, create_line, create_sgen, create_transformer3w, create_switch
from pandapower.estimation import chi2_analysis, remove_bad_data, estimate
from pandapower.file_io import from_json
from pandapower.networks.cigre_networks import create_cigre_network_mv
from pandapower.networks.power_system_test_cases import case9
from pandapower.run import runpp
from pandapower.std_types import create_std_type


def test_2bus():
    # 1. Create network
    net = create_empty_network()
    create_bus(net, name="bus1", vn_kv=1.)
    create_bus(net, name="bus2", vn_kv=1.)
    create_ext_grid(net, 0)
    create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=1, x_ohm_per_km=0.5,
                                c_nf_per_km=0, max_i_ka=1)

    create_measurement(net, "p", "line", 0.0111, 0.05, 0, "from")  # p12
    create_measurement(net, "q", "line", 0.06, 0.05, 0, "from")  # q12

    create_measurement(net, "v", "bus", 1.019, 0.01, 0)  # u1
    create_measurement(net, "v", "bus", 1.04, 0.01, 1)  # u2

    # 2. Do state estimation
    if not estimate(net, init='flat'):
        raise AssertionError("Estimation failed!")

    v_result = net.res_bus_est.vm_pu.values
    delta_result = net.res_bus_est.va_degree.values

    target_v = np.array([[1.02083378, 1.03812899]])
    diff_v = target_v - v_result
    target_delta = np.array([[0.0, 3.11356604]])
    diff_delta = target_delta - delta_result

    if not (np.nanmax(abs(diff_v)) < 1e-6) or not (np.nanmax(abs(diff_delta)) < 1e-6):
        raise AssertionError("Estimation failed!")


def test_3bus():
    # 1. Create network
    net = create_empty_network()
    create_bus(net, name="bus1", vn_kv=1.)
    create_bus(net, name="bus2", vn_kv=1.)
    create_bus(net, name="bus3", vn_kv=1.)
    create_ext_grid(net, 0)
    create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=0.7, x_ohm_per_km=0.2, c_nf_per_km=0,
                                max_i_ka=1)
    create_line_from_parameters(net, 0, 2, 1, r_ohm_per_km=0.8, x_ohm_per_km=0.8, c_nf_per_km=0,
                                max_i_ka=1)
    create_line_from_parameters(net, 1, 2, 1, r_ohm_per_km=1, x_ohm_per_km=0.6, c_nf_per_km=0,
                                max_i_ka=1)

    create_measurement(net, "p", "line", -0.0011, 0.01, 0, "from")  # p12
    create_measurement(net, "q", "line", 0.024, 0.01, 0, "from")  # q12

    create_measurement(net, "p", "bus", -0.018, 0.01, 2)  # p3
    create_measurement(net, "q", "bus", 0.1, 0.01, 2)  # q3

    create_measurement(net, "v", "bus", 1.08, 0.05, 0)  # u1
    create_measurement(net, "v", "bus", 1.015, 0.05, 2)  # u3

    # 2. Do state estimation
    if not estimate(net, init='flat'):
        raise AssertionError("Estimation failed!")
    v_result = net.res_bus_est.vm_pu.values
    delta_result = net.res_bus_est.va_degree.values

    target_v = np.array([1.0627, 1.0589, 1.0317])
    diff_v = target_v - v_result
    target_delta = np.array([0., 0.8677, 3.1381])
    diff_delta = target_delta - delta_result

    if not (np.nanmax(abs(diff_v)) < 1e-4) or \
            not (np.nanmax(abs(diff_delta)) < 1e-4):
        raise AssertionError("Estimation failed!")

    # Backwards check. Use state estimation results for power flow and check for equality
    net.ext_grid.vm_pu = net.res_bus_est.vm_pu.iloc[0]
    create_load(net, 0, net.res_bus_est.p_mw.iloc[0], net.res_bus_est.q_mvar.iloc[0])
    create_load(net, 1, net.res_bus_est.p_mw.iloc[1], net.res_bus_est.q_mvar.iloc[1])
    create_load(net, 2, net.res_bus_est.p_mw.iloc[2], net.res_bus_est.q_mvar.iloc[2])
    _compare_pf_and_se_results(net)


def test_3bus_with_bad_data():
    net = create_empty_network()
    create_bus(net, name="bus1", vn_kv=1.)
    create_bus(net, name="bus2", vn_kv=1.)
    create_bus(net, name="bus3", vn_kv=1.)
    create_ext_grid(net, 0)
    create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=0.7, x_ohm_per_km=0.2, c_nf_per_km=0,
                                max_i_ka=1)
    create_line_from_parameters(net, 0, 2, 1, r_ohm_per_km=0.8, x_ohm_per_km=0.8, c_nf_per_km=0,
                                max_i_ka=1)
    create_line_from_parameters(net, 1, 2, 1, r_ohm_per_km=1, x_ohm_per_km=0.6, c_nf_per_km=0,
                                max_i_ka=1)

    create_measurement(net, "p", "line", -0.0011, 0.01, 0, "from")  # p12
    create_measurement(net, "q", "line", 0.024, 0.01, 0, "from")  # q12

    create_measurement(net, "p", "bus", -0.018, 0.01, 2)  # p3
    create_measurement(net, "q", "bus", 0.1, 0.01, 2)  # q3

    create_measurement(net, "v", "bus", 1.08, 0.05, 0)  # u1
    create_measurement(net, "v", "bus", 1.015, 0.05, 2)  # u3

    # 0. Do chi2-test for corret data
    assert not chi2_analysis(net, init='flat')

    # 1. Create false voltage measurement for testing bad data detection (-> should be removed)
    create_measurement(net, "v", "bus", 1.3, 0.01, 1)  # V at bus 2

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
    if not (np.nanmax(abs(diff_v)) < 1e-4) or \
            not (np.nanmax(abs(diff_delta)) < 1e-4):
        raise AssertionError("Estimation failed!")


def test_3bus_with_out_of_service_bus():
    # Test case from book "Power System State Estimation", A. Abur, A. G. Exposito, p. 20ff.
    # S_ref = 1 MVA (PP standard)
    # V_ref = 1 kV
    # Z_ref = 1 Ohm

    # The example only had per unit values, but pandapower expects kV, MVA, kW, kVar
    # Measurements should be in kW/kVar/A - Voltage in p.u.

    # 1. Create network
    net = create_empty_network()
    create_bus(net, name="bus1", vn_kv=1.)
    create_bus(net, name="bus2", vn_kv=1.)
    create_bus(net, name="bus3", vn_kv=1.)
    create_bus(net, name="bus4", vn_kv=1., in_service=0)  # out-of-service bus test
    create_ext_grid(net, 0)
    create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=.01, x_ohm_per_km=.03, c_nf_per_km=0.,
                                max_i_ka=1)
    create_line_from_parameters(net, 0, 2, 1, r_ohm_per_km=.02, x_ohm_per_km=.05, c_nf_per_km=0.,
                                max_i_ka=1)
    create_line_from_parameters(net, 1, 2, 1, r_ohm_per_km=.03, x_ohm_per_km=.08, c_nf_per_km=0.,
                                max_i_ka=1)

    create_measurement(net, "v", "bus", 1.006, .004, 0)  # V at bus 1
    create_measurement(net, "v", "bus", .968, .004, 1)  # V at bus 2

    create_measurement(net, "p", "bus", .501, .010, 1)  # P at bus 2
    create_measurement(net, "q", "bus", .286, .010, 1)  # Q at bus 2

    create_measurement(net, "p", "line", .888, .008, 0, "from")  # Pline (bus 1 -> bus 2) at bus 1
    create_measurement(net, "p", "line", 1.173, .008, 1, "from")  # Pline (bus 1 -> bus 3) at bus 1
    create_measurement(net, "q", "line", .568, .008, 0, "from")  # Qline (bus 1 -> bus 2) at bus 1
    create_measurement(net, "q", "line", .663, .008, 1, "from")  # Qline (bus 1 -> bus 3) at bus 1

    # 2. Do state estimation
    if not estimate(net, init='flat'):
        raise AssertionError("Estimation failed!")

    v_result = net.res_bus_est.vm_pu.values
    delta_result = net.res_bus_est.va_degree.values

    target_v = np.array([[0.9996, 0.9741, 0.9438, np.nan]])
    diff_v = target_v - v_result
    target_delta = np.array([[0., -1.2475, -2.7457, np.nan]])
    diff_delta = target_delta - delta_result

    if not (np.nanmax(abs(diff_v)) < 1e-4) or \
            not (np.nanmax(abs(diff_delta)) < 1e-4):
        raise AssertionError("Estimation failed!")


def test_3bus_with_transformer():
    np.random.seed(12)

    # 1. Create network
    net = create_empty_network()
    create_bus(net, name="bus1", vn_kv=10.)
    create_bus(net, name="bus2", vn_kv=10.)
    create_bus(net, name="bus3", vn_kv=10.)
    create_bus(net, name="bus4", vn_kv=110.)
    create_ext_grid(net, bus=3, vm_pu=1.01)
    create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=.01, x_ohm_per_km=.03, c_nf_per_km=0.,
                                max_i_ka=1)
    create_line_from_parameters(net, 0, 2, 1, r_ohm_per_km=.02, x_ohm_per_km=.05, c_nf_per_km=0.,
                                max_i_ka=1)
    create_line_from_parameters(net, 1, 2, 1, r_ohm_per_km=.03, x_ohm_per_km=.08, c_nf_per_km=0.,
                                max_i_ka=1)

    create_std_type(net, {"sn_mva": 25, "vn_hv_kv": 110, "vn_lv_kv": 10, "vk_percent": 10.04,
                          "vkr_percent": 0.276, "pfe_kw": 28.51, "i0_percent": 0.073, "shift_degree": 150,
                          "tap_side": "hv", "tap_neutral": 0, "tap_min": -9, "tap_max": 9, "tap_step_degree": 0,
                          "tap_step_percent": 1.5, "tap_changer_type": "Ratio"},
                    "25 MVA 110/10 kV v1.4.3 and older", element="trafo")
    create_transformer(net, 3, 0, std_type="25 MVA 110/10 kV v1.4.3 and older")

    create_load(net, bus=1, p_mw=0.45, q_mvar=0.3)
    create_load(net, bus=2, p_mw=0.35, q_mvar=0.2)

    runpp(net, calculate_voltage_angles=True)

    create_measurement(net, "v", "bus", r2(net.res_bus.vm_pu.iloc[0], .004), .004, element=0)
    create_measurement(net, "v", "bus", r2(net.res_bus.vm_pu.iloc[1], .004), .004, element=1)
    create_measurement(net, "v", "bus", r2(net.res_bus.vm_pu.iloc[3], .004), .004, element=3)

    create_measurement(net, "p", "bus", r2(net.res_bus.p_mw.iloc[1], .01), .01, element=1)
    create_measurement(net, "q", "bus", r2(net.res_bus.q_mvar.iloc[1], .01), .01, element=1)

    create_measurement(net, "p", "bus", r2(net.res_bus.p_mw.iloc[2], .01), .010, element=2)
    create_measurement(net, "q", "bus", r2(net.res_bus.q_mvar.iloc[2], .01), .01, element=2)

    create_measurement(net, "p", "bus", 0., 0.001, element=0)
    create_measurement(net, "q", "bus", 0., 0.001, element=0)

    create_measurement(net, "p", "line", r2(net.res_line.p_from_mw.iloc[0], .008), .008, 0, "from")
    create_measurement(net, "p", "line", r2(net.res_line.p_from_mw.iloc[1], .008), .008, 1, "from")

    create_measurement(net, "p", "trafo", r2(net.res_trafo.p_hv_mw.iloc[0], .01), .01,
                       side="hv", element=0)  # transformer meas.
    create_measurement(net, "q", "trafo", r2(net.res_trafo.q_hv_mvar.iloc[0], .01), .01,
                       side="hv", element=0)  # at hv side

    # 2. Do state estimation
    if not estimate(net, init='slack', tolerance=1e-6, maximum_iterations=10):
        raise AssertionError("Estimation failed!")
    v_result = net.res_bus_est.vm_pu.values
    delta_result = net.res_bus_est.va_degree.values

    diff_v = net.res_bus.vm_pu.values - v_result
    diff_delta = net.res_bus.va_degree.values - delta_result

    if not (np.nanmax(abs(diff_v)) < 6e-4) or \
            not (np.nanmax(abs(diff_delta)) < 8e-4):
        raise AssertionError("Estimation failed!")

    # Backwards check. Use state estimation results for power flow and check for equality
    net.load = net.load.drop(net.load.index)
    net.ext_grid.vm_pu = net.res_bus_est.vm_pu.iloc[net.ext_grid.bus.iloc[0]]
    create_load(net, 0, net.res_bus_est.p_mw.iloc[0], net.res_bus_est.q_mvar.iloc[0])
    create_load(net, 1, net.res_bus_est.p_mw.iloc[1], net.res_bus_est.q_mvar.iloc[1])
    create_load(net, 2, net.res_bus_est.p_mw.iloc[2], net.res_bus_est.q_mvar.iloc[2])

    _compare_pf_and_se_results(net)


def test_3bus_with_2_slacks():
    # load the net which already contains 3 buses
    net = load_3bus_network()
    # add the same net with different slack (no galvanic connection)
    # skip bus index 4 as further stability test
    create_bus(net, name="bus5", vn_kv=1., index=5)
    create_bus(net, name="bus6", vn_kv=1., index=6)
    create_bus(net, name="bus7", vn_kv=1., index=7)
    create_ext_grid(net, 5)
    create_line_from_parameters(net, 5, 6, 1, r_ohm_per_km=.01, x_ohm_per_km=.03, c_nf_per_km=0.,
                                max_i_ka=1)
    create_line_from_parameters(net, 5, 7, 1, r_ohm_per_km=.02, x_ohm_per_km=.05, c_nf_per_km=0.,
                                max_i_ka=1)
    create_line_from_parameters(net, 6, 7, 1, r_ohm_per_km=.03, x_ohm_per_km=.08, c_nf_per_km=0.,
                                max_i_ka=1)

    create_measurement(net, "v", "bus", 1.006, .004, element=5)  # V at bus 5
    create_measurement(net, "v", "bus", .968, .004, element=6)  # V at bus 6

    create_measurement(net, "p", "bus", .501, .010, element=6)  # P at bus 6
    create_measurement(net, "q", "bus", .286, .010, element=6)  # Q at bus 6

    create_measurement(net, "p", "line", .888, .008, 3, "from")  # Pline (bus 5 -> bus 6) at bus 5
    create_measurement(net, "p", "line", 1.173, .008, 4, "from")  # Pline (bus 5 -> bus 7) at bus 5
    create_measurement(net, "q", "line", .568, .008, 3, "from")  # Qline (bus 5 -> bus 6) at bus 5
    create_measurement(net, "q", "line", .663, .008, 4, "from")  # Qline (bus 5 -> bus 7) at bus 5

    # 2. Do state estimation
    if not estimate(net, init='flat', maximum_iterations=10):
        raise AssertionError("Estimation failed!")
    v_result = net.res_bus_est.vm_pu.values
    delta_result = net.res_bus_est.va_degree.values

    target_v = np.array([0.9996, 0.9741, 0.9438, np.nan, 0.9996, 0.9741, 0.9438])
    target_delta = np.array([0.0, -1.2475469989322963, -2.7457167371166862, np.nan, 0.0,
                             -1.2475469989322963, -2.7457167371166862])

    if not np.allclose(v_result, target_v, atol=1e-4, equal_nan=True) or \
            not np.allclose(delta_result, target_delta, atol=1e-4, equal_nan=True):
        raise AssertionError("Estimation failed!")


def test_3bus_with_i_line_measurements():
    np.random.seed(1)
    net = load_3bus_network()
    net.measurement = net.measurement.drop(net.measurement.index)
    create_load(net, 1, p_mw=0.495974966, q_mvar=0.297749528)
    create_load(net, 2, p_mw=1.514220983, q_mvar=0.787528929)
    runpp(net)
    create_measurement(net, "v", "bus", net.res_bus.vm_pu[0] * r(0.01), 0.01, 0)
    create_measurement(net, "v", "bus", net.res_bus.vm_pu[2] * r(0.01), 0.01, 1)
    create_measurement(net, "p", "bus", net.res_bus.p_mw[0] * r(),
                       max(1.0e-3, abs(0.03 * net.res_bus.p_mw[0])), 0)
    create_measurement(net, "q", "bus", net.res_bus.q_mvar[0] * r(),
                       max(1.0e-3, abs(0.03 * net.res_bus.q_mvar[0])), 0)
    create_measurement(net, "p", "bus", net.res_bus.p_mw[2] * r(),
                       max(1.0e-3, abs(0.03 * net.res_bus.p_mw[2])), 2)
    create_measurement(net, "q", "bus", net.res_bus.q_mvar[2] * r(),
                       max(1.0e-3, abs(0.03 * net.res_bus.q_mvar[2])), 2)
    create_measurement(net, "p", "line", net.res_line.p_from_mw[0] * r(),
                       max(1.0e-3, abs(0.03 * net.res_line.p_from_mw[0])), element=0, side="from")
    create_measurement(net, "q", "line", net.res_line.q_from_mvar[0] * r(),
                       max(1.0e-3, abs(0.03 * net.res_line.q_from_mvar[0])), element=0, side="from")
    create_measurement(net, "i", "line", net.res_line.i_from_ka[0] * 1e3 * r(),
                       max(1.0, abs(30 * net.res_line.i_from_ka[0])), element=0, side="from")
    create_measurement(net, "i", "line", net.res_line.i_from_ka[1] * 1e3 * r(),
                       max(1.0, abs(30 * net.res_line.i_from_ka[1])), element=1, side="from")

    if not estimate(net, init='flat'):
        raise AssertionError("Estimation failed!")

    assert (np.nanmax(abs(net.res_bus_est.vm_pu.values - net.res_bus.vm_pu.values)) < 0.045)
    assert (np.nanmax(abs(net.res_bus_est.va_degree.values - net.res_bus.va_degree.values)) < 0.9)


def test_3bus_with_pq_line_from_to_measurements():
    np.random.seed(2017)
    net = load_3bus_network()
    net.measurement = net.measurement.drop(net.measurement.index)
    create_load(net, 1, p_mw=0.495974966, q_mvar=0.297749528)
    create_load(net, 2, p_mw=1.514220983, q_mvar=0.787528929)
    runpp(net)
    create_measurement(net, "v", "bus", net.res_bus.vm_pu[0] * r(0.01), 0.01, 0)
    create_measurement(net, "v", "bus", net.res_bus.vm_pu[2] * r(0.01), 0.01, 1)
    create_measurement(net, "p", "bus", net.res_bus.p_mw[0] * r(),
                       max(1.0e-3, abs(0.03 * net.res_bus.p_mw[0])), 0)
    create_measurement(net, "q", "bus", net.res_bus.q_mvar[0] * r(),
                       max(1.0e-3, abs(0.03 * net.res_bus.q_mvar[0])), 0)
    create_measurement(net, "p", "bus", net.res_bus.p_mw[2] * r(),
                       max(1.0e-3, abs(0.03 * net.res_bus.p_mw[2])), 2)
    create_measurement(net, "q", "bus", net.res_bus.q_mvar[2] * r(),
                       max(1.0e-3, abs(0.03 * net.res_bus.q_mvar[2])), 2)
    create_measurement(net, "p", "line", net.res_line.p_from_mw[0] * r(),
                       max(1.0e-3, abs(0.03 * net.res_line.p_from_mw[0])), element=0, side="from")
    create_measurement(net, "q", "line", net.res_line.q_from_mvar[0] * r(),
                       max(1.0e-3, abs(0.03 * net.res_line.q_from_mvar[0])), element=0, side="from")
    create_measurement(net, "p", "line", net.res_line.p_to_mw[0] * r(),
                       max(1.0e-3, abs(0.03 * net.res_line.p_to_mw[0])), element=0, side="to")
    create_measurement(net, "q", "line", net.res_line.q_to_mvar[0] * r(),
                       max(1.0e-3, abs(0.03 * net.res_line.q_to_mvar[0])), element=0, side="to")

    if not estimate(net, init='flat'):
        raise AssertionError("Estimation failed!")

    assert (np.nanmax(abs(net.res_bus_est.vm_pu.values - net.res_bus.vm_pu.values)) < 0.023)
    assert (np.nanmax(abs(net.res_bus_est.va_degree.values - net.res_bus.va_degree.values)) < 0.12)


def test_3bus_with_side_names():
    np.random.seed(2017)
    net = load_3bus_network()
    net.measurement = net.measurement.drop(net.measurement.index)
    create_load(net, 1, p_mw=0.495974966, q_mvar=0.297749528)
    create_load(net, 2, p_mw=1.514220983, q_mvar=0.787528929)
    runpp(net)
    create_measurement(net, "v", "bus", net.res_bus.vm_pu[0] * r(0.01), 0.01, 0)
    create_measurement(net, "v", "bus", net.res_bus.vm_pu[2] * r(0.01), 0.01, 1)
    create_measurement(net, "p", "bus", net.res_bus.p_mw[0] * r(),
                       max(1.0e-3, abs(0.03 * net.res_bus.p_mw[0])), 0)
    create_measurement(net, "q", "bus", net.res_bus.q_mvar[0] * r(),
                       max(1.0e-3, abs(0.03 * net.res_bus.q_mvar[0])), 0)
    create_measurement(net, "p", "bus", net.res_bus.p_mw[2] * r(),
                       max(1.0e-3, abs(0.03 * net.res_bus.p_mw[2])), 2)
    create_measurement(net, "q", "bus", net.res_bus.q_mvar[2] * r(),
                       max(1.0e-3, abs(0.03 * net.res_bus.q_mvar[2])), 2)
    create_measurement(net, "p", "line", net.res_line.p_from_mw[0] * r(),
                       max(1.0e-3, abs(0.03 * net.res_line.p_from_mw[0])), element=0, side="from")
    create_measurement(net, "q", "line", net.res_line.q_from_mvar[0] * r(),
                       max(1.0e-3, abs(0.03 * net.res_line.q_from_mvar[0])), element=0, side="from")
    create_measurement(net, "p", "line", net.res_line.p_to_mw[0] * r(),
                       max(1.0e-3, abs(0.03 * net.res_line.p_to_mw[0])), element=0, side="to")
    create_measurement(net, "q", "line", net.res_line.q_to_mvar[0] * r(),
                       max(1.0e-3, abs(0.03 * net.res_line.q_to_mvar[0])), element=0, side="to")

    if not estimate(net, init='flat'):
        raise AssertionError("Estimation failed!")

    assert (np.nanmax(abs(net.res_bus_est.vm_pu.values - net.res_bus.vm_pu.values)) < 0.023)
    assert (np.nanmax(abs(net.res_bus_est.va_degree.values - net.res_bus.va_degree.values)) < 0.12)


def test_cigre_network(init='flat'):
    # 1. create network
    # test the mv ring network with all available voltage measurements and bus powers
    # test if switches and transformer will work correctly with the state estimation
    np.random.seed(123456)
    net = create_cigre_network_mv(with_der=False)
    runpp(net)

    for bus, row in net.res_bus.iterrows():
        create_measurement(net, "v", "bus", row.vm_pu * r(0.01), 0.01, bus)
        # if np.random.randint(0, 4) == 0:
        #    continue
        create_measurement(net, "p", "bus", row.p_mw * r(), max(0.001, abs(0.03 * row.p_mw)),
                           bus)
        create_measurement(net, "q", "bus", row.q_mvar * r(), max(0.001, abs(0.03 * row.q_mvar)),
                           bus)

    # 2. Do state estimation
    if not estimate(net, init="flat"):
        raise AssertionError("Estimation failed!")

    v_result = net.res_bus_est.vm_pu.values
    delta_result = net.res_bus_est.va_degree.values

    target_v = net.res_bus.vm_pu.values
    diff_v = target_v - v_result
    target_delta = net.res_bus.va_degree.values
    diff_delta = target_delta - delta_result

    assert (np.nanmax(abs(diff_v)) < 0.0043)
    assert (np.nanmax(abs(diff_delta)) < 0.2)


def test_cigre_network_with_slack_init():
    test_cigre_network(init='slack')


def test_cigre_with_bad_data():
    np.random.seed(123456)
    net = create_cigre_network_mv(with_der=False)
    net.load.q_mvar = net.load["p_mw"].apply(lambda p: p * np.tan(np.arccos(np.random.choice([0.95, 0.9, 0.97]))))
    runpp(net)

    for bus, row in net.res_bus.iterrows():
        if bus == 2:
            continue
        if bus != 6:
            create_measurement(net, "v", "bus", row.vm_pu * r(0.01), 0.01, bus)  # skip our bad data measurement
        create_measurement(net, "p", "bus", row.p_mw * r(), max(0.001, abs(0.03 * row.p_mw)), bus)
        create_measurement(net, "q", "bus", row.q_mvar * r(), max(0.001, abs(0.03 * row.q_mvar)), bus)

    # 2. Do state estimation
    success_SE = estimate(net, init='slack')
    v_est_SE = net.res_bus_est.vm_pu.values
    delta_SE = net.res_bus_est.va_degree.values

    # 3. Create false measurement (very close to useful values)
    create_measurement(net, "v", "bus", 0.85, 0.01, element=6)

    # 4. Do chi2-test
    bad_data_detected = chi2_analysis(net, init='slack')

    # 5. Perform rn_max_test
    success_rn_max = remove_bad_data(net, init='slack')
    v_est_rn_max = net.res_bus_est.vm_pu.values
    delta_est_rn_max = net.res_bus_est.va_degree.values

    diff_v = v_est_SE - v_est_rn_max
    diff_delta = delta_SE - delta_est_rn_max

    assert success_SE
    assert bad_data_detected
    assert success_rn_max
    assert (np.nanmax(abs(diff_v)) < 1e-8)
    assert (np.nanmax(abs(diff_delta)) < 1e-8)


def test_init_slack_with_multiple_transformers():
    np.random.seed(123)
    net = create_empty_network()
    create_bus(net, 220, index=0)
    create_bus(net, 110, index=1)
    create_bus(net, 110, index=2)
    create_bus(net, 110, index=3)
    create_bus(net, 10, index=4)
    create_bus(net, 10, index=5)
    create_bus(net, 10, index=6)
    create_bus(net, 10, index=7, in_service=False)
    create_std_type(net, {"sn_mva": 63, "vn_hv_kv": 110, "vn_lv_kv": 10, "vk_percent": 10.04,
                          "vkr_percent": 0.31, "pfe_kw": 31.51, "i0_percent": 0.078, "shift_degree": 150,
                          "tap_side": "hv", "tap_neutral": 0, "tap_min": -9, "tap_max": 9, "tap_step_degree": 0,
                          "tap_step_percent": 1.5, "tap_changer_type": "Ratio"},
                    "63 MVA 110/10 kV v1.4.3 and older", element="trafo")

    create_transformer(net, 3, 7, std_type="63 MVA 110/10 kV v1.4.3 and older", in_service=False)
    create_transformer(net, 3, 4, std_type="63 MVA 110/10 kV v1.4.3 and older")
    create_transformer(net, 0, 1, std_type="100 MVA 220/110 kV")
    create_line(net, 1, 2, 2.0, std_type="N2XS(FL)2Y 1x120 RM/35 64/110 kV")
    create_line(net, 1, 3, 2.0, std_type="N2XS(FL)2Y 1x120 RM/35 64/110 kV")
    create_line(net, 4, 5, 2.0, std_type="NA2XS2Y 1x95 RM/25 12/20 kV")
    create_line(net, 5, 6, 2.0, std_type="NA2XS2Y 1x95 RM/25 12/20 kV")
    create_load(net, 2, p_mw=5, q_mvar=3.3)
    create_load(net, 5, p_mw=0.9, q_mvar=0.5)
    create_load(net, bus=6, p_mw=0.7, q_mvar=0.3)
    create_ext_grid(net, bus=0, vm_pu=1.04, va_degree=10., name="Slack 220 kV")
    runpp(net, calculate_voltage_angles=True)
    for bus, row in net.res_bus[net.bus.in_service].iterrows():
        create_measurement(net, "v", "bus", row.vm_pu * r(0.01), 0.01, bus)
        if not np.isclose(row.p_mw, 0.0):
            continue
        create_measurement(net, "p", "bus", row.p_mw * r(), max(0.001, abs(0.03 * row.p_mw)),
                           bus)
        create_measurement(net, "q", "bus", row.q_mvar * r(), max(.0001, abs(0.03 * row.q_mvar)),
                           bus)
    create_measurement(net, "p", "line", net.res_line.p_from_mw[0], .01, side="from", element=0)
    create_measurement(net, "q", "line", net.res_line.q_from_mvar[0], 0.01, side="from", element=0)
    create_measurement(net, "p", "line", net.res_line.p_from_mw[2], .01, side="from", element=2)
    create_measurement(net, "q", "line", net.res_line.q_from_mvar[2], .01, side="from", element=2)
    create_measurement(net, "p", "line", net.res_line.p_from_mw[3], .01, side="from", element=3)
    create_measurement(net, "q", "line", net.res_line.q_from_mvar[3], 0.01, side="from", element=3)
    success = estimate(net, init='slack', tolerance=1e-9)

    # pretty high error for vm_pu (half percent!)
    assert success
    assert (np.nanmax(np.abs(net.res_bus.vm_pu.values - net.res_bus_est.vm_pu.values)) < 0.006)
    assert (np.nanmax(np.abs(net.res_bus.va_degree.values - net.res_bus_est.va_degree.values)) < 0.006)


def test_check_existing_measurements():
    np.random.seed(2017)
    net = create_empty_network()
    create_bus(net, 10.)
    create_bus(net, 10.)
    create_line(net, 0, 1, 0.5, std_type="149-AL1/24-ST1A 10.0")
    m1 = create_measurement(net, "v", "bus", 1.006, .004, 0)
    m2 = create_measurement(net, "v", "bus", 1.006, .004, 0)

    # assert m1 == m2
    assert len(net.measurement) == 2
    m3 = create_measurement(net, "v", "bus", 1.006, .004, 0, check_existing=False)
    # assert m3 != m2
    assert len(net.measurement) == 3

    m4 = create_measurement(net, "p", "line", -0.0011, 0.01, side="from", element=0,
                            check_existing=True)
    m5 = create_measurement(net, "p", "line", -0.0011, 0.01, side="from", element=0,
                            check_existing=True)
    assert m4 == m5

    m6 = create_measurement(net, "p", "line", -0.0011, 0.01, side="from", element=0,
                            check_existing=False)
    assert m5 != m6
    assert len(net.measurement) == 5


def load_3bus_network():
    return from_json(os.path.join(pp_dir, "test", "estimation", "3bus_wls.json"))


def test_network_with_trafo3w_pq():
    net = create_empty_network()

    bus_slack = create_bus(net, vn_kv=110)
    create_ext_grid(net, bus=bus_slack)

    bus_20_1 = create_bus(net, vn_kv=20, name="b")
    create_sgen(net, bus=bus_20_1, p_mw=0.03, q_mvar=0.02)

    bus_10_1 = create_bus(net, vn_kv=10)
    create_sgen(net, bus=bus_10_1, p_mw=0.02, q_mvar=0.02)

    bus_10_2 = create_bus(net, vn_kv=10)
    create_load(net, bus=bus_10_2, p_mw=0.06, q_mvar=0.01)
    create_line(net, from_bus=bus_10_1, to_bus=bus_10_2, std_type="149-AL1/24-ST1A 10.0", length_km=2)

    create_transformer3w(net, bus_slack, bus_20_1, bus_10_1, std_type="63/25/38 MVA 110/20/10 kV")

    runpp(net)

    create_measurement(net, "p", "line", net.res_line.p_from_mw[0], 0.001, 0, 'from')
    create_measurement(net, "q", "line", net.res_line.q_from_mvar[0], 0.001, 0, 'from')
    create_measurement(net, "p", "line", net.res_line.p_to_mw[0], 0.001, 0, 'to')
    create_measurement(net, "q", "line", net.res_line.q_to_mvar[0], 0.001, 0, 'to')

    create_measurement(net, "p", "trafo3w", net.res_trafo3w.p_hv_mw[0], 0.001, 0, 'hv')
    create_measurement(net, "q", "trafo3w", net.res_trafo3w.q_hv_mvar[0], 0.001, 0, 'hv')
    create_measurement(net, "p", "trafo3w", net.res_trafo3w.p_mv_mw[0], 0.002, 0, 'mv')
    create_measurement(net, "q", "trafo3w", net.res_trafo3w.q_mv_mvar[0], 0.002, 0, 'mv')
    create_measurement(net, "p", "trafo3w", net.res_trafo3w.p_lv_mw[0], 0.001, 0, 'lv')
    create_measurement(net, "q", "trafo3w", net.res_trafo3w.q_lv_mvar[0], 0.001, 0, 'lv')

    create_measurement(net, "v", "bus", net.res_bus.vm_pu[0], 0.01, 0)
    create_measurement(net, "v", "bus", net.res_bus.vm_pu[1], 0.01, 1)

    if not estimate(net):
        raise AssertionError("Estimation failed!")

    if not (np.nanmax(np.abs(net.res_bus.vm_pu.values - net.res_bus_est.vm_pu.values)) < 0.006) or \
            not (np.nanmax(np.abs(net.res_bus.va_degree.values - net.res_bus_est.va_degree.values)) < 0.006):
        raise AssertionError("Estimation failed")

    # Try estimate with results initialization
    if not estimate(net, init="results"):
        raise AssertionError("Estimation failed!")


def test_network_with_trafo3w_with_disabled_branch():
    net = create_empty_network()

    bus_slack = create_bus(net, vn_kv=110)
    create_ext_grid(net, bus=bus_slack)

    bus_20_1 = create_bus(net, vn_kv=20, name="b")
    create_sgen(net, bus=bus_20_1, p_mw=0.03, q_mvar=0.02)

    bus_10_1 = create_bus(net, vn_kv=10)
    create_sgen(net, bus=bus_10_1, p_mw=0.02, q_mvar=0.02)

    bus_10_2 = create_bus(net, vn_kv=10)
    create_load(net, bus=bus_10_2, p_mw=0.06, q_mvar=0.01)
    create_line(net, from_bus=bus_10_1, to_bus=bus_10_2, std_type="149-AL1/24-ST1A 10.0", length_km=2)
    disabled_line = create_line(net, from_bus=bus_10_1, to_bus=bus_10_2, std_type="149-AL1/24-ST1A 10.0", length_km=2)
    net.line.at[disabled_line, 'in_service'] = False

    create_transformer3w(net, bus_slack, bus_20_1, bus_10_1, std_type="63/25/38 MVA 110/20/10 kV")

    runpp(net)

    create_measurement(net, "p", "line", net.res_line.p_from_mw[0], 0.001, 0, 'from')
    create_measurement(net, "q", "line", net.res_line.q_from_mvar[0], 0.001, 0, 'from')
    create_measurement(net, "p", "line", net.res_line.p_to_mw[0], 0.001, 0, 'to')
    create_measurement(net, "q", "line", net.res_line.q_to_mvar[0], 0.001, 0, 'to')
    create_measurement(net, "p", "line", net.res_line.p_to_mw[1], 0.001, 1, 'to')
    create_measurement(net, "q", "line", net.res_line.q_to_mvar[1], 0.001, 1, 'to')

    create_measurement(net, "p", "trafo3w", net.res_trafo3w.p_hv_mw[0], 0.001, 0, 'hv')
    create_measurement(net, "q", "trafo3w", net.res_trafo3w.q_hv_mvar[0], 0.001, 0, 'hv')
    create_measurement(net, "p", "trafo3w", net.res_trafo3w.p_mv_mw[0], 0.002, 0, 'mv')
    create_measurement(net, "q", "trafo3w", net.res_trafo3w.q_mv_mvar[0], 0.002, 0, 'mv')
    create_measurement(net, "p", "trafo3w", net.res_trafo3w.p_lv_mw[0], 0.001, 0, 'lv')
    create_measurement(net, "q", "trafo3w", net.res_trafo3w.q_lv_mvar[0], 0.001, 0, 'lv')

    create_measurement(net, "v", "bus", net.res_bus.vm_pu[0], 0.01, 0)
    create_measurement(net, "v", "bus", net.res_bus.vm_pu[1], 0.01, 1)

    success = estimate(net)
    assert success
    assert (np.nanmax(np.abs(net.res_bus.vm_pu.values - net.res_bus_est.vm_pu.values)) < 0.006)
    assert (np.nanmax(np.abs(net.res_bus.va_degree.values - net.res_bus_est.va_degree.values)) < 0.006)


def create_net_with_bb_switch():
    net = create_empty_network()
    bus1 = create_bus(net, name="bus1", vn_kv=10.)
    bus2 = create_bus(net, name="bus2", vn_kv=10.)
    bus3 = create_bus(net, name="bus3", vn_kv=10.)
    bus4 = create_bus(net, name="bus4", vn_kv=10.)
    bus5 = create_bus(net, name="bus5", vn_kv=110.)

    create_line_from_parameters(net, bus1, bus2, 10, r_ohm_per_km=.59, x_ohm_per_km=.35, c_nf_per_km=10.1,
                                max_i_ka=1)
    create_transformer(net, bus5, bus1, std_type="40 MVA 110/10 kV")
    create_ext_grid(net, bus=bus5, vm_pu=1.0)
    create_load(net, bus1, p_mw=.350, q_mvar=.100)
    create_load(net, bus2, p_mw=.450, q_mvar=.100)
    create_load(net, bus3, p_mw=.250, q_mvar=.100)
    create_load(net, bus4, p_mw=.150, q_mvar=.100)

    # Created bb switch
    create_switch(net, bus2, element=bus3, et='b')
    create_switch(net, bus1, element=bus4, et='b')
    runpp(net, calculate_voltage_angles=True)

    create_measurement(net, "v", "bus", r2(net.res_bus.vm_pu.iloc[bus1], .002), .002, element=bus1)
    create_measurement(net, "v", "bus", r2(net.res_bus.vm_pu.iloc[bus3], .002), .002, element=bus3)
    create_measurement(net, "v", "bus", r2(net.res_bus.vm_pu.iloc[bus5], .002), .002, element=bus5)

    create_measurement(net, "p", "bus", r2(net.res_bus.p_mw.iloc[bus5], .002), .002, element=bus5)
    create_measurement(net, "q", "bus", r2(net.res_bus.q_mvar.iloc[bus5], .002), .002, element=bus5)

    # If measurement on the bus with bb-switch activated, it will incluence the results of the merged bus
    create_measurement(net, "p", "bus", r2(net.res_bus.p_mw.iloc[bus4], .002), .002, element=bus4)
    create_measurement(net, "q", "bus", r2(net.res_bus.q_mvar.iloc[bus4], .002), .002, element=bus4)
    create_measurement(net, "p", "bus", r2(net.res_bus.p_mw.iloc[bus3], .001), .001, element=bus3)
    create_measurement(net, "q", "bus", r2(net.res_bus.q_mvar.iloc[bus3], .001), .001, element=bus3)
    create_measurement(net, "p", "bus", r2(net.res_bus.p_mw.iloc[bus2], .001), .001, element=bus2)
    create_measurement(net, "q", "bus", r2(net.res_bus.q_mvar.iloc[bus2], .001), .001, element=bus2)
    create_measurement(net, "p", "bus", r2(net.res_bus.p_mw.iloc[bus1], .001), .001, element=bus1)
    create_measurement(net, "q", "bus", r2(net.res_bus.q_mvar.iloc[bus1], .001), .001, element=bus1)

    create_measurement(net, "p", "line", r2(net.res_line.p_from_mw.iloc[0], .002), .002, 0, side='from')
    create_measurement(net, "q", "line", r2(net.res_line.q_from_mvar.iloc[0], .002), .002, 0, side='from')

    create_measurement(net, "p", "trafo", r2(net.res_trafo.p_hv_mw.iloc[0], .001), .01,
                       side="hv", element=0)
    create_measurement(net, "q", "trafo", r2(net.res_trafo.q_hv_mvar.iloc[0], .001), .01,
                       side="hv", element=0)
    return net


def test_net_with_bb_switch_no_fusing():
    net = create_net_with_bb_switch()
    success_none = estimate(net, tolerance=1e-5, fuse_buses_with_bb_switch=None)
    runpp(net, calculate_voltage_angles=True)
    assert success_none
    assert np.allclose(net.res_bus.va_degree.values, net.res_bus_est.va_degree.values, 1e-2)
    assert np.allclose(net.res_bus.vm_pu.values, net.res_bus_est.vm_pu.values, 1e-2)
    # asserting with more tolerance since the added impedance will cause some inaccuracy
    assert np.allclose(net.res_bus.p_mw.values, net.res_bus_est.p_mw.values, 1e-1)
    assert np.allclose(net.res_bus.q_mvar.values, net.res_bus_est.q_mvar.values, 1e-1)


def test_net_with_bb_switch_fuse_one():
    net = create_net_with_bb_switch()
    success = estimate(net, tolerance=1e-5, fuse_buses_with_bb_switch=[1])
    runpp(net, calculate_voltage_angles=True)
    assert success
    assert np.allclose(net.res_bus.va_degree.values, net.res_bus_est.va_degree.values, 1e-2)
    assert np.allclose(net.res_bus.vm_pu.values, net.res_bus_est.vm_pu.values, 1e-2)
    # asserting with more tolerance since the added impedance will cause some inaccuracy
    assert np.allclose(net.res_bus.p_mw.values[[0, 3, 4]], net.res_bus_est.p_mw.values[[0, 3, 4]], 1e-1)
    assert np.allclose(net.res_bus.q_mvar.values[[0, 3, 4]], net.res_bus_est.q_mvar.values[[0, 3, 4]], 1e-1)


@pytest.mark.xfail
def test_net_with_bb_switch_fuse_one_identify_pq():
    net = create_net_with_bb_switch()
    estimate(net, tolerance=1e-5, fuse_buses_with_bb_switch=[1])
    # asserting with more tolerance since the added impedance will cause some inaccuracy
    assert np.allclose(net.res_bus.p_mw.values, net.res_bus_est.p_mw.values, 1e-1)
    assert np.allclose(net.res_bus.q_mvar.values, net.res_bus_est.q_mvar.values, 1e-1)


def test_net_with_bb_switch_fusing():
    net = create_net_with_bb_switch()
    estimate(net, tolerance=1e-5, fuse_buses_with_bb_switch='all')

    assert np.allclose(net.res_bus.va_degree.values, net.res_bus_est.va_degree.values, 5e-2)
    assert np.allclose(net.res_bus.vm_pu.values, net.res_bus_est.vm_pu.values, 5e-2)


def test_net_with_zero_injection():
    # @author: AndersLi
    net = create_empty_network()
    b1 = create_bus(net, name="Bus 1", vn_kv=220, index=1)
    b2 = create_bus(net, name="Bus 2", vn_kv=220, index=2)
    b3 = create_bus(net, name="Bus 3", vn_kv=220, index=3)
    b4 = create_bus(net, name="Bus 4", vn_kv=220, index=4)

    create_ext_grid(net, b1)  # set the slack bus to bus 1
    factor = 48.4 * 2 * np.pi * 50 * 1e-9  # capacity factor

    create_line_from_parameters(net, 1, 2, 1, r_ohm_per_km=.0221 * 48.4,
                                x_ohm_per_km=.1603 * 48.4, c_nf_per_km=0.00274 / factor, max_i_ka=1)
    create_line_from_parameters(net, 2, 3, 1, r_ohm_per_km=.0428 * 48.4,
                                x_ohm_per_km=.242 * 48.4, c_nf_per_km=0.00384 / factor, max_i_ka=1)
    l3 = create_line_from_parameters(net, 2, 4, 1, r_ohm_per_km=.002 * 48.4,
                                     x_ohm_per_km=.0111 * 48.4, c_nf_per_km=0.00018 / factor, max_i_ka=1)

    create_measurement(net, "v", "bus", 1.063548, .001, b1)  # V at bus 1
    create_measurement(net, "v", "bus", 1.068342, .001, b3)  # V at bus 3
    create_measurement(net, "v", "bus", 1.069861, .001, b4)  # V at bus 4
    create_measurement(net, "p", "bus", 40.0, 1, b1)  # P at bus 1
    create_measurement(net, "q", "bus", 9.2, 1, b1)  # Q at bus 1
    # create_measurement(net, "p", "bus", 0, 0.01, b2)              # P at bus 2 - not required anymore
    # create_measurement(net, "q", "bus", 0, 0.01, b2)              # Q at bus 2 - not required anymore
    create_measurement(net, "p", "bus", -10.0, 1, b3)  # P at bus 3
    create_measurement(net, "q", "bus", -1.0, 1, b3)  # Q at bus 3
    create_measurement(net, "p", "bus", -30.0, 1, b4)  # P at bus 4
    create_measurement(net, "q", "bus", 0.100, 1, b4)  # Q at bus 4
    create_measurement(net, "p", "line", 30.100, 1, l3, side="to")  # Pline (bus 2 -> bus 4) at bus 4
    create_measurement(net, "q", "line", -0.099, 1, l3, side="to")  # Qline (bus 2 -> bus 4) at bus 4

    estimate(net, tolerance=1e-10, zero_injection='no_inj_bus', algorithm='wls_with_zero_constraint')
    assert np.abs(net.res_bus_est.at[b2, 'p_mw']) < 1e-8
    assert np.abs(net.res_bus_est.at[b2, 'q_mvar']) < 1e-8

    net_given_bus = deepcopy(net)
    success = estimate(net, tolerance=1e-6, zero_injection="no_inj_bus")
    assert success
    assert np.allclose(net.res_bus_est.va_degree.values, net_given_bus.res_bus_est.va_degree.values, 1e-3)
    assert np.allclose(net.res_bus_est.vm_pu.values, net_given_bus.res_bus_est.vm_pu.values, 1e-3)


def test_zero_injection_aux_bus():
    net = create_empty_network()
    bus1 = create_bus(net, name="bus1", vn_kv=10.)
    bus2 = create_bus(net, name="bus2", vn_kv=10.)
    bus3 = create_bus(net, name="bus3", vn_kv=10.)
    bus4 = create_bus(net, name="bus4", vn_kv=110.)

    create_line_from_parameters(net, bus1, bus2, 10, r_ohm_per_km=.59, x_ohm_per_km=.35, c_nf_per_km=10.1,
                                max_i_ka=1)
    create_line_from_parameters(net, bus2, bus3, 10, r_ohm_per_km=.59, x_ohm_per_km=.35, c_nf_per_km=10.1,
                                max_i_ka=1)
    create_transformer(net, bus4, bus1, std_type="40 MVA 110/10 kV")
    create_ext_grid(net, bus=bus4, vm_pu=1.0)
    create_load(net, bus1, p_mw=.350, q_mvar=.100)
    create_load(net, bus2, p_mw=.450, q_mvar=.100)
    create_load(net, bus3, p_mw=.250, q_mvar=.100)

    net.bus.at[bus3, 'in_service'] = False

    # Created bb switch
    runpp(net, calculate_voltage_angles=True)

    create_measurement(net, "v", "bus", r2(net.res_bus.vm_pu.iloc[bus1], .002), .002, element=bus1)
    create_measurement(net, "v", "bus", r2(net.res_bus.vm_pu.iloc[bus4], .002), .002, element=bus4)

    create_measurement(net, "p", "bus", r2(net.res_bus.p_mw.iloc[bus4], .002), .002, element=bus4)
    create_measurement(net, "q", "bus", r2(net.res_bus.q_mvar.iloc[bus4], .002), .002, element=bus4)

    # If measurement on the bus with bb-switch activated, it will incluence the results of the merged bus
    create_measurement(net, "p", "bus", r2(net.res_bus.p_mw.iloc[bus2], .001), .001, element=bus2)
    create_measurement(net, "q", "bus", r2(net.res_bus.q_mvar.iloc[bus2], .001), .001, element=bus2)
    create_measurement(net, "p", "bus", r2(net.res_bus.p_mw.iloc[bus1], .001), .001, element=bus1)
    create_measurement(net, "q", "bus", r2(net.res_bus.q_mvar.iloc[bus1], .001), .001, element=bus1)

    create_measurement(net, "p", "line", r2(net.res_line.p_from_mw.iloc[0], .002), .002, 0, side='from')
    create_measurement(net, "q", "line", r2(net.res_line.q_from_mvar.iloc[0], .002), .002, 0, side='from')

    create_measurement(net, "p", "trafo", r2(net.res_trafo.p_hv_mw.iloc[0], .001), .01,
                       side="hv", element=0)
    create_measurement(net, "q", "trafo", r2(net.res_trafo.q_hv_mvar.iloc[0], .001), .01,
                       side="hv", element=0)

    net_noinj_bus = deepcopy(net)
    net_aux = deepcopy(net)

    success_none = estimate(net, tolerance=1e-5, zero_injection=None)

    # In this case zero_injection in mode "aux_bus" and "no_inj_bus" should be exact the same
    success_aux = estimate(net_aux, tolerance=1e-5, zero_injection='aux_bus')
    success_noinj_bus = estimate(net_noinj_bus, tolerance=1e-5, zero_injection='no_inj_bus')
    assert success_none and success_aux and success_noinj_bus
    assert np.allclose(net_noinj_bus.res_bus_est.va_degree.values, net_aux.res_bus_est.va_degree.values, 1e-4,
                       equal_nan=True)
    assert np.allclose(net_noinj_bus.res_bus_est.vm_pu.values, net_aux.res_bus_est.vm_pu.values, 1e-4, equal_nan=True)

    # in case zero injection was set to none, the results should be different
    assert ~np.allclose(net.res_bus_est.vm_pu.values, net_aux.res_bus_est.vm_pu.values, 1e-2, equal_nan=True)


@pytest.mark.xfail
def test_net_unobserved_island():
    net = create_empty_network()
    bus1 = create_bus(net, name="bus1", vn_kv=10.)
    bus2 = create_bus(net, name="bus2", vn_kv=10.)
    bus3 = create_bus(net, name="bus3", vn_kv=10.)
    bus4 = create_bus(net, name="bus4", vn_kv=110.)

    create_line_from_parameters(net, bus1, bus2, 10, r_ohm_per_km=.59, x_ohm_per_km=.35, c_nf_per_km=10.1,
                                max_i_ka=1)
    create_line_from_parameters(net, bus2, bus3, 10, r_ohm_per_km=.59, x_ohm_per_km=.35, c_nf_per_km=10.1,
                                max_i_ka=1)
    create_transformer(net, bus4, bus1, std_type="40 MVA 110/10 kV")
    create_ext_grid(net, bus=bus4, vm_pu=1.0)
    create_load(net, bus1, p_mw=.350, q_mvar=.100)
    create_load(net, bus2, p_mw=.450, q_mvar=.100)
    create_load(net, bus3, p_mw=.250, q_mvar=.100)

    # Created bb switch
    runpp(net, calculate_voltage_angles=True)

    create_measurement(net, "v", "bus", r2(net.res_bus.vm_pu.iloc[bus1], .002), .002, element=bus1)
    create_measurement(net, "v", "bus", r2(net.res_bus.vm_pu.iloc[bus4], .002), .002, element=bus4)

    create_measurement(net, "p", "bus", r2(net.res_bus.p_mw.iloc[bus4], .002), .002, element=bus4)
    create_measurement(net, "q", "bus", r2(net.res_bus.q_mvar.iloc[bus4], .002), .002, element=bus4)

    # IF pq of bus2 is not available makes bus3 an unobserved island
    #    create_measurement(net, "p", "bus", -r2(net.res_bus.p_mw.iloc[bus2], .001), .001, element=bus2)
    #    create_measurement(net, "q", "bus", -r2(net.res_bus.q_mvar.iloc[bus2], .001), .001, element=bus2)
    create_measurement(net, "p", "bus", r2(net.res_bus.p_mw.iloc[bus1], .001), .001, element=bus1)
    create_measurement(net, "q", "bus", r2(net.res_bus.q_mvar.iloc[bus1], .001), .001, element=bus1)

    create_measurement(net, "p", "line", r2(net.res_line.p_from_mw.iloc[0], .002), .002, 0, side='from')
    create_measurement(net, "q", "line", r2(net.res_line.q_from_mvar.iloc[0], .002), .002, 0, side='from')

    create_measurement(net, "p", "trafo", r2(net.res_trafo.p_hv_mw.iloc[0], .001), .01,
                       side="hv", element=0)
    create_measurement(net, "q", "trafo", r2(net.res_trafo.q_hv_mvar.iloc[0], .001), .01,
                       side="hv", element=0)

    if not estimate(net, tolerance=1e-6, zero_injection=None):
        raise AssertionError("Estimation failed!")


def test_net_oos_line():
    net = case9()
    net.line.in_service.iat[4] = False
    runpp(net)

    for line_ix in net.line.index:
        create_measurement(net, "p", "line", net.res_line.at[line_ix, "p_from_mw"],
                           0.01, element=line_ix, side="from")
        create_measurement(net, "q", "line", net.res_line.at[line_ix, "q_from_mvar"],
                           0.01, element=line_ix, side="from")

    for bus_ix in net.bus.index:
        create_measurement(net, "v", "bus", net.res_bus.at[bus_ix, "vm_pu"],
                           0.01, element=bus_ix)

    if not estimate(net, tolerance=1e-6, zero_injection=None):
        raise AssertionError("Estimation failed!")


def r(v=0.03):
    return np.random.normal(1.0, v)


def r2(base, v):
    return np.random.normal(base, v)


def _compare_pf_and_se_results(net):
    runpp(net, calculate_voltage_angles=True, trafo_model="t")
    assert (np.allclose(net.res_bus_est.p_mw.values, net.res_bus.p_mw.values, 1e-6))
    assert (np.allclose(net.res_bus_est.q_mvar.values, net.res_bus.q_mvar.values, 1e-6))
    assert (np.allclose(net.res_line_est.p_from_mw.values, net.res_line.p_from_mw.values, 1e-6))
    assert (np.allclose(net.res_line_est.q_from_mvar.values, net.res_line.q_from_mvar.values, 1e-6))
    assert (np.allclose(net.res_line_est.p_to_mw.values, net.res_line.p_to_mw.values, 1e-6))
    assert (np.allclose(net.res_line_est.q_to_mvar.values, net.res_line.q_to_mvar.values, 1e-6))
    if "trafo" in net and net.trafo.shape[0] > 0:
        assert (np.allclose(net.res_trafo_est.p_lv_mw.values, net.res_trafo.p_lv_mw.values, 1e-6))
        assert (np.allclose(net.res_trafo_est.q_lv_mvar.values, net.res_trafo.q_lv_mvar.values, 1e-6))
        assert (np.allclose(net.res_trafo_est.p_hv_mw.values, net.res_trafo.p_hv_mw.values, 1e-6))
        assert (np.allclose(net.res_trafo_est.q_hv_mvar.values, net.res_trafo.q_hv_mvar.values, 1e-6))


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])

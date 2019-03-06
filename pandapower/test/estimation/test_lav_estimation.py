# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os

import numpy as np
import pytest

import pandapower as pp
import pandapower.networks as nw
from pandapower.estimation import chi2_analysis, remove_bad_data, estimate
from copy import deepcopy


def test_2bus():
    # 1. Create network
    net = pp.create_empty_network()
    pp.create_bus(net, name="bus1", vn_kv=1.)
    pp.create_bus(net, name="bus2", vn_kv=1.)
    pp.create_load(net, 1, p_mw=0.0111, q_mvar=0.06)
    pp.create_ext_grid(net, 0, vm_pu=1.038)
    pp.create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=1,x_ohm_per_km=0.5, c_nf_per_km=0,
                                   max_i_ka=1)

    pp.create_measurement(net, "p", "line", -0.0111, 0.003, 0, 1)  # p12
    pp.create_measurement(net, "q", "line", -0.06, 0.003, 0, 1)    # q12

    pp.create_measurement(net, "v", "bus", 1.038, 0.0001, 0)  # u1
    pp.create_measurement(net, "v", "bus", 1.02, 0.1, 1)   # u2

    # 2. Do state estimation
#    success = estimate(net, init='flat', algorithm="lav")
    success = estimate(net, init='flat', algorithm="wls")
    pp.runpp(net)

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

    pp.create_measurement(net, "p", "line", -0.0011, 0.01, 0, 0)  # p12
    pp.create_measurement(net, "q", "line", 0.024, 0.01, 0, 0)    # q12

    pp.create_measurement(net, "p", "bus", 0.018, 0.01, 2)  # p3
    pp.create_measurement(net, "q", "bus", -0.1, 0.01, 2)   # q3

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

    # Backwards check. Use state estimation results for power flow and check for equality
    net.ext_grid.vm_pu = net.res_bus_est.vm_pu.iloc[0]
    pp.create_load(net, 0, net.res_bus_est.p_mw.iloc[0], net.res_bus_est.q_mvar.iloc[0])
    pp.create_load(net, 1, net.res_bus_est.p_mw.iloc[1], net.res_bus_est.q_mvar.iloc[1])
    pp.create_load(net, 2, net.res_bus_est.p_mw.iloc[2], net.res_bus_est.q_mvar.iloc[2])
    _compare_pf_and_se_results(net)


def r(v=0.03):
    return np.random.normal(1.0, v)


def r2(base, v):
    return np.random.normal(base, v)


def _compare_pf_and_se_results(net):
    pp.runpp(net, calculate_voltage_angles=True)
    assert (np.allclose(net.res_bus_est.p_mw.values, net.res_bus.p_mw.values, 1e-6))
    assert (np.allclose(net.res_bus_est.q_mvar.values, net.res_bus.q_mvar.values, 1e-6))
    assert (np.allclose(net.res_line_est.p_from_mw.values, net.res_line.p_from_mw.values, 1e-6))
    assert (np.allclose(net.res_line_est.q_from_mvar.values, net.res_line.q_from_mvar.values, 1e-6))
    assert (np.allclose(net.res_line_est.p_to_mw.values, net.res_line.p_to_mw.values, 1e-6))
    assert (np.allclose(net.res_line_est.q_to_mvar.values, net.res_line.q_to_mvar.values, 1e-6))
    assert (np.allclose(net.res_trafo_est.p_lv_mw.values, net.res_trafo.p_lv_mw.values, 1e-6))
    assert (np.allclose(net.res_trafo_est.q_lv_mvar.values, net.res_trafo.q_lv_mvar.values, 1e-6))
    assert (np.allclose(net.res_trafo_est.p_hv_mw.values, net.res_trafo.p_hv_mw.values, 1e-6))
    assert (np.allclose(net.res_trafo_est.q_hv_mvar.values, net.res_trafo.q_hv_mvar.values, 1e-6))


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])

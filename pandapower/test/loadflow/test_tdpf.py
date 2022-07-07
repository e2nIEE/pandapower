# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import os
import pandas as pd
import pytest
from numpy import in1d, isnan, isclose

import pandapower as pp
import pandapower.control
from pandapower.test.consistency_checks import runpp_with_consistency_checks
from pandapower.test.loadflow.result_test_network_generator import add_test_enforce_qlims, \
    add_test_gen
from pandapower.test.toolbox import assert_res_equal

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks
from pandapower.tdpf.create_jacobian_tdpf import *

import matplotlib.pyplot as plt
import pandapower.plotting
from pandapower.tdpf.test_system import *
from pandapower.tdpf.create_jacobian_tdpf import *

from pandapower.pypower.idx_brch import BR_R, F_BUS, BR_R_REF_OHM_PER_KM


def prepare_case_30():
    net = pp.networks.case30()

    for i in net.line.index.values:
        net.line.at[i, 'name'] = f"{net.line.at[i, 'from_bus'] + 1} - {net.line.at[i, 'to_bus'] + 1}"

    # replicating the calculation from the paper Frank et.al.:
    net.line["alpha"] = 1 / (25 + 228.1)
    net.line["r_theta"] = calc_r_theta_from_t_rise(net, 25)
    net.line["tdpf"] = net.line.r_ohm_per_km != 0

    net.line["temperature_degree_celsius"] = 25
    net.line["reference_temperature_degree_celsius"] = 25
    net.line["ambient_temperature_degree_celsius"] = 25
    net.line["outer_diameter_m"] = 30.6e-3
    net.line["mc_joule_per_m_k"] = 1490
    net.line["wind_speed_m_per_s"] = 0.6
    net.line["wind_angle_degree"] = 45
    net.line["solar_radiation_w_per_sq_m"] = 900
    net.line["gamma"] = 0.5
    net.line["epsilon"] = 0.5
    return net


def simple_test_grid(load_scaling=1., sgen_scaling=1., with_gen=False, distributed_slack=False):
    s_base = 100

    net = pp.create_empty_network(sn_mva=s_base)
    std_type = "490-AL1/64-ST1A 110.0"
    # r = 0.1188
    # std_type = "490-AL1/64-ST1A 220.0"
    r = 0.059
    v_base = 132
    z_base = v_base ** 2 / s_base

    pp.create_buses(net, 5, v_base, geodata=((0, 1), (-1, 0.5), (0, 0), (1, 0.5), (0, 0.5)))

    pp.create_line(net, 0, 1, 0.84e-2 * z_base / r, std_type, name="1-2")
    pp.create_line(net, 0, 3, 0.84e-2 * z_base / r, std_type, name="1-4")
    pp.create_line(net, 1, 2, 0.67e-2 * z_base / r, std_type, name="2-3")
    pp.create_line(net, 1, 4, 0.42e-2 * z_base / r, std_type, name="2-5")
    pp.create_line(net, 2, 3, 0.67e-2 * z_base / r, std_type, name="3-4")
    pp.create_line(net, 3, 4, 0.42e-2 * z_base / r, std_type, name="4-5")
    net.line.c_nf_per_km = 0

    net.line["temperature_degree_celsius"] = 20
    net.line["ambient_temperature_degree_celsius"] = 35
    net.line["alpha"] = 0.004
    net.line["outer_diameter_m"] = 30.6e-3
    net.line["mc_joule_per_m_k"] = 1490
    net.line["wind_speed_m_per_s"] = 0.6
    net.line["wind_angle_degree"] = 45
    net.line["solar_radiation_w_per_sq_m"] = 900
    net.line["gamma"] = 0.5
    net.line["epsilon"] = 0.5
    net.line["tdpf"] = True

    pp.create_ext_grid(net, 3, 1.05, name="G1")
    pp.create_sgen(net, 0, 200, scaling=sgen_scaling, name="R1")
    pp.create_sgen(net, 1, 250, scaling=sgen_scaling, name="R2")
    if with_gen:
        idx = pp.create_gen(net, 2, 600, 1., scaling=sgen_scaling, name="G3")
        pp.create_gen(net, 4, 300, 1., scaling=sgen_scaling, name="G5")
    else:
        idx = pp.create_sgen(net, 2, 600, scaling=sgen_scaling, name="G3")
        pp.create_sgen(net, 4, 300, scaling=sgen_scaling, name="G5")

    if distributed_slack:
        net["gen" if with_gen else "sgen"].at[idx, 'slack_weight'] = 1
        pp.set_user_pf_options(net, distributed_slack=True)
        net.sn_mva = 1000  # otherwise numerical issues

    pp.create_load(net, 1, 600, 240, scaling=load_scaling)
    pp.create_load(net, 3, 1000, 400, scaling=load_scaling)
    pp.create_load(net, 4, 400, 160, scaling=load_scaling)

    return net


def test_tdpf_frank_current():
    net = prepare_case_30()
    net.line["c_nf_per_km"] = 0  # otherwise i_square_pu will not match net.res_line.i_ka
    pp.runpp(net, tdpf=True, tdpf_update_r_theta=False, max_iteration=30)

    branch = net._ppc["branch"]
    V = net._ppc["internal"]["V"]
    Vm = abs(V)
    Va = np.angle(V)

    # check if p_loss_pu is correct and i_square_pu is correct
    g, b = calc_g_b(branch[:, BR_R].real, branch[:, BR_X].real)
    i_square_pu, p_loss_pu = calc_i_square_p_loss(branch, np.arange(len(branch)), g, b, Vm, Va)
    assert np.allclose(p_loss_pu * net.sn_mva, net.res_line.pl_mw)
    # only passes if c_nf_per_km==0
    assert np.allclose(np.sqrt(i_square_pu), net.res_line.i_from_ka / net.sn_mva * (135 * np.sqrt(3)))


def test_tdpf_frank():
    net = prepare_case_30()
    pp.runpp(net, tdpf=True, tdpf_update_r_theta=False, max_iteration=30)

    v_base_kv = net.bus.loc[net.line.from_bus].vn_kv.values
    z_base_ohm = np.square(v_base_kv) / net.sn_mva
    temperature = net.res_line.temperature_degree_celsius
    net.line["r1"] = net.line.r_ohm_per_km / z_base_ohm
    net.line["r2"] = net.res_line.r_ohm_per_km / z_base_ohm
    r_delta = (net.res_line.r_ohm_per_km - net.line.r_ohm_per_km) / net.line.r_ohm_per_km * 100
    sorted_index = net.line.sort_values(by="r_theta", ascending=False).index

    line_mva = np.max(np.vstack([
        np.sqrt(net.res_line.p_from_mw ** 2 + net.res_line.q_from_mvar ** 2).loc[sorted_index].values,
        np.sqrt(net.res_line.p_to_mw ** 2 + net.res_line.q_to_mvar ** 2).loc[sorted_index].values]), axis=0)
    line_max_mva = net.line.max_i_ka.loc[sorted_index] * 135 * np.sqrt(3)
    line_loading = line_mva / line_max_mva

    ref2 = pd.read_csv(os.path.join(pp.pp_dir, "test", "test_files", "tdpf", "case30_branch_details.csv"))
    ref2.sort_values(by="R_THETA", ascending=False, inplace=True)

    # compare p_loss
    assert np.allclose(ref2.PLoss_TDPF * net.sn_mva, net.res_line.loc[sorted_index, "pl_mw"], rtol=0, atol=1e-6)
    # compare R_Theta
    assert np.allclose(ref2.R_THETA, net.line.loc[sorted_index, "r_theta"], rtol=0, atol=1e-6)
    # compare difference of R
    assert np.allclose(ref2.R_diff.fillna(0) * 100, r_delta.loc[sorted_index].fillna(0), rtol=0, atol=1e-6)
    # compare loading
    assert np.allclose(ref2.pctloading_TDPF, line_loading, rtol=0, atol=0.025)


def test_temperature_r():
    net = simple_test_grid()
    r_ref = net.line.r_ohm_per_km.values / 1e3
    a0, a1, a2, tau = calc_a0_a1_a2_tau(35, 80, 20, r_ref, 30.6e-3, 1490, 0.6, 45, 900, 4e-3, 0.5, 0.5)

    for with_gen in (False, True):
        for distributed_slack in (False, True):
            net = simple_test_grid(load_scaling=0.25, sgen_scaling=0.5, with_gen=with_gen,
                                   distributed_slack=distributed_slack)
            pp.runpp(net, tdpf=True, max_iteration=100)

            T = calc_T_ngoko(np.square(net.res_line.i_ka.values * 1e3), a0, a1, a2, None, None, None)
            assert np.allclose(net.res_line.temperature_degree_celsius, T, rtol=0, atol=1e-6)

            net2 = simple_test_grid(load_scaling=0.25, sgen_scaling=0.5, with_gen=with_gen,
                                    distributed_slack=distributed_slack)
            net2.line["temperature_degree_celsius"] = net.res_line.temperature_degree_celsius
            pp.runpp(net2, consider_line_temperature=True)

            net.res_line.drop(["temperature_degree_celsius"], axis=1, inplace=True)
            assert_res_equal(net, net2)

            # now test transient results -> after 5 min
            pp.runpp(net, tdpf=True, tdpf_delay_s=5 * 60, max_iteration=100)

            net2.line["temperature_degree_celsius"] = net.res_line.temperature_degree_celsius
            pp.runpp(net2, consider_line_temperature=True)

            net.res_line.drop(["temperature_degree_celsius"], axis=1, inplace=True)
            assert_res_equal(net, net2)


def test_ngoko_vs_frank():
    net = simple_test_grid(load_scaling=0.25, sgen_scaling=0.5)
    pp.runpp(net)

    t_amb_pu = 35
    alpha_pu = 4e-3
    r_ref = net.line.r_ohm_per_km.values / 1e3
    a0, a1, a2, tau = calc_a0_a1_a2_tau(t_amb_pu, 80, 20, r_ref, 30.6e-3,
                                        1490, 0.6, 45, 900, alpha_pu, 0.5, 0.5)
    T_ngoko = calc_T_ngoko(np.square(net.res_line.i_ka.values * 1e3), a0, a1, a2, None, None, None)

    branch = net._ppc["branch"]
    tdpf_lines = np.ones(len(branch)).astype(bool)
    r = branch[tdpf_lines, BR_R].real
    # r = r * (1 + alpha_pu * (T - 20))
    x = branch[tdpf_lines, BR_X].real
    g, b = calc_g_b(r, x)
    Vm = abs(net._ppc["internal"]["V"])
    Va = np.angle(net._ppc["internal"]["V"])
    i_square_pu, p_loss_pu = calc_i_square_p_loss(branch, tdpf_lines, g, b, Vm, Va)
    # i_square_pu = np.square(net.res_line.i_ka.values*1e3)
    r_theta = calc_r_theta(t_amb_pu, a0, a1, a2, np.square(net.res_line.i_ka.values * 1e3), p_loss_pu)
    T_frank = calc_T_frank(p_loss_pu, t_amb_pu, r_theta, None, None, None)

    assert np.array_equal(T_ngoko, T_frank)

    i_base_a = net.sn_mva / (132 * np.sqrt(3)) * 1e3
    assert np.allclose(net.res_line.i_ka, 1e-3 * i_base_a * np.sqrt(i_square_pu), rtol=0, atol=1e-6)
    assert np.allclose(net.res_line.pl_mw, p_loss_pu * net.sn_mva, rtol=0, atol=1e-6)


def test_tdpf_delay():
    net = simple_test_grid()
    r_ref = net.line.r_ohm_per_km.values / 1e3
    a0, a1, a2, tau = calc_a0_a1_a2_tau(35, 80, 20, r_ref, 30.6e-3, 1490, 0.6, 45, 900, 4e-3, 0.5, 0.5)

    for with_gen in (False, True):
        for distributed_slack in (False, True):
            net = simple_test_grid(load_scaling=0.25, sgen_scaling=0.5, with_gen=with_gen,
                                   distributed_slack=distributed_slack)
            # no delay
            pp.runpp(net, tdpf=True, max_iteration=100, tdpf_delay_s=0)
            assert np.allclose(net.res_line.temperature_degree_celsius, 20, rtol=0, atol=1e-6)

            # infinite delay (steady state)
            pp.runpp(net, tdpf=True, max_iteration=100, tdpf_delay_s=np.inf)
            temp = net.res_line.temperature_degree_celsius.values.copy()
            pp.runpp(net, tdpf=True, max_iteration=100, tdpf_delay_s=None)
            assert np.allclose(net.res_line.temperature_degree_celsius, temp, rtol=0, atol=1e-6)

            # check tau: time to "charge" to approx. 63.2 %; we cannot match it very accurately though
            pp.runpp(net, tdpf=True, max_iteration=100, tdpf_delay_s=tau)
            assert np.allclose(net.res_line.temperature_degree_celsius, 20 + (temp - 20) * 0.632, rtol=0, atol=0.6)


if __name__ == '__main__':
    pytest.main(['-xs', __file__])

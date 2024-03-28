# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import os

import numpy as np
import pytest

from pandapower.pf.create_jacobian_tdpf import calc_r_theta_from_t_rise, calc_i_square_p_loss, calc_g_b, \
    calc_a0_a1_a2_tau, calc_T_ngoko, calc_r_theta, calc_T_frank
from pandapower.test.helper_functions import assert_res_equal

import pandas as pd
import pandapower as pp
import pandapower.networks

from pandapower.pypower.idx_brch import BR_R, BR_X


# pd.set_option("display.max_columns", 1000)
# pd.set_option("display.width", 1000)

@pytest.fixture(
    params=["94-AL1/15-ST1A 0.4", "70-AL1/11-ST1A 10.0", "94-AL1/15-ST1A 10.0", "122-AL1/20-ST1A 10.0",
            "149-AL1/24-ST1A 10.0", "70-AL1/11-ST1A 20.0", "94-AL1/15-ST1A 20.0", "122-AL1/20-ST1A 20.0",
            "149-AL1/24-ST1A 20.0", "184-AL1/30-ST1A 20.0", "243-AL1/39-ST1A 20.0", "70-AL1/11-ST1A 110.0",
            "94-AL1/15-ST1A 110.0", "122-AL1/20-ST1A 110.0", "149-AL1/24-ST1A 110.0", "184-AL1/30-ST1A 110.0",
            "243-AL1/39-ST1A 110.0", "305-AL1/39-ST1A 110.0", "490-AL1/64-ST1A 110.0", "490-AL1/64-ST1A 220.0",
            "490-AL1/64-ST1A 380.0"])
def en_net(request):
    # Create a Simple Example Network with 50 Hz
    # for 15, 24, 34, 48: the temperature is too low (ca. 71 °C rather than 80 °C), 679 ca. 76 °C
    net = create_single_line_net(request.param)
    return net


def create_single_line_net(std_type):
    net = pp.create_empty_network()
    pp.set_user_pf_options(net, init="dc", max_iteration=100)

    vn_kv = float(std_type.split(" ")[-1])
    b1 = pp.create_bus(net, vn_kv=vn_kv, name='b1_hv', type='n')
    b2 = pp.create_bus(net, vn_kv=vn_kv, name='b10', type='n')

    pp.create_line(net, from_bus=b1, to_bus=b2, length_km=vn_kv / 4, std_type=std_type, name='l3')
    max_i_ka = net.line.at[0, 'max_i_ka']

    # Initial standard Value
    net.line['max_loading_percent'] = 100

    # Chose the load to match nominal current
    p_ac = vn_kv * max_i_ka * np.sqrt(3)  # Q=0
    pp.create_load(net, b2, sn_mva=p_ac, p_mw=p_ac, name="load_b8", const_i_percent=100)

    pp.create_ext_grid(net, b1)

    # Declaration of overhead and cable systems
    ol_index = net.line.loc[net.line.type == "ol"].index.values
    pp.parameter_from_std_type(net, 'q_mm2')
    q_mm2 = net.line.loc[ol_index, "q_mm2"].values.astype(np.float64)
    # e.g. 0.0218 for 243-AL1/39-ST1A 110.0
    d_m = (np.sqrt(q_mm2 * 4 * (1 / np.pi))) * 1e-3 * 1.2  # 1.2 because not perfect circle

    # Standard Conditions Germany for Midsummer Weather
    net.line.loc[ol_index, "tdpf"] = True
    net.line.loc[ol_index, 'alpha'] = 0.00403
    net.line.loc[ol_index, 'wind_speed_m_per_s'] = 0.6
    net.line.loc[ol_index, 'wind_angle_degree'] = 90
    net.line.loc[ol_index, 'conductor_outer_diameter_m'] = d_m
    net.line.loc[ol_index, 'air_temperature_degree_celsius'] = 35
    net.line.loc[ol_index, 'temperature_degree_celsius'] = 20
    net.line.loc[ol_index, 'reference_temperature_degree_celsius'] = 20
    net.line.loc[ol_index, 'solar_radiation_w_per_sq_m'] = 900
    net.line.loc[ol_index, 'solar_absorptivity'] = 0.9
    net.line.loc[ol_index, 'emissivity'] = 0.72419

    return net


def prepare_case_30():
    net = pp.networks.case30()

    for i in net.line.index.values:
        net.line.at[i, 'name'] = f"{net.line.at[i, 'from_bus'] + 1} - {net.line.at[i, 'to_bus'] + 1}"

    # replicating the calculation from the paper Frank et.al.:
    net.line["alpha"] = 1 / (25 + 228.1)
    net.line["r_theta_kelvin_per_mw"] = calc_r_theta_from_t_rise(net, 25)
    net.line["tdpf"] = net.line.r_ohm_per_km != 0

    net.line["temperature_degree_celsius"] = 25
    net.line["reference_temperature_degree_celsius"] = 25
    net.line["air_temperature_degree_celsius"] = 25
    net.line["conductor_outer_diameter_m"] = 30.6e-3
    net.line["mc_joule_per_m_k"] = 1490
    net.line["wind_speed_m_per_s"] = 0.6
    net.line["wind_angle_degree"] = 45
    net.line["solar_radiation_w_per_sq_m"] = 900
    net.line["solar_absorptivity"] = 0.5
    net.line["emissivity"] = 0.5
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
    net.line["reference_temperature_degree_celsius"] = 20
    net.line["air_temperature_degree_celsius"] = 35
    net.line["alpha"] = 0.004
    net.line["conductor_outer_diameter_m"] = 30.6e-3
    net.line["mc_joule_per_m_k"] = 1490
    net.line["wind_speed_m_per_s"] = 0.6
    net.line["wind_angle_degree"] = 45
    net.line["solar_radiation_w_per_sq_m"] = 900
    net.line["solar_absorptivity"] = 0.5
    net.line["emissivity"] = 0.5
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
    pp.runpp(net, tdpf=True, tdpf_update_r_theta=False)

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
    pp.runpp(net, tdpf=True, tdpf_update_r_theta=False)

    v_base_kv = net.bus.loc[net.line.from_bus].vn_kv.values
    z_base_ohm = np.square(v_base_kv) / net.sn_mva
    temperature = net.res_line.temperature_degree_celsius
    net.line["r1"] = net.line.r_ohm_per_km / z_base_ohm
    net.line["r2"] = net.res_line.r_ohm_per_km / z_base_ohm
    r_delta = (net.res_line.r_ohm_per_km - net.line.r_ohm_per_km) / net.line.r_ohm_per_km * 100
    sorted_index = net.line.sort_values(by="r_theta_kelvin_per_mw", ascending=False).index

    line_mva = np.max(np.vstack([
        np.sqrt(net.res_line.p_from_mw ** 2 + net.res_line.q_from_mvar ** 2).loc[sorted_index].values,
        np.sqrt(net.res_line.p_to_mw ** 2 + net.res_line.q_to_mvar ** 2).loc[sorted_index].values]), axis=0)
    line_max_mva = net.line.max_i_ka.loc[sorted_index] * 135 * np.sqrt(3)
    line_loading = line_mva / line_max_mva

    ref2 = pd.read_csv(os.path.join(pp.pp_dir, "test", "test_files", "tdpf", "case30_branch_details.csv"))
    ref2 = ref2.sort_values(by="R_THETA", ascending=False)

    # compare p_loss
    assert np.allclose(ref2.PLoss_TDPF * net.sn_mva, net.res_line.loc[sorted_index, "pl_mw"], rtol=0, atol=1e-6)
    # compare R_Theta
    assert np.allclose(ref2.R_THETA, net.line.loc[sorted_index, "r_theta_kelvin_per_mw"], rtol=0, atol=1e-6)
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

            net.res_line = net.res_line.drop(["temperature_degree_celsius", "r_theta_kelvin_per_mw"], axis=1)
            assert_res_equal(net, net2)

            # now test transient results -> after 5 min
            pp.runpp(net, tdpf=True, tdpf_delay_s=5 * 60, max_iteration=100)

            net2.line["temperature_degree_celsius"] = net.res_line.temperature_degree_celsius
            pp.runpp(net2, consider_line_temperature=True)

            net.res_line = net.res_line.drop(["temperature_degree_celsius", "r_theta_kelvin_per_mw"], axis=1)
            assert_res_equal(net, net2)


def test_ngoko_vs_frank():
    net = simple_test_grid(load_scaling=0.25, sgen_scaling=0.5)
    pp.runpp(net)

    t_air_pu = 35
    alpha_pu = 4e-3
    r_ref = net.line.r_ohm_per_km.values / 1e3
    a0, a1, a2, tau = calc_a0_a1_a2_tau(t_air_pu, 80, 20, r_ref, 30.6e-3, 1490, 0.6, 45, 900, alpha_pu, 0.5, 0.5)
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
    r_theta_pu = calc_r_theta(t_air_pu, a0, a1, a2, np.square(net.res_line.i_ka.values * 1e3), p_loss_pu)
    T_frank = calc_T_frank(p_loss_pu, t_air_pu, r_theta_pu, None, None, None)

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


def test_only_pv():
    net = simple_test_grid(load_scaling=0.25, sgen_scaling=0.5, with_gen=True)
    pq = net.bus.loc[~net.bus.index.isin(np.union1d(net.gen.bus, net.ext_grid.bus))].index
    pp.create_gens(net, pq, 0)

    pp.runpp(net, init="flat")

    pp.runpp(net, tdpf=True)


def test_default_parameters():
    # length_km is important in the formulas
    net = pp.networks.case9()
    net.line.length_km = net.line.x_ohm_per_km / 4
    net.line.x_ohm_per_km /= net.line.length_km
    net.line.r_ohm_per_km /= net.line.length_km
    net.line.c_nf_per_km /= net.line.length_km
    net_backup = net.deepcopy()
    pp.runpp(net_backup)

    # test error is raised when 'tdpf' column is missng
    with pytest.raises(UserWarning, match="required columns .* are missing"):
        pp.runpp(net, tdpf=True)

    net.line["tdpf"] = np.nan
    with pytest.raises(UserWarning, match="required columns .*'conductor_outer_diameter_m'.* are missing"):
        pp.runpp(net, tdpf=True)

    # with TDPF algorithm but no relevant tdpf lines the results must match with normal runpp:
    net.line["conductor_outer_diameter_m"] = np.nan
    pp.runpp(net, tdpf=True)
    net.res_line = net.res_line.drop(["r_ohm_per_km", "temperature_degree_celsius", "r_theta_kelvin_per_mw"], axis=1)
    assert_res_equal(net, net_backup)

    with pytest.raises(UserWarning, match="required columns .*'mc_joule_per_m_k'.* are missing"):
        pp.runpp(net, tdpf=True, tdpf_delay_s=120)

    # check for simplified method
    net = net_backup.deepcopy()
    net.line["tdpf"] = np.nan
    with pytest.raises(UserWarning, match="required columns .* are missing"):
        pp.runpp(net, tdpf=True, tdpf_update_r_theta=False)
    net.line["r_theta_kelvin_per_mw"] = np.nan
    pp.runpp(net, tdpf=True, tdpf_update_r_theta=False)
    net.res_line = net.res_line.drop(["r_ohm_per_km", "temperature_degree_celsius", "r_theta_kelvin_per_mw"], axis=1)
    assert_res_equal(net, net_backup)

    # now define some tdpf lines with simplified method
    net.line.loc[[1, 2, 4], 'tdpf'] = True, 1, -8
    net.line.loc[1, 'alpha'] = 4.03e-3
    net.line["r_theta_kelvin_per_mw"] = calc_r_theta_from_t_rise(net, 25)
    with pytest.raises(UserWarning, match="required columns .* are missing"):
        pp.runpp(net, tdpf=True, tdpf_update_r_theta=False)
    net.line['alpha'] = 4.03e-3
    net.line["r_theta_kelvin_per_mw"] = calc_r_theta_from_t_rise(net, 25)
    pp.runpp(net, tdpf=True, tdpf_update_r_theta=False)

    # now test with "normal" TDPF
    net = net_backup.deepcopy()
    net.line.loc[net.line.r_ohm_per_km != 0, "tdpf"] = True
    net.line["conductor_outer_diameter_m"] = 2.5e-2  # 2.5 cm?
    pp.runpp(net, tdpf=True)
    # here all the standard assumptions are filled
    # now we check that user-defined assumptions are preserved
    net = net_backup.deepcopy()
    net.line.loc[net.line.r_ohm_per_km != 0, "tdpf"] = True
    net.line["conductor_outer_diameter_m"] = 2.5e-2  # 2.5 cm?
    net.line.loc[[2, 4], 'temperature_degree_celsius'] = 40
    net.line.loc[[2, 4], 'alpha'] = 3e-3
    net.line.loc[[2, 4], 'wind_speed_m_per_s'] = 0
    pp.runpp(net, tdpf=True)
    assert np.array_equal(net.line.loc[[2, 4], 'temperature_degree_celsius'].values, np.array([40, 40]))
    assert np.array_equal(net.line.loc[[2, 4], 'alpha'].values, np.array([3e-3, 3e-3]))
    assert np.array_equal(net.line.loc[[2, 4], 'wind_speed_m_per_s'].values, np.array([0, 0]))

    net.line["mc_joule_per_m_k"] = 500
    pp.runpp(net, tdpf=True, tdpf_delay_s=10 * 60)


def test_with_user_pf_options():
    net = simple_test_grid(0.5, 0.5)
    net2 = net.deepcopy()
    pp.set_user_pf_options(net, tdpf=True)
    pp.runpp(net)
    assert "r_ohm_per_km" in net.res_line
    assert "temperature_degree_celsius" in net.res_line
    assert len(net.res_line.loc[net.res_line.r_ohm_per_km.isnull()]) == 0
    assert len(net.res_line.loc[net.res_line.temperature_degree_celsius.isnull()]) == 0

    pp.runpp(net2, tdpf=True)
    assert_res_equal(net, net2)

    with pytest.raises(NotImplementedError):
        pp.runpp(net, algorithm="bfsw")


# Testing with a given Example from "IEEE Standard for Calculating the Current-Temperature
# Relationship of Bare Overhead Conductors" Basic Example with Calculation, Page 23 and following
def test_IEEE_example_1():
    # 60Hz according to USA
    net = pp.create_empty_network(f_hz=60.0)

    b1 = pp.create_bus(net, vn_kv=380, name='b1_hv', type='n')
    b2 = pp.create_bus(net, vn_kv=380, name='b10', type='n')

    # 795 kcmil 26/7 Drake ACSR
    pp.create_line_from_parameters(net, from_bus=b1, to_bus=b2, length_km=1, r_ohm_per_km=0.07283, x_ohm_per_km=0.341,
                                   c_nf_per_km=7.7827795, max_i_ka=1.024, type='ol')

    # Chose the load to match nominal current
    p_ac = 380 * 1.024 * np.sqrt(3)
    pp.create_load(net, b2, p_mw=p_ac, q_mvar=0, name="load_b8", const_i_percent=100)

    pp.create_ext_grid(net, b1, vm_pu=1, va_degree=0, s_sc_max_mva=20 * 110 * np.sqrt(3), rx_max=0.1)

    # Defining Overhead-Lines
    ol_index = net.line.loc[net.line.type == "ol"].index.values

    # Initial standard Value
    net.line['max_loading_percent'] = 100

    # Conditions according to IEEE-Example
    net.line.loc[ol_index, "tdpf"] = True
    net.line.loc[ol_index, "alpha"] = 0.003508
    net.line.loc[ol_index, 'conductor_outer_diameter_m'] = 0.02814
    net.line.loc[ol_index, 'air_temperature_degree_celsius'] = 40
    net.line.loc[ol_index, 'temperature_degree_celsius'] = 20
    net.line.loc[ol_index, 'reference_temperature_degree_celsius'] = 20
    net.line.loc[ol_index, 'wind_speed_m_per_s'] = 0.61
    net.line.loc[ol_index, 'wind_angle_degree'] = 90
    net.line.loc[ol_index, 'solar_radiation_w_per_sq_m'] = 1027
    net.line.loc[ol_index, 'solar_absorptivity'] = 0.8
    net.line.loc[ol_index, 'emissivity'] = 0.8

    # Starting Control Loop
    pp.runpp(net, tdpf=True, init="dc")

    # calculating current
    max_cond_temp = 100

    # Values
    assert np.isclose(net.res_line.i_ka.values, 1.025, atol=1e-5, rtol=1e-3)
    assert np.isclose(net.res_line.temperature_degree_celsius, max_cond_temp, atol=4)


# Testing with a given Example from "IEEE Standard for Calculating the Current-Temperature
# Relationship of Bare Overhead Conductors", Annex B, Numerical example 1, Page 44
def test_IEEE_example_2():
    # 60Hz according to USA
    net = pp.create_empty_network(f_hz=60.0)

    b1 = pp.create_bus(net, vn_kv=380, name='b1_hv', type='n')
    b2 = pp.create_bus(net, vn_kv=380, name='b10', type='n')

    # 400 MM2 26/7 Drake ACSR
    pp.create_line_from_parameters(net, from_bus=b1, to_bus=b2, length_km=1, r_ohm_per_km=0.07284, x_ohm_per_km=0.341,
                                   c_nf_per_km=7.7827795, max_i_ka=1.0, type='ol')

    # Initial standard Value
    net.line['max_loading_percent'] = 100

    # Chose the load to match nominal current
    p_ac = 380 * 1.0 * np.sqrt(3)  # Q=0
    pp.create_load(net, b2, p_mw=p_ac, q_mvar=0, name="load_b8", const_i_percent=100)

    pp.create_ext_grid(net, b1, vm_pu=1, va_degree=0, s_sc_max_mva=20 * 110 * np.sqrt(3), rx_max=0.1)

    # Defining Overhead-Lines
    ol_index = net.line.loc[net.line.type == "ol"].index.values

    # Conditions according to IEEE-Example
    net.line.loc[ol_index, 'tdpf'] = True
    net.line.loc[ol_index, "alpha"] = 0.003505
    net.line.loc[ol_index, 'conductor_outer_diameter_m'] = 0.02812
    net.line.loc[ol_index, 'air_temperature_degree_celsius'] = 40
    net.line.loc[ol_index, 'temperature_degree_celsius'] = 20
    net.line.loc[ol_index, 'reference_temperature_degree_celsius'] = 20
    net.line.loc[ol_index, 'wind_speed_m_per_s'] = 0.61
    net.line.loc[ol_index, 'wind_angle_degree'] = 90
    net.line.loc[ol_index, 'solar_radiation_w_per_sq_m'] = 1015
    net.line.loc[ol_index, 'solar_absorptivity'] = 0.5
    net.line.loc[ol_index, 'emissivity'] = 0.5

    # Starting Control Loop
    pp.runpp(net, tdpf=True, init="dc")

    # calculating current
    max_cond_temp = 100.7

    assert np.isclose(net.res_line.i_ka.values, 1.0, atol=1e-5, rtol=1e-3)
    assert np.isclose(net.res_line.temperature_degree_celsius, max_cond_temp, atol=5)


# Testing with a given Example from "IEEE Standard for Calculating the Current-Temperature
# Relationship of Bare Overhead Conductors", Annex B, Numerical example 2, Page 45
def test_IEEE_example_3():
    # 60Hz according to USA
    net = pp.create_empty_network(f_hz=60.0)

    b1 = pp.create_bus(net, vn_kv=380, name='b1_hv', type='n')
    b2 = pp.create_bus(net, vn_kv=380, name='b10', type='n')

    # 400 MM2 26/7 Drake ACSR
    pp.create_line_from_parameters(net, from_bus=b1, to_bus=b2, length_km=1, r_ohm_per_km=0.07284, x_ohm_per_km=0.341,
                                   c_nf_per_km=7.7827795, max_i_ka=1.003, type='ol')

    # Initial standard Value
    net.line['max_loading_percent'] = 100

    # Chose the load to match nominal current
    p_ac = 380 * 1.003 * np.sqrt(3)  # Q=0
    pp.create_load(net, b2, p_mw=p_ac, q_mvar=0, name="load_b8", const_i_percent=100)

    pp.create_ext_grid(net, b1, vm_pu=1, va_degree=0, s_sc_max_mva=20 * 110 * np.sqrt(3), rx_max=0.1)

    # Defining Overhead-Lines
    ol_index = net.line.loc[net.line.type == "ol"].index.values

    # Conditions according to IEEE-Example
    net.line.loc[ol_index, 'tdpf'] = True
    net.line.loc[ol_index, 'alpha'] = 0.003505
    net.line.loc[ol_index, 'conductor_outer_diameter_m'] = 0.02812
    net.line.loc[ol_index, 'air_temperature_degree_celsius'] = 40
    net.line.loc[ol_index, 'temperature_degree_celsius'] = 20
    net.line.loc[ol_index, 'reference_temperature_degree_celsius'] = 20
    net.line.loc[ol_index, 'wind_speed_m_per_s'] = 0.61
    net.line.loc[ol_index, 'wind_angle_degree'] = 90
    net.line.loc[ol_index, 'solar_radiation_w_per_sq_m'] = 1015
    net.line.loc[ol_index, 'solar_absorptivity'] = 0.5
    net.line.loc[ol_index, 'emissivity'] = 0.5

    # Starting Control Loop
    pp.runpp(net, tdpf=True, init="dc")

    # calculating current
    max_cond_temp = 101.1

    assert np.isclose(net.res_line.i_ka.values, 1.003, atol=1e-5, rtol=1e-3)
    assert np.isclose(net.res_line.temperature_degree_celsius, max_cond_temp, atol=5)


def test_EN_standard(en_net):
    """
    When combined with an assumed conductor absorptivity of no less than 0.8 and emissivity of no
    more than 0.1 below absorptivity, this combination can be considered safe for thermal rating
    calculations without field measurements.
    """
    pp.runpp(en_net, tdpf=True)

    max_cond_temp = 80

    assert np.isclose(en_net.res_line.i_ka, en_net.line.max_i_ka, rtol=0, atol=1e-3)
    assert np.isclose(en_net.res_line.temperature_degree_celsius, max_cond_temp, rtol=0,
                      atol=2), en_net.res_line.temperature_degree_celsius


if __name__ == '__main__':
    pytest.main(['-xs', __file__])

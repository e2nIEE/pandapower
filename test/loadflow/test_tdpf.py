# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
import os

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, hstack, vstack

from pandapower.create import (
    create_bus,
    create_ext_grid,
    create_gens,
    create_line,
    create_line_from_parameters,
    create_load,
)
from pandapower.network import pandapowerNet
from pandapower.networks.power_system_test_cases import case9, case30
from pandapower.networks.tutorial_networks import simple_test_grid
from pandapower.pf.create_jacobian_tdpf import (
    ALPHA_TDPF,
    calc_a0_a1_a2_tau,
    calc_g_b,
    calc_i_square_p_loss,
    calc_r_theta,
    calc_r_theta_from_t_rise,
    calc_T_frank,
    calc_T_ngoko,
    create_J_tdpf,
)
from pandapower.pypower.idx_brch import BR_R, BR_STATUS, BR_X, F_BUS, T_BUS
from pandapower.pypower.idx_brch_tdpf import TDPF
from pandapower.pypower.make_y_bus import makeYbus as makeYbus_pypower
from pandapower.pypower.newtonpf import _init_tdpf_ybus_update, _update_tdpf_ybus
from pandapower.run import runpp, set_user_pf_options
from pandapower.std_types import parameter_from_std_type
from test import test_path
from test.helper_functions import assert_res_equal


@pytest.fixture(
    params=[
        "94-AL1/15-ST1A 0.4",
        "70-AL1/11-ST1A 10.0",
        "94-AL1/15-ST1A 10.0",
        "122-AL1/20-ST1A 10.0",
        "149-AL1/24-ST1A 10.0",
        "70-AL1/11-ST1A 20.0",
        "94-AL1/15-ST1A 20.0",
        "122-AL1/20-ST1A 20.0",
        "149-AL1/24-ST1A 20.0",
        "184-AL1/30-ST1A 20.0",
        "243-AL1/39-ST1A 20.0",
        "70-AL1/11-ST1A 110.0",
        "94-AL1/15-ST1A 110.0",
        "122-AL1/20-ST1A 110.0",
        "149-AL1/24-ST1A 110.0",
        "184-AL1/30-ST1A 110.0",
        "243-AL1/39-ST1A 110.0",
        "305-AL1/39-ST1A 110.0",
        "490-AL1/64-ST1A 110.0",
        "490-AL1/64-ST1A 220.0",
        "490-AL1/64-ST1A 380.0",
    ]
)
def en_net(request):
    # Create a Simple Example Network with 50 Hz
    # for 15, 24, 34, 48: the temperature is too low (ca. 71 °C rather than 80 °C), 679 ca. 76 °C
    net = create_single_line_net(request.param)
    return net


def create_single_line_net(std_type):
    net = pandapowerNet(name="create_single_line_net")
    set_user_pf_options(net, init="dc", max_iteration=100)

    vn_kv = float(std_type.split(" ")[-1])
    b1 = create_bus(net, vn_kv=vn_kv, name="b1_hv", type="n")
    b2 = create_bus(net, vn_kv=vn_kv, name="b10", type="n")

    create_line(net, from_bus=b1, to_bus=b2, length_km=vn_kv / 4, std_type=std_type, name="l3")
    max_i_ka = net.line.at[0, "max_i_ka"]

    # Initial standard Value
    net.line["max_loading_percent"] = 100

    # Chose the load to match nominal current
    p_ac = vn_kv * max_i_ka * np.sqrt(3)  # Q=0
    create_load(net, b2, sn_mva=p_ac, p_mw=p_ac, name="load_b8", const_i_p_percent=100, const_i_q_percent=100)

    create_ext_grid(net, b1)

    # Declaration of overhead and cable systems
    ol_index = net.line.loc[net.line.type == "ol"].index.values
    parameter_from_std_type(net, "q_mm2")
    q_mm2 = net.line.loc[ol_index, "q_mm2"].values.astype(np.float64)
    # e.g. 0.0218 for 243-AL1/39-ST1A 110.0
    d_m = (np.sqrt(q_mm2 * 4 * (1 / np.pi))) * 1e-3 * 1.2  # 1.2 because not perfect circle

    # Standard Conditions Germany for Midsummer Weather
    net.line.loc[ol_index, "tdpf"] = True
    net.line.loc[ol_index, "alpha"] = 0.00403
    net.line.loc[ol_index, "wind_speed_m_per_s"] = 0.6
    net.line.loc[ol_index, "wind_angle_degree"] = 90
    net.line.loc[ol_index, "conductor_outer_diameter_m"] = d_m
    net.line.loc[ol_index, "air_temperature_degree_celsius"] = 35
    net.line.loc[ol_index, "temperature_degree_celsius"] = 20
    net.line.loc[ol_index, "reference_temperature_degree_celsius"] = 20
    net.line.loc[ol_index, "solar_radiation_w_per_sq_m"] = 900
    net.line.loc[ol_index, "solar_absorptivity"] = 0.9
    net.line.loc[ol_index, "emissivity"] = 0.72419

    return net


def prepare_case_30():
    net = case30()

    for i in net.line.index.values:
        net.line.at[i, "name"] = f"{net.line.at[i, 'from_bus'] + 1} - {net.line.at[i, 'to_bus'] + 1}"

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


def test_tdpf_frank_current():
    net = prepare_case_30()
    net.line["c_nf_per_km"] = 0  # otherwise i_square_pu will not match net.res_line.i_ka
    runpp(net, tdpf=True, tdpf_update_r_theta=False)

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
    runpp(net, tdpf=True, tdpf_update_r_theta=False)

    v_base_kv = net.bus.loc[net.line.from_bus].vn_kv.values
    z_base_ohm = np.square(v_base_kv) / net.sn_mva
    net.line["r1"] = net.line.r_ohm_per_km / z_base_ohm
    net.line["r2"] = net.res_line.r_ohm_per_km / z_base_ohm
    r_delta = (net.res_line.r_ohm_per_km - net.line.r_ohm_per_km) / net.line.r_ohm_per_km * 100
    sorted_index = net.line.sort_values(by="r_theta_kelvin_per_mw", ascending=False).index

    line_mva = np.max(
        np.vstack(
            [
                np.sqrt(net.res_line.p_from_mw**2 + net.res_line.q_from_mvar**2).loc[sorted_index].values,
                np.sqrt(net.res_line.p_to_mw**2 + net.res_line.q_to_mvar**2).loc[sorted_index].values,
            ]
        ),
        axis=0,
    )
    line_max_mva = net.line.max_i_ka.loc[sorted_index] * 135 * np.sqrt(3)
    line_loading = line_mva / line_max_mva

    ref2 = pd.read_csv(os.path.join(test_path, "test_files", "tdpf", "case30_branch_details.csv"))
    ref2 = ref2.sort_values(by="R_THETA", ascending=False)

    # compare p_loss
    assert np.allclose(ref2.PLoss_TDPF * net.sn_mva, net.res_line.loc[sorted_index, "pl_mw"], rtol=0, atol=1e-6)
    # compare R_Theta
    assert np.allclose(ref2.R_THETA, net.line.loc[sorted_index, "r_theta_kelvin_per_mw"], rtol=0, atol=1e-6)
    # compare difference of R
    assert np.allclose(ref2.R_diff.fillna(0) * 100, r_delta.loc[sorted_index].fillna(0), rtol=0, atol=1e-6)
    # compare loading
    assert np.allclose(ref2.pctloading_TDPF, line_loading, rtol=0, atol=0.025)


@pytest.fixture(scope="module")
def calc_a0_a1_a2_tau_for_simple_test_grid():
    net = simple_test_grid()
    r_ref = net.line.r_ohm_per_km.values / 1e3
    return calc_a0_a1_a2_tau(35, 80, 20, r_ref, 30.6e-3, 1490, 0.6, 45, 900, 4e-3, 0.5, 0.5)


@pytest.mark.parametrize("with_gen", [False, True])
@pytest.mark.parametrize("distributed_slack", [False, True])
def test_temperature_r(calc_a0_a1_a2_tau_for_simple_test_grid, with_gen, distributed_slack):
    a0, a1, a2, _ = calc_a0_a1_a2_tau_for_simple_test_grid
    net = simple_test_grid(load_scaling=0.25, sgen_scaling=0.5, with_gen=with_gen, distributed_slack=distributed_slack)
    runpp(net, tdpf=True, max_iteration=100)

    T = calc_T_ngoko(np.square(net.res_line.i_ka.values * 1e3), a0, a1, a2, None, None, None)
    assert np.allclose(net.res_line.temperature_degree_celsius, T, rtol=0, atol=1e-6)

    net2 = simple_test_grid(load_scaling=0.25, sgen_scaling=0.5, with_gen=with_gen, distributed_slack=distributed_slack)
    net2.line["temperature_degree_celsius"] = net.res_line.temperature_degree_celsius
    runpp(net2, consider_line_temperature=True)

    net.res_line = net.res_line.drop(["temperature_degree_celsius", "r_theta_kelvin_per_mw"], axis=1)
    assert_res_equal(net, net2)

    # now test transient results -> after 5 min
    runpp(net, tdpf=True, tdpf_delay_s=5 * 60, max_iteration=100)

    net2.line["temperature_degree_celsius"] = net.res_line.temperature_degree_celsius
    # test default value for alpha = 0.004
    net2.line.drop(columns="alpha", inplace=True)
    runpp(net2, consider_line_temperature=True)
    net2.line["alpha"] = ALPHA_TDPF

    net.res_line = net.res_line.drop(["temperature_degree_celsius", "r_theta_kelvin_per_mw"], axis=1)
    assert_res_equal(net, net2)


def test_ngoko_vs_frank():
    net = simple_test_grid(load_scaling=0.25, sgen_scaling=0.5)
    runpp(net)

    t_air_pu = 35
    alpha_pu = 4e-3
    r_ref = net.line.r_ohm_per_km.values / 1e3
    a0, a1, a2, _ = calc_a0_a1_a2_tau(t_air_pu, 80, 20, r_ref, 30.6e-3, 1490, 0.6, 45, 900, alpha_pu, 0.5, 0.5)
    T_ngoko = calc_T_ngoko(np.square(net.res_line.i_ka.values * 1e3), a0, a1, a2, None, None, None)

    branch = net._ppc["branch"]
    tdpf_lines = np.ones(len(branch)).astype(bool)
    r = branch[tdpf_lines, BR_R].real
    x = branch[tdpf_lines, BR_X].real
    g, b = calc_g_b(r, x)
    Vm = abs(net._ppc["internal"]["V"])
    Va = np.angle(net._ppc["internal"]["V"])
    i_square_pu, p_loss_pu = calc_i_square_p_loss(branch, tdpf_lines, g, b, Vm, Va)
    r_theta_pu = calc_r_theta(t_air_pu, a0, a1, a2, np.square(net.res_line.i_ka.values * 1e3), p_loss_pu)
    T_frank = calc_T_frank(p_loss_pu, t_air_pu, r_theta_pu, None, None, None)

    assert np.array_equal(T_ngoko, T_frank)

    i_base_a = net.sn_mva / (132 * np.sqrt(3)) * 1e3
    assert np.allclose(net.res_line.i_ka, 1e-3 * i_base_a * np.sqrt(i_square_pu), rtol=0, atol=1e-6)
    assert np.allclose(net.res_line.pl_mw, p_loss_pu * net.sn_mva, rtol=0, atol=1e-6)


@pytest.mark.parametrize("with_gen", [False, True])
@pytest.mark.parametrize("distributed_slack", [False, True])
def test_tdpf_delay(calc_a0_a1_a2_tau_for_simple_test_grid, with_gen, distributed_slack):
    _, _, _, tau = calc_a0_a1_a2_tau_for_simple_test_grid
    net = simple_test_grid(load_scaling=0.25, sgen_scaling=0.5, with_gen=with_gen, distributed_slack=distributed_slack)
    # no delay
    runpp(net, tdpf=True, max_iteration=100, tdpf_delay_s=0)
    assert np.allclose(net.res_line.temperature_degree_celsius, 20, rtol=0, atol=1e-6)

    # infinite delay (steady state)
    runpp(net, tdpf=True, max_iteration=100, tdpf_delay_s=np.inf)
    temp = net.res_line.temperature_degree_celsius.values.copy()
    runpp(net, tdpf=True, max_iteration=100, tdpf_delay_s=None)
    assert np.allclose(net.res_line.temperature_degree_celsius, temp, rtol=0, atol=1e-6)

    # check tau: time to "charge" to approx. 63.2 %; we cannot match it very accurately though
    runpp(net, tdpf=True, max_iteration=100, tdpf_delay_s=tau)
    assert np.allclose(net.res_line.temperature_degree_celsius, 20 + (temp - 20) * 0.632, rtol=0, atol=0.6)


def test_only_pv():
    net = simple_test_grid(load_scaling=0.25, sgen_scaling=0.5, with_gen=True)
    pq = net.bus.loc[~net.bus.index.isin(np.union1d(net.gen.bus, net.ext_grid.bus))].index
    create_gens(net, pq, 0)

    runpp(net, init="flat")

    runpp(net, tdpf=True)


def test_default_parameters():
    # length_km is important in the formulas
    net = case9()
    net.line.length_km = net.line.x_ohm_per_km / 4
    net.line.x_ohm_per_km /= net.line.length_km
    net.line.r_ohm_per_km /= net.line.length_km
    net.line.c_nf_per_km /= net.line.length_km
    net_backup = copy.deepcopy(net)
    runpp(net_backup)

    # test error is raised when 'tdpf' column is missng
    with pytest.raises(UserWarning, match="required columns .* are missing"):
        runpp(net, tdpf=True)

    net.line["tdpf"] = np.nan
    with pytest.raises(UserWarning, match="required columns .*'conductor_outer_diameter_m'.* are missing"):
        runpp(net, tdpf=True)

    # with TDPF algorithm but no relevant tdpf lines the results must match with normal runpp:
    net.line["conductor_outer_diameter_m"] = np.nan
    runpp(net, tdpf=True)
    net.res_line = net.res_line.drop(["r_ohm_per_km", "temperature_degree_celsius", "r_theta_kelvin_per_mw"], axis=1)
    assert_res_equal(net, net_backup)

    with pytest.raises(UserWarning, match="required columns .*'mc_joule_per_m_k'.* are missing"):
        runpp(net, tdpf=True, tdpf_delay_s=120)

    # check for simplified method
    net = copy.deepcopy(net_backup)
    net.line["tdpf"] = np.nan
    with pytest.raises(UserWarning, match="required columns .* are missing"):
        runpp(net, tdpf=True, tdpf_update_r_theta=False)
    net.line["r_theta_kelvin_per_mw"] = np.nan
    runpp(net, tdpf=True, tdpf_update_r_theta=False)
    net.res_line = net.res_line.drop(["r_ohm_per_km", "temperature_degree_celsius", "r_theta_kelvin_per_mw"], axis=1)
    assert_res_equal(net, net_backup)

    # now define some tdpf lines with simplified method
    net.line.loc[[1, 2, 4], "tdpf"] = True, 1, -8
    net.line.loc[1, "alpha"] = 4.03e-3
    net.line["r_theta_kelvin_per_mw"] = calc_r_theta_from_t_rise(net, 25)
    with pytest.raises(UserWarning, match="required columns .* are missing"):
        runpp(net, tdpf=True, tdpf_update_r_theta=False)
    net.line["alpha"] = 4.03e-3
    net.line["r_theta_kelvin_per_mw"] = calc_r_theta_from_t_rise(net, 25)
    runpp(net, tdpf=True, tdpf_update_r_theta=False)

    # now test with "normal" TDPF
    net = copy.deepcopy(net_backup)
    net.line.loc[net.line.r_ohm_per_km != 0, "tdpf"] = True
    net.line["conductor_outer_diameter_m"] = 2.5e-2  # 2.5 cm?
    runpp(net, tdpf=True)
    # here all the standard assumptions are filled
    # now we check that user-defined assumptions are preserved
    net = copy.deepcopy(net_backup)
    net.line.loc[net.line.r_ohm_per_km != 0, "tdpf"] = True
    net.line["conductor_outer_diameter_m"] = 2.5e-2  # 2.5 cm?
    net.line.loc[[2, 4], "temperature_degree_celsius"] = 40
    net.line.loc[[2, 4], "alpha"] = 3e-3
    net.line.loc[[2, 4], "wind_speed_m_per_s"] = 0
    runpp(net, tdpf=True)
    assert np.array_equal(net.line.loc[[2, 4], "temperature_degree_celsius"].values, np.array([40, 40]))
    assert np.array_equal(net.line.loc[[2, 4], "alpha"].values, np.array([3e-3, 3e-3]))
    assert np.array_equal(net.line.loc[[2, 4], "wind_speed_m_per_s"].values, np.array([0, 0]))

    net.line["mc_joule_per_m_k"] = 500
    runpp(net, tdpf=True, tdpf_delay_s=10 * 60)


def test_with_user_pf_options():
    net = simple_test_grid(0.5, 0.5)
    net2 = copy.deepcopy(net)
    set_user_pf_options(net, tdpf=True)
    runpp(net)
    assert "r_ohm_per_km" in net.res_line
    assert "temperature_degree_celsius" in net.res_line
    assert len(net.res_line.loc[net.res_line.r_ohm_per_km.isnull()]) == 0
    assert len(net.res_line.loc[net.res_line.temperature_degree_celsius.isnull()]) == 0

    runpp(net2, tdpf=True)
    assert_res_equal(net, net2)

    with pytest.raises(NotImplementedError):
        runpp(net, algorithm="bfsw")


# Testing with a given Example from "IEEE Standard for Calculating the Current-Temperature
# Relationship of Bare Overhead Conductors" Basic Example with Calculation, Page 23 and following
def test_IEEE_example_1():
    # 60Hz according to USA
    net = pandapowerNet(name="test_IEEE_example_1", f_hz=60.0)

    b1 = create_bus(net, vn_kv=380, name="b1_hv", type="n")
    b2 = create_bus(net, vn_kv=380, name="b10", type="n")

    # 795 kcmil 26/7 Drake ACSR
    create_line_from_parameters(
        net,
        from_bus=b1,
        to_bus=b2,
        length_km=1,
        r_ohm_per_km=0.07283,
        x_ohm_per_km=0.341,
        c_nf_per_km=7.7827795,
        max_i_ka=1.024,
        type="ol",
    )

    # Chose the load to match nominal current
    p_ac = 380 * 1.024 * np.sqrt(3)
    create_load(net, b2, p_mw=p_ac, q_mvar=0, name="load_b8", const_i_p_percent=100, const_i_q_percent=100)

    create_ext_grid(net, b1, vm_pu=1, va_degree=0, s_sc_max_mva=20 * 110 * np.sqrt(3), rx_max=0.1)

    # Defining Overhead-Lines
    ol_index = net.line.loc[net.line.type == "ol"].index.values

    # Initial standard Value
    net.line["max_loading_percent"] = 100

    # Conditions according to IEEE-Example
    net.line.loc[ol_index, "tdpf"] = True
    net.line.loc[ol_index, "alpha"] = 0.003508
    net.line.loc[ol_index, "conductor_outer_diameter_m"] = 0.02814
    net.line.loc[ol_index, "air_temperature_degree_celsius"] = 40
    net.line.loc[ol_index, "temperature_degree_celsius"] = 20
    net.line.loc[ol_index, "reference_temperature_degree_celsius"] = 20
    net.line.loc[ol_index, "wind_speed_m_per_s"] = 0.61
    net.line.loc[ol_index, "wind_angle_degree"] = 90
    net.line.loc[ol_index, "solar_radiation_w_per_sq_m"] = 1027
    net.line.loc[ol_index, "solar_absorptivity"] = 0.8
    net.line.loc[ol_index, "emissivity"] = 0.8

    # Starting Control Loop
    runpp(net, tdpf=True, init="dc")

    # calculating current
    max_cond_temp = 100

    # Values
    assert np.isclose(net.res_line.i_ka.values, 1.025, atol=1e-5, rtol=1e-3)
    assert np.isclose(net.res_line.temperature_degree_celsius, max_cond_temp, atol=4)


# Testing with a given Example from "IEEE Standard for Calculating the Current-Temperature
# Relationship of Bare Overhead Conductors", Annex B, Numerical example 1, Page 44
def test_IEEE_example_2():
    # 60Hz according to USA
    net = pandapowerNet(name="test_IEEE_example_2", f_hz=60.0)

    b1 = create_bus(net, vn_kv=380, name="b1_hv", type="n")
    b2 = create_bus(net, vn_kv=380, name="b10", type="n")

    # 400 MM2 26/7 Drake ACSR
    create_line_from_parameters(
        net,
        from_bus=b1,
        to_bus=b2,
        length_km=1,
        r_ohm_per_km=0.07284,
        x_ohm_per_km=0.341,
        c_nf_per_km=7.7827795,
        max_i_ka=1.0,
        type="ol",
    )

    # Initial standard Value
    net.line["max_loading_percent"] = 100

    # Chose the load to match nominal current
    p_ac = 380 * 1.0 * np.sqrt(3)  # Q=0
    create_load(net, b2, p_mw=p_ac, q_mvar=0, name="load_b8", const_i_p_percent=100, const_i_q_percent=100)

    create_ext_grid(net, b1, vm_pu=1, va_degree=0, s_sc_max_mva=20 * 110 * np.sqrt(3), rx_max=0.1)

    # Defining Overhead-Lines
    ol_index = net.line.loc[net.line.type == "ol"].index.values

    # Conditions according to IEEE-Example
    net.line.loc[ol_index, "tdpf"] = True
    net.line.loc[ol_index, "alpha"] = 0.003505
    net.line.loc[ol_index, "conductor_outer_diameter_m"] = 0.02812
    net.line.loc[ol_index, "air_temperature_degree_celsius"] = 40
    net.line.loc[ol_index, "temperature_degree_celsius"] = 20
    net.line.loc[ol_index, "reference_temperature_degree_celsius"] = 20
    net.line.loc[ol_index, "wind_speed_m_per_s"] = 0.61
    net.line.loc[ol_index, "wind_angle_degree"] = 90
    net.line.loc[ol_index, "solar_radiation_w_per_sq_m"] = 1015
    net.line.loc[ol_index, "solar_absorptivity"] = 0.5
    net.line.loc[ol_index, "emissivity"] = 0.5

    # Starting Control Loop
    runpp(net, tdpf=True, init="dc")

    # calculating current
    max_cond_temp = 100.7

    assert np.isclose(net.res_line.i_ka.values, 1.0, atol=1e-5, rtol=1e-3)
    assert np.isclose(net.res_line.temperature_degree_celsius, max_cond_temp, atol=5)


# Testing with a given Example from "IEEE Standard for Calculating the Current-Temperature
# Relationship of Bare Overhead Conductors", Annex B, Numerical example 2, Page 45
def test_IEEE_example_3():
    # 60Hz according to USA
    net = pandapowerNet(name="test_IEEE_example_3", f_hz=60.0)

    b1 = create_bus(net, vn_kv=380, name="b1_hv", type="n")
    b2 = create_bus(net, vn_kv=380, name="b10", type="n")

    # 400 MM2 26/7 Drake ACSR
    create_line_from_parameters(
        net,
        from_bus=b1,
        to_bus=b2,
        length_km=1,
        r_ohm_per_km=0.07284,
        x_ohm_per_km=0.341,
        c_nf_per_km=7.7827795,
        max_i_ka=1.003,
        type="ol",
    )

    # Initial standard Value
    net.line["max_loading_percent"] = 100

    # Chose the load to match nominal current
    p_ac = 380 * 1.003 * np.sqrt(3)  # Q=0
    create_load(net, b2, p_mw=p_ac, q_mvar=0, name="load_b8", const_i_p_percent=100, const_i_q_percent=100)

    create_ext_grid(net, b1, vm_pu=1, va_degree=0, s_sc_max_mva=20 * 110 * np.sqrt(3), rx_max=0.1)

    # Defining Overhead-Lines
    ol_index = net.line.loc[net.line.type == "ol"].index.values

    # Conditions according to IEEE-Example
    net.line.loc[ol_index, "tdpf"] = True
    net.line.loc[ol_index, "alpha"] = 0.003505
    net.line.loc[ol_index, "conductor_outer_diameter_m"] = 0.02812
    net.line.loc[ol_index, "air_temperature_degree_celsius"] = 40
    net.line.loc[ol_index, "temperature_degree_celsius"] = 20
    net.line.loc[ol_index, "reference_temperature_degree_celsius"] = 20
    net.line.loc[ol_index, "wind_speed_m_per_s"] = 0.61
    net.line.loc[ol_index, "wind_angle_degree"] = 90
    net.line.loc[ol_index, "solar_radiation_w_per_sq_m"] = 1015
    net.line.loc[ol_index, "solar_absorptivity"] = 0.5
    net.line.loc[ol_index, "emissivity"] = 0.5

    # Starting Control Loop
    runpp(net, tdpf=True, init="dc")

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
    runpp(en_net, tdpf=True)

    max_cond_temp = 80

    assert np.isclose(en_net.res_line.i_ka, en_net.line.max_i_ka, rtol=0, atol=1e-3)
    assert np.isclose(en_net.res_line.temperature_degree_celsius, max_cond_temp, rtol=0, atol=2), (
        en_net.res_line.temperature_degree_celsius
    )


@pytest.mark.parametrize("with_numba", [False, True])
def test_tdpf_ybus_consistent_after_run(with_numba):
    if with_numba:
        pytest.importorskip("numba")

    net = prepare_case_30()
    # an out-of-service TDPF line must not break the incremental Ybus update
    net.line.loc[net.line.index[5], "tdpf"] = True
    net.line.loc[net.line.index[5], "in_service"] = False
    runpp(net, tdpf=True, numba=with_numba)

    internal = net._ppc["internal"]
    Ybus, Yf, Yt = makeYbus_pypower(internal["baseMVA"], internal["bus"], internal["branch"])
    np.testing.assert_allclose(internal["Ybus"].toarray(), Ybus.toarray(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(internal["Yf"].toarray(), Yf.toarray(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(internal["Yt"].toarray(), Yt.toarray(), rtol=0, atol=1e-12)


@pytest.mark.parametrize("with_numba", [False, True])
def test_tdpf_incremental_ybus_matches_makeYbus(with_numba):
    if with_numba:
        pytest.importorskip("numba")
        from pandapower.pf.make_y_bus_numba import makeYbus as make_ybus_numba

        make_ybus = make_ybus_numba
    else:
        make_ybus = makeYbus_pypower

    net = prepare_case_30()
    runpp(net, tdpf=True, numba=False)
    ppci = net._ppc
    bus = ppci["bus"]
    branch = ppci["branch"].copy()
    tdpf_lines = np.flatnonzero(np.nan_to_num(branch[:, TDPF]))

    # add a parallel line (shared Ybus entry positions) and take one line out of service
    branch = np.vstack([branch, branch[tdpf_lines[0]].copy()])
    tdpf_lines = np.append(tdpf_lines, len(branch) - 1)
    branch[tdpf_lines[1], BR_STATUS] = 0

    Ybus, _, _ = make_ybus(ppci["baseMVA"], bus, branch)
    Ybus = Ybus.tocsr()
    update = _init_tdpf_ybus_update(Ybus, branch, tdpf_lines)
    assert update is not None
    positions, in_service = update
    lines = tdpf_lines[in_service]
    y_old = branch[lines, BR_STATUS].real / (branch[lines, BR_R] + 1j * branch[lines, BR_X])

    for factor in (1.02, 1.1, 0.95, 1.0, 1.3):
        branch[lines, BR_R] = branch[lines, BR_R].real * factor
        g, b = calc_g_b(branch[tdpf_lines, BR_R].real, branch[tdpf_lines, BR_X].real)
        y_old = _update_tdpf_ybus(Ybus, positions, in_service, y_old, g, b)
        expected, _, _ = make_ybus(ppci["baseMVA"], bus, branch)
        np.testing.assert_allclose(Ybus.toarray(), expected.toarray(), rtol=0, atol=1e-12)

    # no in-service TDPF line -> fall back to a full makeYbus rebuild
    all_off = branch.copy()
    all_off[tdpf_lines, BR_STATUS] = 0
    assert _init_tdpf_ybus_update(Ybus, all_off, tdpf_lines) is None


def _reference_create_J_tdpf(
    branch,
    tdpf_lines,
    alpha_pu,
    r_ref_pu,
    pvpq,
    pq,
    pvpq_lookup,
    pq_lookup,
    tau,
    tdpf_delay_s,
    Vm,
    Va,
    r_theta_pu,
    J,
    r,
    x,
    g,
):
    """Dense reference implementation of the TDPF Jacobian, as it was before the sparse assembly."""
    C = np.ones_like(tdpf_lines, dtype=np.float64)
    if tdpf_delay_s is not None and tdpf_delay_s != np.inf:
        C *= 1 - np.exp(-tdpf_delay_s / tau)

    dg_dT = (np.square(x) - np.square(r)) * alpha_pu * r_ref_pu / np.square(np.square(r) + np.square(x))
    db_dT = 2 * x * g * alpha_pu * r_ref_pu / (np.square(r) + np.square(x))

    in_pq_f = np.isin(branch[tdpf_lines, F_BUS].real.astype(np.int64), pq)
    in_pq_t = np.isin(branch[tdpf_lines, T_BUS].real.astype(np.int64), pq)
    in_pvpq_f = np.isin(branch[tdpf_lines, F_BUS].real.astype(np.int64), pvpq)
    in_pvpq_t = np.isin(branch[tdpf_lines, T_BUS].real.astype(np.int64), pvpq)

    n_branch = len(branch)

    def _dense_coupling(in_f, in_t, lookup, nrow, sign):
        block = np.zeros((nrow, n_branch), dtype=np.float64)
        if nrow == 0:
            return block
        f = branch[tdpf_lines, F_BUS].real.astype(np.int64)
        t = branch[tdpf_lines, T_BUS].real.astype(np.int64)
        for in_side, m, n in ((in_f, f, t), (in_t, t, f)):
            if not np.any(in_side):
                continue
            mm = m[in_side]
            nn = n[in_side]
            A_mn = Vm[mm] ** 2 - Vm[mm] * Vm[nn] * np.cos(Va[mm] - Va[nn])
            B_mn = Vm[mm] * Vm[nn] * np.sin(Va[mm] - Va[nn])
            if sign > 0:
                values = A_mn * dg_dT[in_side] - B_mn * db_dT[in_side]
            else:
                values = -A_mn * db_dT[in_side] - B_mn * dg_dT[in_side]
            block[lookup[mm], tdpf_lines[in_side]] = values
        return block

    def _dense_thermal(in_f, in_t, lookup, ncol, is_voltage):
        block = np.zeros((n_branch, ncol), dtype=np.float64)
        if ncol == 0:
            return block
        f = branch[tdpf_lines, F_BUS].real.astype(np.int64)
        t = branch[tdpf_lines, T_BUS].real.astype(np.int64)
        for in_side, m, n in ((in_f, f, t), (in_t, t, f)):
            if not np.any(in_side):
                continue
            mm = m[in_side]
            nn = n[in_side]
            B_mn = Vm[mm] * Vm[nn] * np.sin(Va[mm] - Va[nn])
            if is_voltage:
                values = -2 * r_theta_pu[in_side] * g[in_side] * B_mn * C[in_side]
            else:
                values = (
                    -2 * r_theta_pu[in_side] * g[in_side] * (Vm[mm] - Vm[nn] * np.cos(Va[mm] - Va[nn])) * C[in_side]
                )
            block[tdpf_lines[in_side], lookup[mm]] = values
        return block

    J13 = _dense_coupling(in_pvpq_f, in_pvpq_t, pvpq_lookup, len(pvpq), 1)
    J23 = _dense_coupling(in_pq_f, in_pq_t, pq_lookup, len(pq), -1)
    J31 = _dense_thermal(in_pvpq_f, in_pvpq_t, pvpq_lookup, len(pvpq), True)
    J32 = _dense_thermal(in_pq_f, in_pq_t, pq_lookup, len(pq), False)

    J33 = np.zeros((n_branch, n_branch), dtype=np.float64)
    J33[np.arange(n_branch), np.arange(n_branch)] = 1
    i = branch[tdpf_lines, F_BUS].real.astype(np.int64)
    j = branch[tdpf_lines, T_BUS].real.astype(np.int64)
    J33[tdpf_lines, tdpf_lines] = (
        1 - r_theta_pu * (Vm[i] ** 2 + Vm[j] ** 2 - 2 * Vm[i] * Vm[j] * np.cos(Va[i] - Va[j])) * dg_dT
    )

    Jright = vstack([csr_matrix(J13), csr_matrix(J23)], format="csr")
    Jbtm = hstack([csr_matrix(J31), csr_matrix(J32), csr_matrix(J33)], format="csr")
    return vstack([hstack([J, Jright]), Jbtm], format="csr")


@pytest.mark.parametrize("dist_slack", [False, True])
@pytest.mark.parametrize("tdpf_delay_s", [None, 0, np.inf, 120.0])
@pytest.mark.parametrize("empty_pq", [False, True])
def test_create_J_tdpf_matches_dense_reference(dist_slack, tdpf_delay_s, empty_pq):
    rng = np.random.default_rng(42)
    n_bus = 12
    n_branch = 15

    branch = np.zeros((n_branch, 30), dtype=np.float64)
    branch[:, F_BUS] = rng.integers(0, n_bus, n_branch)
    branch[:, T_BUS] = (branch[:, F_BUS] + rng.integers(1, n_bus, n_branch)) % n_bus
    branch[:, BR_R] = rng.uniform(0.01, 0.1, n_branch)
    branch[:, BR_X] = rng.uniform(0.05, 0.3, n_branch)

    tdpf_lines = np.arange(0, n_branch, 2)
    nl = len(tdpf_lines)

    pvpq = np.arange(1, n_bus)
    pq = np.arange(2, n_bus) if not empty_pq else np.array([], dtype=np.int64)
    if dist_slack:
        pvpq = np.arange(n_bus)
        if not empty_pq:
            pq = np.arange(1, n_bus)

    n_pvpq = len(pvpq)
    n_pq = len(pq)
    pvpq_lookup = np.zeros(int(max(pvpq)) + 1, dtype=np.int64)
    pvpq_lookup[pvpq] = np.arange(n_pvpq)
    if n_pq > 0:
        pq_lookup = np.zeros(int(max(pvpq)) + 1, dtype=np.int64)
        pq_lookup[pq] = np.arange(n_pq)
    else:
        pq_lookup = np.array([])

    Vm = rng.uniform(0.9, 1.1, n_bus)
    Va = rng.uniform(-0.3, 0.3, n_bus)
    r = branch[tdpf_lines, BR_R].real
    x = branch[tdpf_lines, BR_X].real
    g = r / (np.square(r) + np.square(x))
    alpha_pu = rng.uniform(0.3, 0.5, nl)
    r_ref_pu = rng.uniform(0.01, 0.05, nl)
    r_theta_pu = rng.uniform(0.5, 2.0, nl)
    tau = 10.0

    J = csr_matrix(rng.uniform(-1, 1, (n_pvpq + n_pq, n_pvpq + n_pq)))

    result = create_J_tdpf(
        branch,
        tdpf_lines,
        alpha_pu,
        r_ref_pu,
        pvpq,
        pq,
        pvpq_lookup,
        pq_lookup,
        tau,
        tdpf_delay_s,
        Vm,
        Va,
        r_theta_pu,
        J,
        r,
        x,
        g,
    )
    expected = _reference_create_J_tdpf(
        branch,
        tdpf_lines,
        alpha_pu,
        r_ref_pu,
        pvpq,
        pq,
        pvpq_lookup,
        pq_lookup,
        tau,
        tdpf_delay_s,
        Vm,
        Va,
        r_theta_pu,
        J,
        r,
        x,
        g,
    )

    assert result.shape == expected.shape
    assert (result != 0).nnz == (expected != 0).nnz
    np.testing.assert_allclose(result.toarray(), expected.toarray(), rtol=0, atol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

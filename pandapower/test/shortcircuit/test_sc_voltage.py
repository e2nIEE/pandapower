# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import os

import pytest

import numpy as np
import pandapower as pp
import pandapower.shortcircuit as sc

import pandas as pd
from pandapower.test.shortcircuit.test_iec60909_4 import iec_60909_4

#pd.set_option("display.width", 1000)
#pd.set_option("display.max_columns", 1000)


def simple_grid():
    net = pp.create_empty_network(sn_mva=4)
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)

    pp.create_ext_grid(net, b1, s_sc_max_mva=100., s_sc_min_mva=80., rx_min=0.4, rx_max=0.4)
    pp.create_line(net, b1, b2, std_type="305-AL1/39-ST1A 110.0", length_km=20.)
    pp.create_line(net, b2, b3, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV", length_km=15.)
    net.line["endtemp_degree"] = 80

    return net


def test_voltage_sgen():
    net = simple_grid()
    pp.create_sgen(net, 1, sn_mva=200., p_mw=0, k=1.3)
    # net.sn_mva = 110 * np.sqrt(3)
    sc.calc_sc(net, case="max", ip=True, branch_results=True, bus=2)

    assert np.isclose(net.res_bus_sc.at[2, "ikss_ka"], 1.825315, atol=1e-6, rtol=0)
    assert np.isclose(net.res_bus_sc.at[2, "skss_mw"], 347.769263, atol=1e-5, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "ikss_ka"], [0.460706, 1.825315], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "p_from_mw"], [5.019259, 14.843061], atol=1e-5, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "q_from_mvar"], [10.701325, 23.389066], atol=1e-5, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "p_to_mw"], [-3.810707, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "q_to_mvar"], [-5.862024, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "vm_from_pu"], [0.134660, 0.079654], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "va_from_degree"], [-2.919632, -10.818133], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "vm_to_pu"], [0.079654, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "va_to_degree"], [-10.818133, 0], atol=1e-6, rtol=0)

    sc.calc_sc(net, case="max", ip=True, branch_results=True, bus=1)

    assert np.isclose(net.res_bus_sc.at[1, "ikss_ka"], 1.860576, atol=1e-6, rtol=0)
    assert np.isclose(net.res_bus_sc.at[1, "skss_mw"], 354.487419, atol=1e-5, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "ikss_ka"], [0.495930, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "p_from_mw"], [1.400422, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "q_from_mvar"], [5.607589, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "p_to_mw"], [0, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "q_to_mvar"], [0, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "vm_from_pu"], [0.06117, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "va_from_degree"], [7.348078, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "vm_to_pu"], [0, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "va_to_degree"], [0, 0], atol=1e-6, rtol=0)


def test_voltage_gen():
    net = simple_grid()
    z_base_ohm = np.square(110) / 100
    pp.create_gen(net, 1, p_mw=0, sn_mva=100, vn_kv=110,
                  xdss_pu=0.14, rdss_ohm=0.00001653 * z_base_ohm, cos_phi=0.85, pg_percent=0)
    sc.calc_sc(net, case="max", ip=True, branch_results=True, bus=2)

    assert np.isclose(net.res_bus_sc.at[2, "ikss_ka"], 3.87955, atol=1e-6, rtol=0)
    assert np.isclose(net.res_bus_sc.at[2, "skss_mw"], 739.153586, atol=1e-5, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "ikss_ka"], [0.428409, 3.879550], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "p_from_mw"], [11.853270, 67.051803], atol=1e-5, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "q_from_mvar"], [12.794845, 105.657389], atol=1e-5, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "p_to_mw"], [-10.808227, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "q_to_mvar"], [-8.610268, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "vm_from_pu"], [0.213685, 0.169299], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "va_from_degree"], [-17.016990, -25.662429], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "vm_to_pu"], [0.169299, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "va_to_degree"], [-25.662429, 0], atol=1e-6, rtol=0)

    sc.calc_sc(net, case="max", ip=True, branch_results=True, bus=1)

    assert np.isclose(net.res_bus_sc.at[1, "ikss_ka"], 4.491006, atol=1e-6, rtol=0)
    assert np.isclose(net.res_bus_sc.at[1, "skss_mw"], 855.651656, atol=1e-5, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "ikss_ka"], [0.495930, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "p_from_mw"], [1.400422, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "q_from_mvar"], [5.607589, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "p_to_mw"], [0, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "q_to_mvar"], [0, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "vm_from_pu"], [0.06117, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "va_from_degree"], [7.348078, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "vm_to_pu"], [0, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "va_to_degree"], [0, 0], atol=1e-6, rtol=0)


def test_voltage_simple():
    net = simple_grid()
    sc.calc_sc(net, case="max", ip=True, ith=True, branch_results=True, bus=2)

    assert np.isclose(net.res_bus_sc.at[2, "ikss_ka"], 0.486532, atol=1e-6, rtol=0)
    assert np.isclose(net.res_bus_sc.at[2, "skss_mw"], 92.696717, atol=1e-5, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "ikss_ka"], [0.486532, 0.486532], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "p_from_mw"], [2.4024, 1.054556], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "q_from_mvar"], [7.058781, 1.661725], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "p_to_mw"], [-1.054556, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "q_to_mvar"], [-1.661725, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "vm_from_pu"], [0.080439, 0.021232], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "va_from_degree"], [2.786108, -10.818133], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "vm_to_pu"], [0.021232, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "va_to_degree"], [-10.818133, 0], atol=1e-6, rtol=0)

    sc.calc_sc(net, case="max", ip=True, ith=True, branch_results=True, bus=1)

    assert np.isclose(net.res_bus_sc.at[1, "ikss_ka"], 0.49593, atol=1e-6, rtol=0)
    assert np.isclose(net.res_bus_sc.at[1, "skss_mw"], 94.487419, atol=1e-5, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "ikss_ka"], [0.49593, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "p_from_mw"], [1.400422, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "q_from_mvar"], [5.607589, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "p_to_mw"], [0, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "q_to_mvar"], [0, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "vm_from_pu"], [0.06117, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "va_from_degree"], [7.348078, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "vm_to_pu"], [0, 0], atol=1e-6, rtol=0)
    assert np.allclose(net.res_line_sc.loc[:, "va_to_degree"], [0, 0], atol=1e-6, rtol=0)


def test_voltage_very_simple():
    net = pp.create_empty_network(sn_mva=12)
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    pp.create_ext_grid(net, b1, s_sc_max_mva=100., s_sc_min_mva=80., rx_min=0.4, rx_max=0.4)
    pp.create_line(net, b1, b2, std_type="305-AL1/39-ST1A 110.0", length_km=20.)
    sc.calc_sc(net, case="max", ip=True, ith=True, branch_results=True, bus=1)

    assert np.isclose(net.res_bus_sc.at[1, "ikss_ka"], 0.49593, atol=1e-6, rtol=0)
    assert np.isclose(net.res_bus_sc.at[1, "skss_mw"], 94.487419, atol=1e-5, rtol=0)
    assert np.isclose(net.res_line_sc.at[0, "ikss_ka"], 0.49593, atol=1e-6, rtol=0)
    assert np.isclose(net.res_line_sc.at[0, "p_from_mw"], 1.400422, atol=1e-6, rtol=0)
    assert np.isclose(net.res_line_sc.at[0, "q_from_mvar"], 5.607589, atol=1e-6, rtol=0)
    assert np.isclose(net.res_line_sc.at[0, "p_to_mw"], 0, atol=1e-6, rtol=0)
    assert np.isclose(net.res_line_sc.at[0, "q_to_mvar"], 0, atol=1e-6, rtol=0)
    assert np.isclose(net.res_line_sc.at[0, "vm_from_pu"], 0.06117, atol=1e-6, rtol=0)
    assert np.isclose(net.res_line_sc.at[0, "va_from_degree"], 7.348078, atol=1e-6, rtol=0)
    assert np.isclose(net.res_line_sc.at[0, "vm_to_pu"], 0, atol=1e-6, rtol=0)
    assert np.isclose(net.res_line_sc.at[0, "va_to_degree"], 0, atol=1e-6, rtol=0)


def test_iec_60909_4():
    net = iec_60909_4()
    sc.calc_sc(net, case="max", ip=True, ith=True, branch_results=True, bus=2)


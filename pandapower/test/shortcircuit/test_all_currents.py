# -*- coding: utf-8 -*-

# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pytest
from scipy.linalg import inv

import pandapower as pp
import pandapower.shortcircuit as sc
from pandapower.pf.makeYbus_numba import makeYbus
from pandapower.pypower.idx_bus_sc import IKSS1, IKSS2, PHI_IKSS1_DEGREE, C_MAX, SKSS
from pandapower.pypower.idx_bus import BS, GS
from pandapower.pypower.idx_brch import F_BUS, T_BUS, TAP, BR_R, BR_X

from pandapower.pypower.idx_bus_sc import *
from pandapower.pypower.idx_bus import *


def three_bus_example():
    net = pp.create_empty_network(sn_mva=56)
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)

    pp.create_ext_grid(net, b1, s_sc_max_mva=100., s_sc_min_mva=80., rx_min=0.4, rx_max=0.4)
    pp.create_line(net, b1, b2, std_type="305-AL1/39-ST1A 110.0" , length_km=20.)
    pp.create_line(net, b2, b3, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV" , length_km=15.)
    net.line["endtemp_degree"] = 80

    pp.create_sgen(net, b2, sn_mva=2, p_mw=0, k=1.2)

    net.ext_grid['x0x_min'] = 0.1
    net.ext_grid['r0x0_min'] = 0.1
    net.ext_grid['x0x_max'] = 0.1
    net.ext_grid['r0x0_max'] = 0.1

    net.line['r0_ohm_per_km'] = 0.1
    net.line['x0_ohm_per_km'] = 0.1
    net.line['c0_nf_per_km'] = 0.1
    net.line["endtemp_degree"] = 80
    return net


def three_bus_permuted_index():
    net = pp.create_empty_network(sn_mva=67)
    b1 = pp.create_bus(net, 110, index=4)
    b2 = pp.create_bus(net, 110, index=3)
    b3 = pp.create_bus(net, 110, index=0)

    pp.create_ext_grid(net, b1, s_sc_max_mva=100., s_sc_min_mva=80., rx_min=0.4, rx_max=0.4)
    pp.create_line(net, b1, b2, std_type="305-AL1/39-ST1A 110.0" , length_km=20., index=1)
    pp.create_line(net, b2, b3, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV" , length_km=15., index=0)
    net.line["endtemp_degree"] = 80

    pp.create_sgen(net, b2, sn_mva=2, p_mw=0, k=1.2)
    return net


# def gen_three_bus_example():
#     net = pp.create_empty_network(sn_mva=2)
#     b1 = pp.create_bus(net, vn_kv=10.)
#     b2 = pp.create_bus(net, vn_kv=10.)
#     b3 = pp.create_bus(net, vn_kv=10.)
#     #pp.create_bus(net, vn_kv=0.4, in_service=False)
#     pp.create_gen(net, b2, vn_kv=10.5, xdss_pu=0.2, rdss_pu=0.001, cos_phi=0.8, p_mw=0.1, sn_mva=2.5)
#     pp.create_line_from_parameters(net, b1, b2, length_km=1.0, max_i_ka=0.29,
#                                    r_ohm_per_km=0.1548, x_ohm_per_km=0.0816814, c_nf_per_km=165)
#     pp.create_line_from_parameters(net, b2, b3, length_km=1.0, max_i_ka=0.29,
#                                    r_ohm_per_km=0.1548, x_ohm_per_km=0.0816814, c_nf_per_km=165)
#     net.line["endtemp_degree"] = 165
#     pp.create_ext_grid(net, b1, s_sc_max_mva=10., s_sc_min_mva=8., rx_min=0.4, rx_max=0.4)
#     #pp.create_switch(net, b3, b1, et="b")
#     return net


def net_transformer_simple():
    net = pp.create_empty_network(sn_mva=2)
    b1 = pp.create_bus(net, vn_kv=10.)
    b2 = pp.create_bus(net, vn_kv=.4)
    pp.create_ext_grid(net, b1, s_sc_max_mva=100., s_sc_min_mva=40., rx_min=0.1, rx_max=0.1)
    pp.create_transformer_from_parameters(net, b1, b2, vn_hv_kv=10., vn_lv_kv=0.4, vk_percent=6.,
                                          vkr_percent=0.5, pfe_kw=14, shift_degree=0.0,
                                          tap_side="hv", tap_neutral=0, tap_min=-2, tap_max=2, tap_pos=0,
                                          tap_step_percent=2.5, parallel=1, sn_mva=0.4, i0_percent=0.5)
    return net


def net_transformer_simple_2():
    net = pp.create_empty_network(sn_mva=2)
    b1 = pp.create_bus(net, vn_kv=10.)
    b1a = pp.create_bus(net, vn_kv=10.)
    b2 = pp.create_bus(net, vn_kv=.4)
    b3 = pp.create_bus(net, vn_kv=.4)
    b4 = pp.create_bus(net, vn_kv=.4)
    pp.create_ext_grid(net, b1, s_sc_max_mva=100., s_sc_min_mva=40., rx_min=0.1, rx_max=0.1)
    pp.create_transformers_from_parameters(net, [b1, b1a], [b2, b3], vn_hv_kv=10., vn_lv_kv=0.4, vk_percent=6.,
                                           vkr_percent=0.5, pfe_kw=14, shift_degree=0.0,
                                           tap_side="hv", tap_neutral=0, tap_min=-2, tap_max=2, tap_pos=0,
                                           tap_step_percent=2.5, parallel=1, sn_mva=0.4, i0_percent=0.5)
    pp.create_line_from_parameters(net, b1, b1a, 1, 0.099, 0.156, 125, 0.457)
    pp.create_line_from_parameters(net, b3, b4, 1, 0.099, 0.156, 125, 0.457)
    return net


def net_transformer_simple_3():
    net = pp.create_empty_network(sn_mva=100)
    pp.create_buses(net, 2, 30)
    pp.create_buses(net, 3, 10)
    pp.create_buses(net, 2, 0.4)

    pp.create_ext_grid(net, 0, s_sc_max_mva=100., s_sc_min_mva=40., rx_min=0.1, rx_max=0.1)
    # 30/10
    pp.create_transformers_from_parameters(net, [0, 1], [2, 3], sn_mva=10, vn_hv_kv=30., vn_lv_kv=10,
                                           vk_percent=6., vkr_percent=0.5, i0_percent=0.1, pfe_kw=10,
                                           shift_degree=0.0, tap_side="hv", tap_neutral=0, tap_min=-2,
                                           tap_max=2, tap_pos=0, tap_step_percent=2.5, parallel=1)
    # 10/0.4
    pp.create_transformer_from_parameters(net, 4, 5, sn_mva=0.4, vn_hv_kv=10., vn_lv_kv=0.4, vk_percent=6.,
                                          vkr_percent=0.5, pfe_kw=14, i0_percent=0.5, shift_degree=0.0,
                                          tap_side="hv", tap_neutral=0, tap_min=-2, tap_max=2, tap_pos=0,
                                          tap_step_percent=2.5, parallel=1)
    pp.create_lines_from_parameters(net, [0, 3, 5], [1, 4, 6], 1, 0.099, 0.156, 400, 0.457)
    return net


def net_transformer_simple_4():
    net = net_transformer_simple_3()

    pp.create_bus(net, 10)
    pp.create_bus(net, 0.4)
    pp.create_lines_from_parameters(net, [2, 8], [7, 6], 1, 0.099, 0.156, 400, 0.457)
    pp.create_transformer_from_parameters(net, 7, 8, sn_mva=0.4, vn_hv_kv=10., vn_lv_kv=0.4, vk_percent=6.,
                                          vkr_percent=0.5, pfe_kw=14, i0_percent=0.5, shift_degree=0.0,
                                          tap_side="hv", tap_neutral=0, tap_min=-2, tap_max=2, tap_pos=0,
                                          tap_step_percent=2.5, parallel=1)
    pp.create_switch(net, 6, 4, "l", closed=True)
    pp.create_sgen(net, 3, 0, 0, 20, k=1.2, kappa=1)
    pp.create_load(net, 6, 0.1)
    return net


def net_transformer():
    net = pp.create_empty_network(sn_mva=2)
    b1a = pp.create_bus(net, vn_kv=10.)
    b1b = pp.create_bus(net, vn_kv=10.)
    b2 = pp.create_bus(net, vn_kv=.4)
    pp.create_bus(net, vn_kv=0.4, in_service=False) #add out of service bus to test oos indexing
    pp.create_ext_grid(net, b1a, s_sc_max_mva=100., s_sc_min_mva=40., rx_min=0.1, rx_max=0.1)
    pp.create_switch(net, b1a, b1b, et="b")
    pp.create_transformer_from_parameters(net, b1b, b2, vn_hv_kv=11., vn_lv_kv=0.42, vk_percent=6.,
                                          vkr_percent=0.5, pfe_kw=14, shift_degree=0.0,
                                          tap_side="hv", tap_neutral=0, tap_min=-2, tap_max=2, tap_pos=2,
                                          tap_step_percent=2.5, parallel=2, sn_mva=0.4, i0_percent=0.5)
    pp.create_shunt(net, b2, q_mvar=0.050, p_mw=0.0500) #adding a shunt shouldn't change the result
    return net


def test_all_currents_sgen():
    #
    # eg--0---l0---1---l1---2
    #              |
    #              g
    #
    net = three_bus_example()
    sc.calc_sc(net, case="max", ip=True, ith=True, branch_results=True, return_all_currents=True)
    assert np.allclose(net.res_line_sc.ikss_ka.values,
                       np.array([0.01259673, 0.49593036, 0.48628848, 0., 0., 0.49888962]), atol=1e-5)
    assert np.allclose(net.res_line_sc.ip_ka.values,
                       np.array([0.01781447, 0.92787447, 0.90729584, 0., 0., 0.92511655]), atol=1e-5)
    assert np.allclose(net.res_line_sc.ith_ka.values,
                       np.array([0.01265116, 0.4981196, 0.48841266, 0., 0., 0.50106884]), atol=1e-5)

    sc.calc_sc(net, case="min", ip=True, ith=True, branch_results=True, return_all_currents=True)
    assert np.allclose(net.res_line_sc.ikss_ka.values,
                       np.array([0.01259673, 0.3989686, 0.39170662, 0., 0., 0.40431286]), atol=1e-5)
    assert np.allclose(net.res_line_sc.ip_ka.values,
                       np.array([0.01781447, 0.74438751, 0.72793774, 0., 0., 0.74576565]), atol=1e-5)
    assert np.allclose(net.res_line_sc.ith_ka.values,
                       np.array([0.01265116, 0.40071219, 0.39339323, 0., 0., 0.40605375]), atol=1e-5)


def test_all_currents_1ph_max():
    # Only check coherence between branch currents and bus currents
    #
    # eg--0---l0---1---l1---2
    #              |
    #              g
    #
    # With generator
    net = three_bus_example()
    sc.calc_sc(net, case="max", fault='1ph', branch_results=True, return_all_currents=True)
    i_bus_with_sgen = net.res_bus_sc.copy()
    i_line_with_gen = net.res_line_sc.copy()

    # Without generator
    net = three_bus_example()
    net.sgen.in_service = False
    sc.calc_sc(net, case="max", fault='1ph')
    i_bus_without_sgen = net.res_bus_sc.copy()

    # Isolate sgen contrib
    i_bus_only_sgen = i_bus_with_sgen - i_bus_without_sgen

    assert np.isclose(i_line_with_gen.ikss_ka.loc[(0, 0)], i_bus_only_sgen.ikss_ka.at[0], atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.loc[(0, 1)], i_bus_without_sgen.ikss_ka.at[1], atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.loc[(0, 2)], i_bus_without_sgen.ikss_ka.at[2] -
                      (i_bus_only_sgen.ikss_ka.at[1] - i_bus_only_sgen.ikss_ka.at[2]) , atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.loc[(1, 0)], 0., atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.loc[(1, 1)], 0., atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.loc[(1, 2)], i_bus_with_sgen.ikss_ka.at[2], atol=1e-4)


def test_all_currents_1ph_min():
    # Only check coherence between branch currents and bus currents
    #
    # eg--0---l0---1---l1---2
    #              |
    #              g
    #
    # With generator
    net = three_bus_example()
    sc.calc_sc(net, case="min", fault='1ph', branch_results=True, return_all_currents=True)
    i_bus_with_sgen = net.res_bus_sc.copy()
    i_line_with_gen = net.res_line_sc.copy()

    # Without generator
    net.sgen.in_service = False
    sc.calc_sc(net, case="min", fault='1ph', branch_results=True)
    i_bus_without_sgen = net.res_bus_sc.copy()

    # Isolate sgen contrib
    i_bus_only_sgen = i_bus_with_sgen - i_bus_without_sgen

    assert np.isclose(i_line_with_gen.ikss_ka.loc[(0, 0)], i_bus_only_sgen.ikss_ka.at[0], atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.loc[(0, 1)], i_bus_without_sgen.ikss_ka.at[1], atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.loc[(0, 2)], i_bus_without_sgen.ikss_ka.at[2] -
                      (i_bus_only_sgen.ikss_ka.at[1] - i_bus_only_sgen.ikss_ka.at[2]) , atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.loc[(1, 0)], 0., atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.loc[(1, 1)], 0., atol=1e-4)
    assert np.isclose(i_line_with_gen.ikss_ka.loc[(1, 2)], i_bus_with_sgen.ikss_ka.at[2], atol=1e-4)


def test_with_permuted_index():
    # Check that if element's index are permuted the results are still consistent
    #
    # eg--4---l1---3---l0---1
    #              |
    #              g
    #
    net = three_bus_permuted_index()
    sc.calc_sc(net, case="max", ip=True, ith=True, branch_results=True, return_all_currents=True)
    assert np.allclose(net.res_line_sc.ikss_ka.loc[[(1, 4), (1, 3), (1, 0), (0, 4), (0, 3), (0, 0)]].values,
                       np.array([0.01259673, 0.49593036, 0.48628848, 0., 0., 0.49888962]), atol=1e-5)
    sc.calc_sc(net, case="min", ip=True, ith=True, branch_results=True, return_all_currents=True)
    assert np.allclose(net.res_line_sc.ikss_ka.loc[[(1, 4), (1, 3), (1, 0), (0, 4), (0, 3), (0, 0)]].values,
                       np.array([0.01259673, 0.3989686, 0.39170662, 0., 0., 0.40431286]), atol=1e-5)


def test_all_currents_with_oos_elements():

    net = three_bus_example()
    net.bus.in_service.loc[2] = False
    net.line.in_service.loc[1] = False
    sc.calc_sc(net, case="max", branch_results=True, return_all_currents=True)

    assert np.allclose(net.res_line_sc.ikss_ka.loc[[(0, 0), (0, 1)]].values,
                       np.array([0.01259673, 0.49593036]), atol=1e-5)
    assert np.allclose(net.res_line_sc.ikss_ka.loc[[(0, 2), (1, 0), (1, 1), (1, 2)]].values,
                       0, atol=1e-10)

    sc.calc_sc(net, case="min", branch_results=True, return_all_currents=True)
    assert np.allclose(net.res_line_sc.ikss_ka.loc[[(0, 0), (0, 1)]].values,
                       np.array([0.01259673, 0.3989686]), atol=1e-5)
    assert np.allclose(net.res_line_sc.ikss_ka.loc[[(0, 2), (1, 0), (1, 1), (1, 2)]].values,
                       0, atol=1e-10)


# TODO: This example should not work anymore
# def test_branch_all_currents_gen():
#     net = gen_three_bus_example()
#     sc.calc_sc(net, case="max", branch_results=True, return_all_currents=True)
#     assert np.allclose(net.res_line_sc.ikss_ka.values,
#                        np.array([0.76204252, 0.57040645, 0.55786693, 0., 0., 1.28698045]))

#     sc.calc_sc(net, case="min", branch_results=True, return_all_currents=True)
#     assert np.allclose(net.res_line_sc.ikss_ka.values,
#                        np.array([0.69255026, 0.45574755, 0.44487882, 0., 0., 1.10747517]))


def test_branch_all_currents_trafo_simple():
    net = net_transformer_simple()
    sc.calc_sc(net, case='max', ip=True, ith=True, lv_tol_percent=6., branch_results=True, bus=1)

    assert np.isclose(net.res_bus_sc.ikss_ka, 9.749917, atol=1e-6, rtol=0)
    assert np.isclose(net.res_bus_sc.skss_mw, 6.75494, atol=1e-6, rtol=0)

    assert np.isclose(net.res_trafo_sc.ikss_hv_ka, 0.389997, atol=1e-6, rtol=0)
    assert np.isclose(net.res_trafo_sc.ikss_lv_ka, 9.749916, atol=1e-6, rtol=0)

    assert np.isclose(net.res_trafo_sc.p_hv_mw, 0.549236, atol=1e-6, rtol=0)
    assert np.isclose(net.res_trafo_sc.q_hv_mvar, 6.567902, atol=1e-6, rtol=0)
    assert np.isclose(net.res_trafo_sc.p_lv_mw, 0, atol=1e-6, rtol=0)
    assert np.isclose(net.res_trafo_sc.q_lv_mvar, 0, atol=1e-6, rtol=0)

    assert np.isclose(net.res_trafo_sc.vm_hv_pu, 0.975705, atol=1e-6, rtol=0)
    assert np.isclose(net.res_trafo_sc.va_hv_degree, 0.065838, atol=1e-6, rtol=0)
    assert np.isclose(net.res_trafo_sc.vm_lv_pu, 0, atol=1e-6, rtol=0)
    assert np.isclose(net.res_trafo_sc.va_lv_degree, 0, atol=1e-6, rtol=0)

    sc.calc_sc(net, case='max', ip=True, ith=True, lv_tol_percent=6., branch_results=False, bus=0)

    assert np.isclose(net.res_bus_sc.ikss_ka, 5.773503, atol=1e-6, rtol=0)
    assert np.isclose(net.res_bus_sc.skss_mw, 100, atol=1e-6, rtol=0)


def test_branch_all_currents_trafo_simple_other_voltage():
    net = net_transformer_simple()
    net.sn_mva=1
    # net.trafo.vn_hv_kv = 11
    bus_idx = 1
    pp.create_load(net, 1, 0.5)
    sc.calc_sc(net, case='max', ip=True, ith=True, lv_tol_percent=6., branch_results=True, bus=bus_idx)

    pp.runpp(net)
    sc.calc_sc(net, case='max', ip=True, ith=True, lv_tol_percent=6., branch_results=True, bus=bus_idx, use_pre_fault_voltage=True)

    ppci = net.ppci
    bus = ppci["bus"]
    branch = ppci["branch"]
    Ybus = ppci["internal"]["Ybus"]
    Yf = ppci["internal"]["Yf"]
    Yt = ppci["internal"]["Yt"]
    Zbus = ppci["internal"]["Zbus"]
    baseI = ppci["internal"]["baseI"]

    ikss1 = ppci["bus"][:, IKSS1] * np.exp(1j * np.deg2rad(ppci["bus"][:, PHI_IKSS1_DEGREE]))

    assert np.allclose(Zbus, inv(Ybus.toarray()))

    V_ikss = (ikss1 * baseI * Zbus)[:, bus_idx]
    Ibus = Ybus * V_ikss

    V_ikss_t = 1.05 - V_ikss
    Ibus_t = Ybus * V_ikss_t

    Ibus[abs(Ibus) < 1e-10] = np.nan

    assert np.allclose(ikss1 * baseI, Ibus, equal_nan=True)

    Sbus = V_ikss * Ibus.conj()
    abs(Sbus)

    Sbus = ppci["bus"][:, SKSS]

    Sbus_t = V_ikss_t * Ibus_t.conj()
    abs(Sbus_t)

    Zbus.real

    tap = ppci["branch"][:, TAP]
    branch_nonzero = np.flatnonzero(tap!=1)
    ppci["branch"][branch_nonzero, BR_R] *= tap[branch_nonzero]
    ppci["branch"][branch_nonzero, BR_X] *= tap[branch_nonzero]
    Ybus, Yf, Yt = makeYbus(net.sn_mva, ppci["bus"], ppci["branch"])
    Zbus = inv(Ybus.toarray())
    V_ikss = (ikss1 * baseI * Zbus)  # making it a complex calculation
    V_ikss = V_ikss[:, bus_idx]
    V_ikss = V_ikss[np.argmax(np.abs(V_ikss), axis=0)] - V_ikss  # numpy indexing issue

    # Z_line = 0.02407386 + 0.28788144j
    Z_line = 0.02648124 + 0.31666959j
    U_0 = ikss1[bus_idx] * baseI[bus_idx] * Z_line
    U_0_ref = 1.086619 * np.exp(np.deg2rad(0.055088)*1j)
    Z_line_ref = U_0_ref / (ikss1[bus_idx] * baseI[bus_idx])
    U_0_try = ikss1[bus_idx] * baseI[bus_idx] * Z_line_ref
    Z_line_ref / Z_line


def add_aux_trafo(net, trafo_idx):
    hv_bus = net.trafo.at[trafo_idx, 'hv_bus']
    aux_bus = pp.create_bus(net, net.trafo.at[trafo_idx, 'vn_hv_kv'])
    net.trafo.at[trafo_idx, 'hv_bus'] = aux_bus
    pp.create_transformer_from_parameters(net, hv_bus, aux_bus, net.trafo.at[trafo_idx, 'sn_mva'],
                                          net.bus.at[hv_bus, 'vn_kv'], net.bus.at[aux_bus, 'vn_kv'],
                                          1e-6, 1e-5, 0, 0)




def test_branch_all_currents_trafo_simple_other_voltage2():
    net = net_transformer_simple_2()
    net.trafo.vn_hv_kv.at[1] = 11

    sc.calc_sc(net, case='max', lv_tol_percent=6., branch_results=True, bus=4, use_pre_fault_voltage=False)


def test_branch_all_currents_trafo_simple_other_voltage3():
    net = net_transformer_simple_3()
    net.trafo.vn_hv_kv.at[1] = 31
    net.trafo.vn_hv_kv.at[2] = 11

    sc.calc_sc(net, case='max', lv_tol_percent=6., branch_results=True, bus=6)

    assert np.allclose(net.res_bus_sc.loc[6].values,
                       [1.16712302, 0.80860656, 0.10127268, 0.18141131], rtol=0, atol=1e-6)

    res_line_sc = np.array([[0.013691, 0.013691, -60.827605, 0.013691, 119.172395, 0.4132998, 0.7358034, -0.413244,
                             -0.735716, 1.186326, -0.150470, 1.186180, -0.150091],
                            [0.042441, 0.042441, -60.827605, 0.042441, 119.172395, 0.412972, 0.732456, -0.4124366,
                             -0.7316133, 1.143870, -0.242663, 1.142514, -0.239117],
                            [1.167123, 1.167123, -60.827605, 1.167123, 119.172395, 0.404566, 0.637498, 0, 0, 0.933749,
                             -3.227445, 0.000000, 0.000000]])
    assert np.allclose(net.res_line_sc.values, res_line_sc, rtol=0, atol=1e-6)
    res_trafo_sc = np.array([[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.186326,
                              - 0.150470, 1.186326, - 0.150470],
                             [0.013691, - 60.827605, 0.042441, 119.172395, 0.413244, 0.735716, - 0.412972, - 0.732456,
                              1.186180, - 0.150091, 1.143870, - 0.242663],
                             [0.042441, - 60.827605, 1.167123, 119.172395, 0.412437, 0.731613, - 0.404566, - 0.637498,
                              1.142514, - 0.239117, 0.933749, - 3.227445]])
    assert np.allclose(net.res_trafo_sc.values, res_trafo_sc, rtol=0, atol=1e-6)

    net.trafo.vn_hv_kv.at[1] = 29
    net.trafo.vn_hv_kv.at[2] = 9

    sc.calc_sc(net, case='max', lv_tol_percent=6., branch_results=True, bus=6)

    assert np.allclose(net.res_bus_sc.loc[6].values,
                       [1.15940813, 0.80326152, 0.10147573, 0.18288051], rtol=0, atol=1e-6)

    res_line_sc = np.array([[0.017769, 0.017769, -60.975224, 0.017769, 119.024776, 0.408286, 0.728168, -0.4081925,
                             -0.7280205, 0.904182, - 0.254709, 0.903993, - 0.254055],
                            [0.051529, 0.051529, -60.975224, 0.051529, 119.024776, 0.407791, 0.723216, -0.4070021,
                             -0.7219731, 0.930252, - 0.392031, 0.928605, - 0.386736],
                            [1.159408, 1.159408, -60.975224, 1.159408, 119.024776, 0.399235, 0.629098, 0, 0, 0.927576,
                             - 3.375064, 0, 0]])
    assert np.allclose(net.res_line_sc.values, res_line_sc, rtol=0, atol=1e-6)
    res_trafo_sc = np.array([[0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.904182,
                              - 0.254709, 0.904182, - 0.254709],
                             [0.017769, - 60.975224, 0.051529, 119.024776, 0.408192, 0.728021, - 0.407791, - 0.723216,
                              0.903993, - 0.254055, 0.930252, - 0.392031],
                             [0.051529, - 60.975224, 1.159408, 119.024776, 0.407002, 0.721973, - 0.399235, - 0.629098,
                              0.928605, - 0.386736, 0.927576, - 3.375064]])
    assert np.allclose(net.res_trafo_sc.values, res_trafo_sc, rtol=0, atol=1e-6)

    net.trafo.vn_hv_kv.at[1] = 31
    net.trafo.vn_hv_kv.at[2] = 11

    sc.calc_sc(net, case='max', lv_tol_percent=6., branch_results=True, bus=4)
    assert np.allclose(net.res_bus_sc.loc[4].values,
                       [3.490484425, 60.45696064, 0.26224866, 1.80047754], rtol=0, atol=3e-6)
    res_line_sc = np.array([[1.125963, 1.125963, -81.712857, 1.125963, 98.287143, 5.838649, 28.341696, -5.462115,
                             -27.748369, 0.494590, - 3.353461, 0.483378, - 2.848843],
                            [3.490484, 3.490484, -81.712857, 3.490484, 98.287143, 3.618494, 5.701869, 0, 0, 0.111701,
                             - 24.112697, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert np.allclose(net.res_line_sc.values, res_line_sc, rtol=0, atol=3e-6)
    res_trafo_sc = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0.494590, -3.353461, 0.494590, -3.353461],
                             [1.125963, - 81.712857, 3.490484, 98.287143, 5.462115, 27.748369, -3.618494, -5.701869,
                              0.483378, -2.848843, 0.111701, -24.112697],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert np.allclose(net.res_trafo_sc.values, res_trafo_sc, rtol=0, atol=2e-6)

    net.trafo.vn_hv_kv.at[1] = 30
    net.trafo.vn_hv_kv.at[2] = 10
    net.trafo.vn_lv_kv.at[1] = 9
    net.trafo.vn_lv_kv.at[2] = 0.42

    sc.calc_sc(net, case='max', lv_tol_percent=6., branch_results=True, bus=6)

    assert np.allclose(net.res_bus_sc.loc[6].values,
                       [1.153265, 0.799006, 0.101542, 0.184117], rtol=0, atol=1e-6)
    res_line_sc = np.array([[0.014531, 0.014531, - 61.122864, 0.014531, 118.877136, 0.404535, 0.728398, - 0.404473,
                             - 0.728299, 1.103480, - 0.169658, 1.103325, - 0.169187],
                            [0.048437, 0.048437, - 61.122864, 0.048437, 118.877136, 0.404185, 0.724860, - 0.403488,
                             - 0.723762, 0.989244, - 0.267095, 0.987697, - 0.261989],
                            [1.153265, 1.153265, - 61.122864, 1.153265, 118.877136, 0.395016, 0.622450, 0, 0, 0.922662,
                             - 3.522703, 0, 0]])
    assert np.allclose(net.res_line_sc.values, res_line_sc, rtol=0, atol=1e-6)
    res_trafo_sc = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1.103480, - 0.169658, 1.103480, - 0.169658],
                             [0.014531, - 61.122864, 0.048437, 118.877136, 0.404473, 0.728299, - 0.404185, - 0.72486,
                              1.103325, - 0.169187, 0.989244, - 0.267095],
                             [0.048437, - 61.122864, 1.153265, 118.877136, 0.403488, 0.723762, - 0.395016, - 0.62245,
                              0.987697, - 0.261989, 0.922662, - 3.522703]])
    assert np.allclose(net.res_trafo_sc.values, res_trafo_sc, rtol=0, atol=1e-6)


def test_type_c_trafo_simple_other_voltage4():
    # this tests for 2 different topology configurations: ring and open ring,
    # with options for different transformer nominal voltages, with load and sgen
    # topology tests: open switch, out of service bus (important to test internal indexing)
    net = net_transformer_simple_4()

    # net.trafo.vn_hv_kv.at[1] = 31
    net.trafo.vn_hv_kv.at[2] = 11

    pp.runpp(net)
    sc.calc_sc(net, case='max', lv_tol_percent=6., branch_results=True, bus=6, use_pre_fault_voltage=True)

    baseMVA = net.sn_mva  # MVA
    baseV = net.bus.vn_kv.values  # kV
    baseI = baseMVA / (baseV * np.sqrt(3))
    baseZ = baseV ** 2 / baseMVA

    assert np.allclose(net.res_bus_sc.loc[6].values,
                       [2.096122,  1.452236,  0.054549,  0.085178], rtol=0, atol=1e-6)

    res_line_sc = np.array([[0.012104, 0.010331, -52.747518, 0.012104, 121.050735, 0.322524, 0.419803, -0.322487,
                             -0.529717, 0.986152, -0.281725, 0.986032, -0.281941],
                            [0.036634, 0.036019, -59.720977, 0.036634, 119.710805, 0.312602, 0.527369, -0.312210,
                             -0.538872, 0.982660, -0.378583, 0.981499, -0.376180],
                            [0.998604, 0.998594, -61.123876, 0.998604, 118.875767, 0.296171, 0.466687, 0, 0, 0.798926,
                             -3.524072, 0, 0],
                            [0.044289, 0.043675, -59.839870, 0.044289, 119.691851, 0.377698, 0.639716, -0.377123,
                             -0.650913, 0.982060, -0.398116, 0.980654, -0.395135],
                            [1.097518, 1.097507, -61.142831, 1.097518, 118.856813, 0.357750, 0.563720, 0, 0, 0.878062,
                             -3.543027, 0, 0]])
    non_i_degree = [i for i, c in enumerate(net.res_line_sc.columns) if c not in ["ikss_from_degree", "ikss_to_degree"]]
    i_degree = [i for i, c in enumerate(net.res_line_sc.columns) if c in ["ikss_from_degree", "ikss_to_degree"]]
    assert np.allclose(net.res_line_sc.values[:, non_i_degree], res_line_sc[:, non_i_degree], rtol=0, atol=2e-6)
    assert np.allclose(net.res_line_sc.values[:, i_degree], res_line_sc[:, i_degree], rtol=0, atol=2e-4)

    res_trafo_sc = np.array([[0.014655, -59.201943, 0.043675, 120.160131, 0.387670, 0.643162, -0.377698, -0.639716,
                              0.986152, -0.281725, 0.982060, -0.398116],
                             [0.012104, -58.949266, 0.036019, 120.279024, 0.322487, 0.529717, -0.312602, -0.527369,
                              0.986032, -0.281941, 0.982660, -0.378583],
                             [0.036634, -60.289196, 0.998594, 118.876124, 0.312210, 0.538872, -0.296171, -0.466687,
                              0.981499, -0.376180, 0.798926, -3.524072],
                             [0.044289, -60.308150, 1.097507, 118.857170, 0.377123, 0.650913, -0.357750, -0.563720,
                              0.980654, -0.395135, 0.878062, -3.543027]])
    non_i_degree = [i for i, c in enumerate(net.res_trafo_sc.columns) if c not in ["ikss_hv_degree", "ikss_lv_degree"]]
    i_degree = [i for i, c in enumerate(net.res_trafo_sc.columns) if c in ["ikss_lv_degree", "ikss_lv_degree"]]
    assert np.allclose(net.res_trafo_sc.values[:, non_i_degree], res_trafo_sc[:, non_i_degree], rtol=0, atol=2e-6)
    assert np.allclose(net.res_trafo_sc.values[:, i_degree], res_trafo_sc[:, i_degree], rtol=0, atol=1e-5)


def test_type_c_trafo_simple_other_voltage4_sgen():
    # this tests for 2 different topology configurations: ring and open ring,
    # with options for different transformer nominal voltages, with load and sgen
    # topology tests: open switch, out of service bus (important to test internal indexing)
    net = net_transformer_simple_4()
    net.sgen.p_mw = 15
    net.sgen.kappa = 0

    # net.trafo.vn_hv_kv.at[1] = 31
    # net.trafo.vn_hv_kv.at[2] = 11

    pp.runpp(net)
    sc.calc_sc(net, case='max', lv_tol_percent=6., branch_results=True, bus=6, use_pre_fault_voltage=True)

    baseMVA = net.sn_mva  # MVA
    baseV = net.bus.vn_kv.values  # kV
    baseI = baseMVA / (baseV * np.sqrt(3))
    baseZ = baseV ** 2 / baseMVA

    assert np.allclose(net.res_bus_sc.loc[6].values,
                       [2.096122,  1.452236,  0.054549,  0.085178], rtol=0, atol=1e-6)

    res_line_sc = np.array([[0.012104, 0.010331, -52.747518, 0.012104, 121.050735, 0.322524, 0.419803, -0.322487,
                             -0.529717, 0.986152, -0.281725, 0.986032, -0.281941],
                            [0.036634, 0.036019, -59.720977, 0.036634, 119.710805, 0.312602, 0.527369, -0.312210,
                             -0.538872, 0.982660, -0.378583, 0.981499, -0.376180],
                            [0.998604, 0.998594, -61.123876, 0.998604, 118.875767, 0.296171, 0.466687, 0, 0, 0.798926,
                             -3.524072, 0, 0],
                            [0.044289, 0.043675, -59.839870, 0.044289, 119.691851, 0.377698, 0.639716, -0.377123,
                             -0.650913, 0.982060, -0.398116, 0.980654, -0.395135],
                            [1.097518, 1.097507, -61.142831, 1.097518, 118.856813, 0.357750, 0.563720, 0, 0, 0.878062,
                             -3.543027, 0, 0]])
    non_i_degree = [i for i, c in enumerate(net.res_line_sc.columns) if c not in ["ikss_from_degree", "ikss_to_degree"]]
    i_degree = [i for i, c in enumerate(net.res_line_sc.columns) if c in ["ikss_from_degree", "ikss_to_degree"]]
    assert np.allclose(net.res_line_sc.values[:, non_i_degree], res_line_sc[:, non_i_degree], rtol=0, atol=2e-6)
    assert np.allclose(net.res_line_sc.values[:, i_degree], res_line_sc[:, i_degree], rtol=0, atol=2e-4)

    res_trafo_sc = np.array([[0.014655, -59.201943, 0.043675, 120.160131, 0.387670, 0.643162, -0.377698, -0.639716,
                              0.986152, -0.281725, 0.982060, -0.398116],
                             [0.012104, -58.949266, 0.036019, 120.279024, 0.322487, 0.529717, -0.312602, -0.527369,
                              0.986032, -0.281941, 0.982660, -0.378583],
                             [0.036634, -60.289196, 0.998594, 118.876124, 0.312210, 0.538872, -0.296171, -0.466687,
                              0.981499, -0.376180, 0.798926, -3.524072],
                             [0.044289, -60.308150, 1.097507, 118.857170, 0.377123, 0.650913, -0.357750, -0.563720,
                              0.980654, -0.395135, 0.878062, -3.543027]])
    non_i_degree = [i for i, c in enumerate(net.res_trafo_sc.columns) if c not in ["ikss_hv_degree", "ikss_lv_degree"]]
    i_degree = [i for i, c in enumerate(net.res_trafo_sc.columns) if c in ["ikss_lv_degree", "ikss_lv_degree"]]
    assert np.allclose(net.res_trafo_sc.values[:, non_i_degree], res_trafo_sc[:, non_i_degree], rtol=0, atol=2e-6)
    assert np.allclose(net.res_trafo_sc.values[:, i_degree], res_trafo_sc[:, i_degree], rtol=0, atol=1e-5)


def test_type_c_sgen_trafo4():
    net = pp.create_empty_network(sn_mva=100)
    pp.create_buses(net, 2, 110)
    pp.create_ext_grid(net, 0, s_sc_max_mva=100, s_sc_min_mva=80, rx_max=0.4, rx_min=0.4)
    pp.create_line_from_parameters(net, 0, 1, 20, 0.0949, 0.38, 9.2, 0.74)
    pp.create_sgen(net, 1, 30, 0, 50, k=1, kappa=1)
    pp.runpp(net)
    sc.calc_sc(net, use_pre_fault_voltage=True, branch_results=True, bus=1)

    assert np.allclose(net.res_bus_sc.values, [0.680859,  129.720994, 3.049814, 152.46668], rtol=0, atol=1e-6)

    res_line_sc = [0.4184267, 0.41833479, -87.77946741, 0.4184267, 92.21738882, 0.99691065,
                   3.99091307, 0., 0., 0.05161055, -11.80466263, 0., 0.]
    assert np.allclose(net.res_line_sc.values, res_line_sc, rtol=0, atol=1e-6)

    net.sgen.at[0, 'kappa'] = 1.2
    sc.calc_sc(net, use_pre_fault_voltage=True, branch_results=True, bus=1)

    assert np.allclose(net.res_bus_sc.values, [0.6821, 129.95747, 3.049814, 152.46668], rtol=0, atol=1e-6)

    res_line_sc = [0.4184267, 0.41833479, -87.77946741, 0.4184267, 92.21738882, 0.99691065,
                   3.99091307, 0., 0., 0.05161055, -11.80466263, 0., 0.]
    assert np.allclose(net.res_line_sc.values, res_line_sc, rtol=0, atol=1e-6)


def test_load_type_c():
    net = pp.create_empty_network(sn_mva=100)
    pp.create_buses(net, 3, 110)
    pp.create_ext_grid(net, 0, s_sc_max_mva=100, rx_max=0.1)
    pp.create_line_from_parameters(net, 0, 1, 20, 0.0949, 0.38, 9.2, 0.74)
    pp.create_line_from_parameters(net, 1, 2, 20, 0.099, 0.156, 400, 0.74)
    pp.create_load(net, 1, 10, 3)

    pp.runpp(net)
    sc.calc_sc(net, use_pre_fault_voltage=True, branch_results=True, bus=2)

    res_bus_sc = np.array([0.31443438,  59.9077952,  66.64954063, 194.98202182])
    assert np.allclose(net.res_bus_sc.loc[2].values, res_bus_sc, rtol=0, atol=1e-5)

    res_line_sc = np.array([[0.31393855, 0.31381306, -71.61339493, 0.31393855, 108.37684429, 1.15160782, 3.16677295,
                             -0.59052299, -0.92129652, 0.05635891, -1.59736465, 0.01829535, -14.28178926],
                            [0.31443438, 0.31320255, -71.73882892, 0.31443438, 108.11805023, 0.58728173, 0.92032414, 0.,
                             0., 0.01829535, -14.28178926, 0., 0.]])
    assert np.allclose(net.res_line_sc.values, res_line_sc, rtol=0, atol=1e-6)

    # now test for fault at the load bus
    sc.calc_sc(net, use_pre_fault_voltage=True, branch_results=True, bus=1)
    res_bus_sc = np.array([ 0.32128852,  61.21368488,  63.19657030, 190.67254950])
    assert np.allclose(net.res_bus_sc.loc[1].values, res_bus_sc, rtol=0, atol=1e-5)

    res_line_sc = np.array([[0.32128852, 0.32121795, -72.27006831, 0.32128852, 107.72678791, 0.58777062, 2.35301071,
                             -0.0000001, -0.0000001, 0.03962911, 3.70473647, 0., 0.],
                            [0., 0., 0., 0., 0., 0., -0., 0., 0., 0., 0., 0., 0.]])
    assert np.allclose(net.res_line_sc.values, res_line_sc, rtol=0, atol=1e-6)

    # now try with positive p_mw, negative q_mvar
    net.load.q_mvar = -2
    pp.runpp(net)
    sc.calc_sc(net, use_pre_fault_voltage=True, branch_results=True, bus=2)

    res_bus_sc = np.array([ 0.29176418,  55.58854147,  77.40936019, 208.87369732])
    assert np.allclose(net.res_bus_sc.loc[2].values, res_bus_sc, rtol=0, atol=1e-5)

    res_line_sc = np.array([[0.29093901, 0.29082267, -70.14924675, 0.29093901, 109.84097608, 0.9903086, 2.72035994,
                             -0.50842396, -0.79184572, 0.05224782, -0.1525674, 0.01697628, -12.86251045],
                            [0.29176418, 0.29062116, -70.31955011, 0.29176418, 109.53732905, 0.50565042, 0.79240043, 0,
                             0, 0.01697628, -12.86251045, 0., 0.]])
    assert np.allclose(net.res_line_sc.values, res_line_sc, rtol=0, atol=1e-6)

    # now try with negative p_mw, negative q_mvar
    net.load.p_mw = -5
    pp.runpp(net)
    sc.calc_sc(net, use_pre_fault_voltage=True, branch_results=True, bus=2)

    res_bus_sc = np.array([ 0.27677319,  52.73237570,  22.92942615, 234.25805144])
    assert np.allclose(net.res_bus_sc.loc[2].values, res_bus_sc, rtol=0, atol=1.1e-5)

    res_line_sc = np.array([[0.27533477, 0.27522443, -84.57867986, 0.27533477, 95.41156578, 0.88536175, 2.43975772,
                             -0.45378216, -0.71256769, 0.04949597, -14.52396818, 0.01610403, -27.07850858],
                            [0.27677319, 0.27568891, -84.53554824, 0.27677319, 95.32133091, 0.4550242, 0.7130645, 0, 0,
                             0.01610403, -27.07850858, 0., 0.]])
    assert np.allclose(net.res_line_sc.values, res_line_sc, rtol=0, atol=1.1e-6)

    # now try with zero p_mw and q_mvar, but the load still present
    net.load.p_mw = 0
    net.load.q_mvar = 0

    pp.runpp(net)
    sc.calc_sc(net, use_pre_fault_voltage=True, branch_results=True, bus=2)

    res_bus_sc = np.array([ 0.28905271 ,  55.07193832,  40.36906664, 221.27455269])
    assert np.allclose(net.res_bus_sc.loc[2].values, res_bus_sc, rtol=0, atol=1e-5)

    res_line_sc = np.array([[0.28792032, 0.28780504, -79.93483554, 0.28792032, 100.05540903, 0.96823224, 2.66644109,
                             -0.49629573, -0.77774077, 0.05173395, -9.8916964, 0.01681852, -22.48755131],
                            [0.28905271, 0.28792032, -79.94459097, 0.28905271, 99.91228819, 0.49629573, 0.77774077, 0,
                             0, 0.01681852, -22.48755131, 0., 0.]])
    assert np.allclose(net.res_line_sc.values, res_line_sc, rtol=0, atol=1e-6)


def test_sgen_type_c():
    net = pp.create_empty_network(sn_mva=100)
    pp.create_buses(net, 3, 110)
    pp.create_ext_grid(net, 0, s_sc_max_mva=100, rx_max=0.1)
    pp.create_line_from_parameters(net, 0, 1, 20, 0.0949, 0.38, 9.2, 0.74)
    pp.create_line_from_parameters(net, 1, 2, 20, 0.099, 0.156, 400, 0.74)
    pp.create_sgen(net, 1, 10, 3, 20, k=1, kappa=1)

    pp.runpp(net)
    sc.calc_sc(net, use_pre_fault_voltage=True, branch_results=True, bus=2)

    res_bus_sc = np.array([0.37460791, 71.37239201, 1.86837663, 240.35204316])
    assert np.allclose(net.res_bus_sc.loc[2].values, res_bus_sc, rtol=0, atol=1.1e-5)

    i_degree = [i for i, c in enumerate(net.res_line_sc.columns) if c in ["ikss_from_degree", "ikss_to_degree"]]
    non_i_degree = [i for i, c in enumerate(net.res_line_sc.columns) if c not in ["ikss_from_degree", "ikss_to_degree"]]
    res_line_sc = np.array([[0.26731345, 0.26718739, -89.36194328, 0.26731345, 90.6256479, 1.00620834, 2.56195025,
                             -0.59943766, -0.93434154, 0.05406945, -20.80440736, 0.02179654, -32.05702438],
                            [0.37460791, 0.37314034, -89.51406404, 0.37460791, 90.34281512, 0.83356663, 1.30627509, 0,
                             0, 0.02179654, -32.05702438, 0., 0.]])
    # assert np.allclose(net.res_line_sc.values[:, i_degree], res_line_sc[:, i_degree], rtol=0, atol=1e-1)
    # assert np.allclose(net.res_line_sc.values[:, non_i_degree], res_line_sc[:, non_i_degree], rtol=0, atol=2e-6)
    #
    # assert np.allclose(net.res_line_sc.values, res_line_sc, rtol=0, atol=1e-6)


def test_trafo_impedance():
    net = pp.create_empty_network(sn_mva=0.16)
    pp.create_bus(net, 20)
    pp.create_buses(net, 2, 0.4)
    pp.create_ext_grid(net, 0, s_sc_max_mva=346.4102, rx_max=0.1)
    v_lv = 410
    pp.create_transformer_from_parameters(net, 0, 1, 0.4, 20, v_lv / 1e3, 1.15, 4, 0, 0)
    pp.create_line_from_parameters(net, 1, 2, 0.004, 0.208, 0.068, 0, 1, parallel=2)
    # pp.create_load(net, 2, 0.1)

    pp.runpp(net)

    sc.calc_sc(net, case='max', lv_tol_percent=6., bus=2, branch_results=True, use_pre_fault_voltage=False)


    # trafo:
    z_tlv = 4 / 100 * v_lv ** 2 / (400 * 1e3)
    r_tlv = 4600 * v_lv ** 2 / ((400 * 1e3) ** 2)
    x_tlv = np.sqrt(z_tlv**2 - r_tlv**2)
    z_tlv = r_tlv + 1j*x_tlv
    x_t = x_tlv * 400*1e3 / (v_lv ** 2)
    k_t = 0.95 * 1.05 / (1+0.6 * x_t)
    z_tk = k_t * z_tlv

    # line:
    z_l = 0.416 * 1e-3 + 1j * 0.136 * 1e-3  # Ohm

    # assert np.allclose(net.res_bus_sc.rk_ohm * 1e3, 5.18, rtol=0, atol=1e-6)
    # assert np.allclose(net.res_bus_sc.xk_ohm * 1e3, 16.37, rtol=0, atol=1e-6)
    assert np.allclose(net._ppc['branch'][:, BR_R].real, [0.416*1e-3, z_tk.real], rtol=0, atol=1e-6)
    assert np.allclose(net._ppc['branch'][:, BR_X].real, [0.136*1e-3, z_tk.imag], rtol=0, atol=1e-6)

    ppci = net.ppci
    tap = ppci["branch"][:, TAP].real
    ikss1 = ppci["bus"][:, IKSS1] * np.exp(1j * np.deg2rad(ppci["bus"][:, PHI_IKSS1_DEGREE]))

    v_1 = ikss1[2] * z_l / 0.4 * np.sqrt(3)  # kA * Ohm / V_base -> p.u.
    np.abs(v_1)
    np.angle(v_1, deg=True)

    v_0 = v_1 + ikss1[2] * z_tk / 0.4 * np.sqrt(3) * 0.4/0.41
    np.abs(v_0)
    np.angle(v_0, deg=True)

    v_0_ref = v_1 + ikss1[2] * z_tk / 0.4 * np.sqrt(3)

    Yf = ppci["internal"]["Yf"]
    Yt = ppci["internal"]["Yt"]
    V_diff = np.ones_like(net.bus.index.values, dtype=np.complex128)
    V_diff[0] = v_0
    V_diff[1] = v_1
    V_diff[2] = 0
    i_f = Yf.dot(V_diff) / ppci["internal"]["baseI"][ppci["branch"][:, F_BUS].real.astype(int)]
    i_t = Yt.dot(V_diff) / ppci["internal"]["baseI"][ppci["branch"][:, T_BUS].real.astype(int)]
    abs(i_f)
    abs(i_t)


def test_one_line():
    net = pp.create_empty_network(sn_mva=1)
    b1 = pp.create_bus(net, vn_kv=10.)
    b2 = pp.create_bus(net, vn_kv=10.)
    pp.create_ext_grid(net, b1, s_sc_max_mva=100., s_sc_min_mva=40., rx_min=0.1, rx_max=0.1)
    pp.create_line_from_parameters(net, b1, b2, 1, 0.099, 0.156, 400, 0.457)
    pp.create_load(net, b2, 20)
    pp.runpp(net)

    bus_idx = 1
    sc.calc_sc(net, case='max', branch_results=True, bus=bus_idx, use_pre_fault_voltage=True)

    ppci = net.ppci
    bus = ppci["bus"]
    branch = ppci["branch"]
    Ybus = ppci["internal"]["Ybus"]
    Yf = ppci["internal"]["Yf"]
    Yt = ppci["internal"]["Yt"]
    Zbus = ppci["internal"]["Zbus"]
    baseI = ppci["internal"]["baseI"]

    ikss1 = ppci["bus"][:, IKSS1] * np.exp(1j * np.deg2rad(ppci["bus"][:, PHI_IKSS1_DEGREE]))

    assert np.allclose(Zbus, inv(Ybus.toarray()))

    V_ikss = (ikss1 * baseI * Zbus)[:, bus_idx]
    Ibus = Ybus * V_ikss

    V_ikss_t = 1.1 - V_ikss
    Ibus_t = Ybus * V_ikss_t

    Ibus[abs(Ibus) < 1e-10] = np.nan

    assert np.allclose(ikss1 * baseI, Ibus, equal_nan=True)

    Sbus = V_ikss * Ibus.conj()
    abs(Sbus)

    Sbus_t = V_ikss_t * Ibus_t.conj()
    abs(Sbus_t)

    Zbus.real

    Z_line = 0.00099+0.00156j
    U_0 = ikss1[bus_idx] * baseI[bus_idx] * Z_line

    Yf @ np.array([U_0, 0])

    pp.runpp(net)
    ppci = net._ppc
    ppci["bus"][0, GS] = 100
    ppci["bus"][0, BS] = 100
    ppci["bus"][1, GS] = 14.27472390373151
    ppci["bus"][1, BS] = 85.57762474778227
    Ybus, Yf, Yt = makeYbus(ppci["baseMVA"], ppci["bus"], ppci["branch"])
    # Ybus = net._ppc["internal"]["Ybus"]
    # Yf = net._ppc["internal"]["Yf"]
    # Yt = net._ppc["internal"]["Yt"]

    Zbus = inv(Ybus.toarray())
    V_ikss_corr = (ikss1 * baseI * Zbus)[:, 1]











def test_branch_all_currents_trafo():
    net = net_transformer()
    sc.calc_sc(net, case='max', ip=True, ith=True, lv_tol_percent=10., branch_results=True, return_all_currents=True)

    assert (abs(net.res_trafo_sc.ikss_lv_ka.loc[(0,0)] - 0.) <1e-5)
    assert (abs(net.res_trafo_sc.ikss_lv_ka.loc[(0,1)] - 0.) <1e-5)
    assert (abs(net.res_trafo_sc.ikss_lv_ka.loc[(0,2)] - 16.992258758) <1e-5)

    assert (abs(net.res_trafo_sc.ikss_hv_ka.loc[(0,0)] - 0.) <1e-5)
    assert (abs(net.res_trafo_sc.ikss_hv_ka.loc[(0,1)] - 0.) <1e-5)
    assert (abs(net.res_trafo_sc.ikss_hv_ka.loc[(0,2)] - 0.648795) <1e-5)


def test_against_single_sc_results_line():
    net = three_bus_permuted_index()

    sc.calc_sc(net, case="max", branch_results=True, return_all_currents=True)
    multi_results = net.res_line_sc.copy()

    for bus in net.bus.index:
        sc.calc_sc(net, bus=bus, case="max", branch_results=True, return_all_currents=True)
        line_bus_indices = [(line, bus) for line in net.line.index]
        single_result = net.res_line_sc.ikss_ka.values
        multi_result = multi_results.ikss_ka.loc[line_bus_indices].values
        assert np.allclose(single_result, multi_result)


def test_against_single_sc_results_trafo():
    net = net_transformer()
    sc.calc_sc(net, case="max", branch_results=True, return_all_currents=True, inverse_y=False)
    multi_results = net.res_trafo_sc.copy()

    for bus in net.bus.index[net.bus.in_service]:
        sc.calc_sc(net, bus=bus, case="max", branch_results=True, return_all_currents=True, inverse_y=False)
        trafo_bus_indices = [(trafo, bus) for trafo in net.trafo.index]
        single_result_lv = net.res_trafo_sc.ikss_lv_ka.values
        multi_result_lv = multi_results.ikss_lv_ka.loc[trafo_bus_indices].values
        assert np.allclose(single_result_lv, multi_result_lv)

        single_result_hv = net.res_trafo_sc.ikss_hv_ka.values
        multi_result_hv = multi_results.ikss_hv_ka.loc[trafo_bus_indices].values
        assert np.allclose(single_result_hv, multi_result_hv)


def test_ward():
    net = pp.create_empty_network(sn_mva=9)
    pp.create_buses(net, 2, 110)
    pp.create_ext_grid(net, 0, s_sc_max_mva=100, rx_max=0.1)
    pp.create_line_from_parameters(net, 0, 1, 1, 0.5, 0.5, 0, 1000)
    pp.create_ward(net, 1, 10, 5, 200, 100)
    sc.calc_sc(net)
    ikss_ka = [1.209707, 1.209818]
    rk_ohm = [57.719840, 57.678686]
    xk_ohm = [-1.834709, -2.740132]
    assert np.allclose(net.res_bus_sc.ikss_ka, ikss_ka, atol=1e-6, rtol=0)
    assert np.allclose(net.res_bus_sc.rk_ohm, rk_ohm, atol=1e-6, rtol=0)
    assert np.allclose(net.res_bus_sc.xk_ohm, xk_ohm, atol=1e-6, rtol=0)


def test_xward():
    net = pp.create_empty_network(sn_mva=4)
    pp.create_buses(net, 2, 110)
    pp.create_ext_grid(net, 0, s_sc_max_mva=100, rx_max=0.1)
    pp.create_line_from_parameters(net, 0, 1, 1, 0.5, 0.5, 0, 1000)
    pp.create_xward(net, 1, 10, 5, 200, 100, 3, 1, vm_pu=1.02)
    sc.calc_sc(net)
    ikss_ka = [1.209707, 1.209818]
    rk_ohm = [57.719840, 57.678686]
    xk_ohm = [-1.834709, -2.740132]
    assert np.allclose(net.res_bus_sc.ikss_ka, ikss_ka, atol=1e-6, rtol=0)
    assert np.allclose(net.res_bus_sc.rk_ohm, rk_ohm, atol=1e-6, rtol=0)
    assert np.allclose(net.res_bus_sc.xk_ohm, xk_ohm, atol=1e-6, rtol=0)


if __name__ == '__main__':
    pytest.main([__file__])

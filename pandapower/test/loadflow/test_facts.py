# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest
import numpy as np
import pandapower as pp

import matplotlib.pyplot as plt

import pandapower.networks
from pandapower.pf.create_jacobian_facts import calc_y_svc, calc_y_svc_pu, calc_tcsc_p_pu
from pandapower.pypower.idx_bus import BS, SVC_THYRISTOR_FIRING_ANGLE


def facts_case_study_grid():
    net = pp.create_empty_network()

    b1 = pp.create_bus(net, name="B1", vn_kv=18)
    b2 = pp.create_bus(net, name="B2", vn_kv=16.5)
    b3 = pp.create_bus(net, name="B3", vn_kv=230)
    b4 = pp.create_bus(net, name="B4", vn_kv=230)
    b5 =pp.create_bus(net, name="B5", vn_kv=230)
    b6 =pp.create_bus(net, name="B6", vn_kv=230)
    b7 =pp.create_bus(net, name="B7", vn_kv=230)
    b8 =pp.create_bus(net, name="B8", vn_kv=230)

    pp.create_ext_grid(net,bus=b1,vm_pu=1,va_degree=0)

    pp.create_line_from_parameters(net,name="L1",from_bus=b3,to_bus=b4,length_km=30, r_ohm_per_km=0.049, x_ohm_per_km=0.136,g_us_per_km=0,c_nf_per_km=142,max_i_ka=1.5)
    pp.create_line_from_parameters(net,name="L2",from_bus=b3,to_bus=b4,length_km=30, r_ohm_per_km=0.049, x_ohm_per_km=0.136,g_us_per_km=0,c_nf_per_km=142,max_i_ka=1.5)
    pp.create_line_from_parameters(net,name="L3",from_bus=b4,to_bus=b5,length_km=100, r_ohm_per_km=0.081, x_ohm_per_km=0.312,g_us_per_km=0,c_nf_per_km=11,max_i_ka=1.5)
    pp.create_line_from_parameters(net,name="L4",from_bus=b4,to_bus=b6,length_km=100, r_ohm_per_km=0.081, x_ohm_per_km=0.312,g_us_per_km=0,c_nf_per_km=11,max_i_ka=1.5)
    pp.create_line_from_parameters(net,name="L5",from_bus=b5,to_bus=b7,length_km=220, r_ohm_per_km=0.081, x_ohm_per_km=0.312,g_us_per_km=0,c_nf_per_km=11,max_i_ka=1.5)
    pp.create_line_from_parameters(net,name="L6",from_bus=b6,to_bus=b8,length_km=140, r_ohm_per_km=0.081, x_ohm_per_km=0.312,g_us_per_km=0,c_nf_per_km=11,max_i_ka=1.5)
    pp.create_line_from_parameters(net,name="L7",from_bus=b5,to_bus=b6,length_km=180, r_ohm_per_km=0.081, x_ohm_per_km=0.312,g_us_per_km=0,c_nf_per_km=11,max_i_ka=1.5)
    pp.create_line_from_parameters(net,name="L8",from_bus=b7,to_bus=b8,length_km=180, r_ohm_per_km=0.081, x_ohm_per_km=0.312,g_us_per_km=0,c_nf_per_km=11,max_i_ka=1.5)

   # pp.create_line_from_parameters(net,name="L9",from_bus=3,to_bus=4,length_km=100, r_ohm_per_km=0.312, x_ohm_per_km=0.312,g_us_per_km=0,c_nf_per_km=11)

    pp.create_transformer_from_parameters(net, name="trafo1",hv_bus=b8,lv_bus=b1,sn_mva=192,vn_hv_kv=230,vn_lv_kv=18,vkr_percent=0,vector_group="Yy0",pfe_kw=0,vk_percent=12,i0_percent=0)
    pp.create_transformer_from_parameters(net, name="trafo2",hv_bus=b3,lv_bus=b2,sn_mva=500,vn_hv_kv=230,vn_lv_kv=16.5,vkr_percent=0,vector_group="Yy0",pfe_kw=0,vk_percent=16,i0_percent=0)

    pp.create_gen(net,bus=b2,p_mw=500,vm_pu=1)
    # pp.create_sgen(net,bus = 2, p_mw=500,name="WT")
    #
    pp.create_load(net,bus=b4,p_mw=130,q_mvar=50)
    pp.create_load(net,bus=b5,p_mw=120,q_mvar=50)
    pp.create_load(net,bus=b6,p_mw=80,q_mvar=25)
    pp.create_load(net,bus=b7,p_mw=50,q_mvar=25)

    # pp.create_load(net,bus=4,p_mw=0,q_mvar=0)
    # pp.create_load(net,bus=5,p_mw=0,q_mvar=0)
    # pp.create_load(net,bus=6,p_mw=0,q_mvar=0)
    # pp.create_load(net,bus=7,p_mw=0,q_mvar=0)

    return net
import matplotlib.pyplot

@pytest.mark.parametrize("vm_set_pu", [0.96, 1., 1.04])
def test_svc(vm_set_pu):
    net = pp.networks.case9()
    net3 = net.deepcopy()
    lidx = pp.create_load(net3, 3, 0, 0)
    pp.create_shunt(net, 3, 0, 0, 345)
    net2 = net.deepcopy()
    net.shunt["controllable"] = True
    net.shunt["set_vm_pu"] = vm_set_pu
    net.shunt["thyristor_firing_angle_degree"] = 90.
    net.shunt["svc_x_l_ohm"] = 1
    net.shunt["svc_x_cvar_ohm"] = -10
    pp.runpp(net)
    assert 90 <= net.shunt.at[0, "thyristor_firing_angle_degree"] <= 180
    assert np.isclose(net.res_bus.at[3, 'vm_pu'], net.shunt.at[0, 'set_vm_pu'], rtol=0, atol=1e-6)

    net3.load.loc[lidx, "q_mvar"] = net.res_shunt.q_mvar.at[0]
    pp.runpp(net3)

    net2.shunt.q_mvar.at[0] = -net._ppc["bus"][net._pd2ppc_lookups["bus"][net.shunt.bus.values], BS]
    pp.runpp(net2)
    assert np.isclose(net2.res_bus.at[3, 'vm_pu'], net.shunt.at[0, 'set_vm_pu'], rtol=0, atol=1e-6)
    assert np.isclose(net2.res_bus.at[3, 'q_mvar'], net.res_bus.at[3, 'q_mvar'], rtol=0, atol=1e-6)
    assert np.isclose(net2.res_shunt.at[0, 'vm_pu'], net.res_shunt.at[0, 'vm_pu'], rtol=0, atol=1e-6)
    assert np.isclose(net2.res_shunt.at[0, 'q_mvar'], net.res_shunt.at[0, 'q_mvar'], rtol=0, atol=1e-6)

    pp.runpp(net)
    assert np.allclose(net.shunt.q_mvar, -net._ppc["bus"][net._pd2ppc_lookups["bus"][net.shunt.bus.values], BS],
                       rtol=0, atol=1e-6)
    assert np.allclose(np.deg2rad(net.shunt.thyristor_firing_angle_degree),
                       net._ppc["bus"][net._pd2ppc_lookups["bus"][net.shunt.bus.values], SVC_THYRISTOR_FIRING_ANGLE],
                       rtol=0, atol=1e-6)

    net.shunt.controllable = False
    pp.runpp(net)
    assert np.isclose(net.res_bus.at[3, 'vm_pu'], net.shunt.at[0, 'set_vm_pu'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[3, 'q_mvar'], net2.res_bus.at[3, 'q_mvar'], rtol=0, atol=1e-5)
    assert np.isclose(net.res_shunt.at[0, 'vm_pu'], net2.res_shunt.at[0, 'vm_pu'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_shunt.at[0, 'q_mvar'], net2.res_shunt.at[0, 'q_mvar'], rtol=0, atol=1e-5)


def test_tcsc_case5():
    net = pp.networks.case5()
    pp.replace_line_by_impedance(net, [2, 3], 50, False)
    net.impedance['controllable'] = True
    net.impedance['set_p_to_mw'] = 150, 50
    net.impedance["thyristor_firing_angle_degree"] = 90.
    net.impedance["tcsc_x_l_ohm"] = 1
    net.impedance["tcsc_x_cvar_ohm"] = -10
    pp.runpp(net)


def test_tcsc_simple():
    net = pp.create_empty_network()
    pp.create_buses(net, 2, 110)
    pp.create_ext_grid(net, 0)
    pp.create_impedance(net, 0, 1, 0, 0.001, 1)
    pp.create_line_from_parameters(net, 0, 1, 100, 0.0487, 0.13823, 160, 0.664)
    pp.create_load(net, 1, 100, 25)

    net.impedance['controllable'] = True
    net.impedance['set_p_to_mw'] = 20
    net.impedance["thyristor_firing_angle_degree"] = 130.
    net.impedance["tcsc_x_l_ohm"] = 1
    net.impedance["tcsc_x_cvar_ohm"] = -10

    pp.runpp(net, max_iteration=10)

    net.impedance.controllable = False
    y = calc_y_svc_pu(np.deg2rad(116.09807835), 0.5, -1)
#    net.impedance.rft_pu
    net.impedance.xft_pu = -1/y
    net.impedance.xtf_pu = -1/y
    pp.runpp(net)


def test_tcsc_simple2():
    baseMVA = 100  # MVA
    baseV = 110  # kV
    baseI = baseMVA / (baseV * np.sqrt(3))
    baseZ = baseV ** 2 / baseMVA
    xl = 0.1
    xc = -5

    net = pp.create_empty_network(sn_mva=baseMVA)
    pp.create_buses(net, 3, 110)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_impedance(net, 1, 2, 0.005, 0.01, 1)
    # pp.create_line_from_parameters(net, 1, 2, 100, 0.0487, 0.13823, 160, 0.664)
    pp.create_load(net, 2, 10, 1)

    pp.runpp(net)
    V = net._ppc["internal"]["V"]
    pp.set_user_pf_options(net, init_vm_pu=np.abs(V), init_va_degree=np.angle(V, deg=True))

    net.impedance['controllable'] = True
    net.impedance['set_p_to_mw'] = -2
    net.impedance["thyristor_firing_angle_degree"] = 150
    net.impedance["tcsc_x_l_ohm"] = 100
    net.impedance["tcsc_x_cvar_ohm"] = -500

    pp.runpp(net, max_iteration=10)

    net.impedance.controllable = False
    y = calc_y_svc_pu(np.deg2rad(120), xl / baseZ, xc / baseZ)
    (1 / y) * baseZ
#    net.impedance.rft_pu
    net.impedance.xft_pu = -1/y
    net.impedance.xtf_pu = -1/y
    pp.runpp(net)

    plot_z(baseZ, 0.1, -5)


def test_tcsc_simple8():
    baseMVA = 100  # MVA
    baseV = 110  # kV
    baseI = baseMVA / (baseV * np.sqrt(3))
    baseZ = baseV ** 2 / baseMVA
    xl = 0.2
    xc = -20
    # plot_z(baseZ, xl, xc)

    # (0)-------------(1)-----------------(3)->
    #                  |--(TCSC)--(2)------|

    net = pp.create_empty_network(sn_mva=baseMVA)
    pp.create_buses(net, 4, baseV)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 3, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 2, 3, 20, 0.0487, 0.13823, 160, 0.664)


    # pp.create_switch(net, 1, 2, 'b', closed=False)
    x = 140
    y = calc_y_svc_pu(np.deg2rad(x), xl / baseZ, xc / baseZ)
    # pp.create_impedance(net, 1, 2, 0, 1/y, baseMVA)

    pp.create_load(net, 3, 100, 40)

    # pp.runpp(net)

    # V = net._ppc["internal"]["V"]
    # pp.set_user_pf_options(net, init_vm_pu=np.abs(V)*.9999, init_va_degree=np.angle(V, deg=True))

    # pp.create_tcsc(net, 1, 2, xl, xc, -20, 154.933536, "Test", controllable=False, min_angle_degree=90, max_angle_degree=180)
    pp.create_tcsc(net, 1, 2, xl, xc, 5, 170, "Test", controllable=True, min_angle_degree=90, max_angle_degree=180)
    # net.tcsc.at[0, "thyristor_firing_angle"] = x
    # pp.create_impedance(net, 1, 2, 0, 8.263537 / baseZ, baseMVA)

    pp.runpp(net)

    Ybus = net._ppc["internal"]["Ybus"].toarray()
    Ybus.round(2)
    Ybus_tcsc = makeYbus_tcsc(Ybus, np.deg2rad(x), np.array([xl/baseZ]), np.array([xc/baseZ]), np.array([1]), np.array([2]))
    (Ybus + Ybus_tcsc).round(2)
    J = net._ppc["internal"]["J"].toarray()
    J[:6, :6].round(1)

    # todo:
    #  test by creating an equivalent net.impedance element and check results, compare J
    #  test for pv, pq buses
    #  test with distributed slack
    #  test with multiple tcsc in the grid, with mix of "controllable" True / False
    #  test results by comparing impedance result to formula; p, q, i by comparing to line results; vm, va by comparing to bus results
    #  test with some other grids, e.g. the grid from the source


def plot_z(baseZ, xl, xc):
    x = np.arange(90, 181)
    y = calc_y_svc_pu(np.deg2rad(x), xl / baseZ, xc / baseZ)
    z = (1 / y) * baseZ
    plt.plot(x, z)


def makeYbus_tcsc(Ybus, x_control, tcsc_x_l_pu, tcsc_x_cvar_pu, tcsc_fb, tcsc_tb):
    Ybus_tcsc = np.zeros(Ybus.shape, dtype=np.complex128)
    Y_TCSC = calc_y_svc_pu(x_control, tcsc_x_l_pu, tcsc_x_cvar_pu)
    print("Y_TCSC", np.round(Y_TCSC, 3))
    Y_TCSC_c = -1j * Y_TCSC
    for y_tcsc_pu_i, i, j in zip(Y_TCSC_c, tcsc_fb, tcsc_tb):
        Ybus_tcsc[i, i] += y_tcsc_pu_i
        Ybus_tcsc[i, j] += -y_tcsc_pu_i
        Ybus_tcsc[j, i] += -y_tcsc_pu_i
        Ybus_tcsc[j, j] += y_tcsc_pu_i
    return Ybus_tcsc



def test_tcsc_simple3():
    baseMVA = 100  # MVA
    baseV = 110  # kV
    baseI = baseMVA / (baseV * np.sqrt(3))
    baseZ = baseV ** 2 / baseMVA

    net = pp.create_empty_network()
    pp.create_buses(net, 3, 110)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 100, 0.0487, 0.13823, 160, 0.664)
    pp.create_impedance(net, 1, 2, 0, 999999999, 1)
    pp.create_load(net, 2, 100, 0)

    net.impedance['controllable'] = True
    net.impedance['set_p_to_mw'] = -100
    net.impedance["thyristor_firing_angle_degree"] = 130.
    net.impedance["tcsc_x_l_ohm"] = 1
    net.impedance["tcsc_x_cvar_ohm"] = -10

    pp.runpp(net, max_iteration=10)

#     net.impedance.controllable = False
#     y = calc_y_svc_pu(np.deg2rad(116.09807835), 1, -10)
# #    net.impedance.rft_pu
#     net.impedance.xft_pu = -1/y
#     net.impedance.xtf_pu = -1/y
#     pp.runpp(net)
#

def test_tcsc_simple3():

    net = pp.create_empty_network()
    pp.create_buses(net, 2, 110)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 100, 0.0487, 0.13823, 160, 0.664)
    #pp.create_impedance(net, 1, 2, 0, 0.001, 1)
    pp.create_line_from_parameters(net, 0, 1, 1, 0,  -0.082121 , 160, 0.664)
    pp.create_load(net, 1, 100, 25)

    import pandas as pd
    pd.set_option('display.max_columns', None)

    z_base_ohm = 110**2 / 1
    y = calc_y_svc_pu(np.deg2rad(141), 1 / z_base_ohm, -10 / z_base_ohm)
    # z_old = 1/y
    # z_eq = np.sqrt(np.square(.0487) + np.square(.13823))
    # z_new = (-z_eq * (z_eq + z_old ))/ z_old
    #
    net.line.x_ohm_per_km[net.line.index == 1] = z_base_ohm/y

    pp.runpp(net, max_iteration=100)




def test_calc_tcsc_p_pu():
    net = pp.create_empty_network()
    pp.create_buses(net, 2, 110)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 100, 0.0487, 0.13823, 160, 0.664)
    pp.create_load(net, 1, 100, 25)

    pp.runpp(net)

    V = net._ppc["internal"]["V"]
    tcsc_fb = 0
    tcsc_tb = 1
    Ybus = net._ppc["internal"]["Ybus"]

    z_base_ohm = np.square(110) / 1
    x_pu = net.line.length_km.values * (net.line.r_ohm_per_km.values + net.line.x_ohm_per_km.values * 1j) / z_base_ohm
    y_pu = 1 / x_pu
    S = V * np.conj(Ybus * V)

    abs(V[0]) * abs(y_pu) * abs(V[1]) * np.cos(np.angle(V[1]) - np.angle(V[0]) + np.angle(y_pu))


    p, A, phi = calc_tcsc_p_pu(Ybus, V, tcsc_fb, tcsc_tb)


def test_tcsc_firing_angle_formula():
    net = pp.create_empty_network()
    pp.create_buses(net, 2, 110)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 100, 0.0487, 0.13823, 160, 0.664)
    pp.create_impedance(net, 0, 1, 0, 0.001, 1)
    #pp.create_line_from_parameters(net, 1, 2, 100, 0.0487, 0.13823, 160, 0.664)
    pp.create_load(net, 1, 100, 25)

    z_base_ohm = np.square(110) / 1
    #y_pu = calc_y_svc_pu(np.deg2rad(134.438395), 0.5 / z_base_ohm, -2 / z_base_ohm)
    y_pu = calc_y_svc_pu(np.deg2rad(135.401298), 0.5 / z_base_ohm, -2 / z_base_ohm)
    y_pu = 1/(-18.9/z_base_ohm)
    print((1/y_pu) * z_base_ohm)
    #    net.impedance.rft_pu
    net.impedance.xft_pu = 1 / y_pu
    net.impedance.xtf_pu = 1 / y_pu
    pp.runpp(net)
    print(net.res_line.loc[0])
    print(net.res_impedance.loc[0])


def test_tcsc_firing_angle_formula():
    net = pp.create_empty_network()
    pp.create_buses(net, 2, 110)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 100, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 0, 1, 1, 0, 0.01, 0, 0.664)
    pp.create_load(net, 1, 100, 25)

    z_base_ohm = np.square(110) / 1
    #y_pu = calc_y_svc_pu(np.deg2rad(134.438395), 0.5 / z_base_ohm, -2 / z_base_ohm)
    y_pu = calc_y_svc_pu(np.deg2rad(135.401298), 0.5 / z_base_ohm, -2 / z_base_ohm)
    #y_pu = calc_y_svc_pu(np.deg2rad(135.401298), 0.5 / z_base_ohm, -2 / z_base_ohm)
    print((1/y_pu) * z_base_ohm)
    #    net.impedance.rft_pu
    #net.line.loc[1, "x_ohm_per_km"] = (1 / y_pu) * z_base_ohm
    net.line.loc[1, "x_ohm_per_km"] = -18.9
    pp.runpp(net, max_iteration=100)
    print(net.res_line)


def test_tcsc_firing_angle_formula():
    net = pp.create_empty_network()
    pp.create_buses(net, 2, 110)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 100, 0.0487, 0.13823, 160, 0.664)
    pp.create_load(net, 1, 100, 25)

    z_base_ohm = np.square(110) / 1
    #y_pu = calc_y_svc_pu(np.deg2rad(134.438395), 0.5 / z_base_ohm, -2 / z_base_ohm)
    y_pu = calc_y_svc_pu(np.deg2rad(141), 1 / z_base_ohm, -10 / z_base_ohm)
    #y_pu = calc_y_svc_pu(np.deg2rad(135.401298), 0.5 / z_base_ohm, -2 / z_base_ohm)
    print((1/y_pu) )
    print((1/y_pu) * z_base_ohm)
    print(xtcsc(np.deg2rad(141), 1, -10) / z_base_ohm)
    #    net.impedance.rft_pu
    pp.create_shunt(net, 1, -y_pu, y_pu)
    pp.runpp(net)
    print(net.res_line.loc[0])


def xtcr(x, x_l):
    return np.pi * x_l / (2*(np.pi - x) + np.sin(2*x))

def xtcsc(x, x_l, x_c):
    return np.pi * x_l / (2*(np.pi-x) + np.sin(2*x) + np.pi*x_l/x_c)


def test_tcsc_simple5():
    net = pp.create_empty_network(sn_mva=100)
    pp.create_buses(net, 4, 110)
    pp.create_ext_grid(net, 0)
    pp.create_line_from_parameters(net, 0, 1, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 2, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_line_from_parameters(net, 1, 3, 20, 0.0487, 0.13823, 160, 0.664)
    pp.create_impedance(net, 2, 3, 1e20, 1e20, 1)
    pp.create_load(net, 3, 100, 25)

    net.impedance['controllable'] = True
    net.impedance['set_p_to_mw'] = -20
    net.impedance["thyristor_firing_angle_degree"] = 90
    net.impedance["tcsc_x_l_ohm"] = 10
    net.impedance["tcsc_x_cvar_ohm"] = -100

    pp.runpp(net, max_iteration=200)

    net.impedance.controllable = False
    y = calc_y_svc_pu(np.deg2rad(116.09807835), 1, -10)
#    net.impedance.rft_pu
    net.impedance.xft_pu = -1/y
    net.impedance.xtf_pu = -1/y
    pp.runpp(net)


if __name__ == "__main__":
    pytest.main([__file__])

# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import copy
import os
from itertools import product

import numpy as np
import pytest

from pandapower import pp_dir, create_b2b_vsc, LoadflowNotConverged
from pandapower.converter.powerfactory.validate import validate_pf_conversion
from pandapower.create import create_impedance, create_shunts, create_buses, create_gens, create_bus,  \
    create_empty_network, create_line_from_parameters, create_gen, create_load_dc, \
    create_load, create_ext_grid, create_vsc, create_line_dc_from_parameters, \
    create_buses_dc, create_bus_dc, create_line_dc, create_lines_from_parameters, create_lines_dc

from pandapower.file_io import from_json
from pandapower.pf.makeYbus_facts import calc_y_svc_pu
from pandapower.run import runpp
from pandapower.test.consistency_checks import runpp_with_consistency_checks
from pandapower.test.loadflow.test_facts import compare_ssc_impedance_gen


def plot_z(baseZ, xl, xc):
    import matplotlib.pyplot as plt

    x = np.arange(90, 181)
    y = calc_y_svc_pu(np.deg2rad(x), xl / baseZ, xc / baseZ)
    z = (1 / y) * baseZ
    plt.plot(x, z)


def xtcr(x, x_l):
    return np.pi * x_l / (2 * (np.pi - x) + np.sin(2 * x))


def xtcsc(x, x_l, x_c):
    return np.pi * x_l / (2 * (np.pi - x) + np.sin(2 * x) + np.pi * x_l / x_c)


def copy_with_impedance(net):
    baseMVA = net.sn_mva  # MVA
    baseV = net.bus.vn_kv.values  # kV
    baseI = baseMVA / (baseV * np.sqrt(3))
    baseZ = baseV ** 2 / baseMVA

    net_ref = copy.deepcopy(net)
    for i in net.tcsc.index.values:
        create_impedance(net_ref, net.tcsc.from_bus.at[i], net.tcsc.to_bus.at[i], 0,
                         net.res_tcsc.x_ohm.at[i] / baseZ[net.tcsc.to_bus.at[i]], baseMVA,
                         in_service=net.tcsc.in_service.at[i], name="tcsc")
    net_ref.tcsc.in_service = False

    if len(net.svc) > 0:
        # create_loads(net_ref, net.svc.bus.values, 0, net.res_svc.q_mvar.values, in_service=net.svc.in_service.values)
        # create shunts because of Ybus comparison
        q = np.square(net.bus.loc[net.svc.bus.values, 'vn_kv']) / net.res_svc.x_ohm.values
        create_shunts(net_ref, net.svc.bus.values, q.fillna(0), in_service=net.svc.in_service.values, name="svc")
        net_ref.svc.in_service = False

    if len(net.ssc) > 0:
        # create shunts because of Ybus comparison
        in_service = net.ssc.in_service.values
        ssc_bus = net.ssc.bus.values
        aux_bus = create_buses(net_ref, len(net.ssc), net.bus.loc[ssc_bus, "vn_kv"].values)
        for fb, tb, r, x, i in zip(ssc_bus, aux_bus, net.ssc.r_ohm.values / baseZ[ssc_bus],
                                   net.ssc.x_ohm.values / baseZ[ssc_bus], in_service):
            create_impedance(net_ref, fb, tb, r, x, baseMVA, name="ssc", in_service=i)
        if len(net.res_ssc) > 0:
            vm_pu = net.res_ssc.vm_internal_pu.fillna(1)
        else:
            vm_pu = net.ssc.set_vm_pu.fillna(1)
        create_gens(net_ref, aux_bus, 0, vm_pu, in_service=in_service)
        net_ref.ssc.in_service = False

    if len(net.vsc) > 0:
        # create shunts because of Ybus comparison
        in_service = net.vsc.in_service.values
        vsc_bus = net.vsc.bus.values
        aux_bus = create_buses(net_ref, len(net.vsc), net.bus.loc[vsc_bus, "vn_kv"].values)
        for fb, tb, r, x, i in zip(vsc_bus, aux_bus, net.vsc.r_ohm.values / baseZ[vsc_bus],
                                   net.vsc.x_ohm.values / baseZ[vsc_bus], in_service):
            create_impedance(net_ref, fb, tb, r, x, baseMVA, name="vsc", in_service=i)
        g = create_gens(net_ref, aux_bus, 0, net.res_vsc.vm_internal_pu.fillna(1), in_service=in_service)
        vsc_pv = net.vsc.control_mode_ac.values == "vm_pu"
        net_ref.gen.loc[g[vsc_pv], "vm_pu"] = net.vsc.loc[vsc_pv, "control_value_ac"].values
        net_ref.vsc.in_service = False
        net_ref.bus_dc.loc[net.vsc.bus_dc.values, "in_service"] = False

    return net_ref


#
# def test_tcsc_firing_angle_formula():
#     net = create_empty_network()
#     create_buses(net, 2, 110)
#     create_ext_grid(net, 0)
#     create_line_from_parameters(net, 0, 1, 100, 0.0487, 0.13823, 160, 0.664)
#     create_impedance(net, 0, 1, 0, 0.001, 1)
#     #create_line_from_parameters(net, 1, 2, 100, 0.0487, 0.13823, 160, 0.664)
#     create_load(net, 1, 100, 25)
#
#     z_base_ohm = np.square(110) / 1
#     #y_pu = calc_y_svc_pu(np.deg2rad(134.438395), 0.5 / z_base_ohm, -2 / z_base_ohm)
#      y_pu = calc_y_svc_pu(np.deg2rad(135.401298), 0.5 / z_base_ohm, -2 / z_base_ohm)
#     y_pu = 1/(-18.9/z_base_ohm)
#     print((1/y_pu) * z_base_ohm)
#     #    net.impedance.rft_pu
#     net.impedance.xft_pu = 1 / y_pu
#     net.impedance.xtf_pu = 1 / y_pu
#     runpp_with_consistency_checks(net)
#     print(net.res_line.loc[0])
#     print(net.res_impedance.loc[0])

#
# def test_tcsc_firing_angle_formula():
#     net = create_empty_network()
#     create_buses(net, 2, 110)
#     create_ext_grid(net, 0)
#     create_line_from_parameters(net, 0, 1, 100, 0.0487, 0.13823, 160, 0.664)
#     create_line_from_parameters(net, 0, 1, 1, 0, 0.01, 0, 0.664)
#     create_load(net, 1, 100, 25)
#
#     z_base_ohm = np.square(110) / 1
#     #y_pu = calc_y_svc_pu(np.deg2rad(134.438395), 0.5 / z_base_ohm, -2 / z_base_ohm)
#     y_pu = calc_y_svc_pu(np.deg2rad(135.401298), 0.5 / z_base_ohm, -2 / z_base_ohm)
#     #y_pu = calc_y_svc_pu(np.deg2rad(135.401298), 0.5 / z_base_ohm, -2 / z_base_ohm)
#     print((1/y_pu) * z_base_ohm)
#     #    net.impedance.rft_pu
#     #net.line.loc[1, "x_ohm_per_km"] = (1 / y_pu) * z_base_ohm
#     net.line.loc[1, "x_ohm_per_km"] = -18.9
#     runpp(net, max_iteration=100)
#     print(net.res_line)
#
#
# def test_tcsc_firing_angle_formula():
#     net = create_empty_network()
#     create_buses(net, 2, 110)
#     create_ext_grid(net, 0)
#     create_line_from_parameters(net, 0, 1, 100, 0.0487, 0.13823, 160, 0.664)
#     create_load(net, 1, 100, 25)
#
#     z_base_ohm = np.square(110) / 1
#     #y_pu = calc_y_svc_pu(np.deg2rad(134.438395), 0.5 / z_base_ohm, -2 / z_base_ohm)
#     y_pu = calc_y_svc_pu(np.deg2rad(141), 1 / z_base_ohm, -10 / z_base_ohm)
#     #y_pu = calc_y_svc_pu(np.deg2rad(135.401298), 0.5 / z_base_ohm, -2 / z_base_ohm)
#     print((1/y_pu) )
#     print((1/y_pu) * z_base_ohm)
#     print(xtcsc(np.deg2rad(141), 1, -10) / z_base_ohm)
#     #    net.impedance.rft_pu
#     create_shunt(net, 1, -y_pu, y_pu)
#     runpp_with_consistency_checks(net)
#     print(net.res_line.loc[0])


def test_vsc_hvdc():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110, geodata=[(0, 0), (100, 0), (200, 0)])
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 2, 10, 5)

    # DC part
    create_bus_dc(net, 110, 'A', geodata=(100, 10))
    create_bus_dc(net, 110, 'B', geodata=(200, 10))

    create_line_dc_from_parameters(net, 0, 1, 100, 0.1, 1)

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1.,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 1, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1.,
               control_mode_dc="p_mw", control_value_dc=5)

    runpp(net)
    runpp_with_consistency_checks(net)


def test_vsc_hvdc_control_q():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 2, 10, 5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_line_dc_from_parameters(net, 0, 1, 100, 0.1, 1)

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="q_mvar", control_value_ac=7.5,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 1, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1.,
               control_mode_dc="p_mw", control_value_dc=5)

    runpp_with_consistency_checks(net)


def test_vsc_multiterminal_hvdc():
    net = create_empty_network()
    # AC part
    create_buses(net, 5, 110, geodata=[(0, 50), (50, 100), (200, 100), (50, 0), (200, 0)])
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 0, 3, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 3, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 3, 4, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 2, 10, 5)

    # DC part
    create_bus_dc(net, 320, 'A', geodata=(50, 100))
    create_bus_dc(net, 320, 'B', geodata=(200, 50))
    create_bus_dc(net, 320, 'C', geodata=(200, 100))
    create_bus_dc(net, 320, 'D', geodata=(200, 0))
    create_bus_dc(net, 320, 'E', geodata=(50, 0))

    create_line_dc_from_parameters(net, 0, 1, 100, 0.1, 1)
    create_line_dc_from_parameters(net, 1, 2, 100, 0.1, 1)
    create_line_dc_from_parameters(net, 1, 3, 100, 0.1, 1)
    create_line_dc_from_parameters(net, 1, 4, 100, 0.1, 1)

    create_vsc(net, 1, 0, 0.1, 5,0.15, control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 2, 0.1, 5,0.15, control_value_dc=5)
    create_vsc(net, 4, 3, 0.1, 5,0.15, control_value_dc=15)
    create_vsc(net, 3, 4, 0.1, 5,0.15, control_mode_dc="vm_pu", control_value_dc=1.02)
    """
    create_vsc(net, 1, 0, 0.1, 5, 0.15,
            control_mode_ac="q_mvar", control_value_ac=3,
            control_mode_dc="vm_pu", control_value_dc=1.)
    create_vsc(net, 2, 2, 0.1, 5, 0.15,
            control_mode_ac="q_mvar", control_value_ac=10,
            control_mode_dc="p_mw", control_value_dc=5)
    create_vsc(net, 4, 3, 0.1, 5, 0.15,
            control_mode_ac="vm_pu", control_value_ac=1.05,
            control_mode_dc="p_mw", control_value_dc=15)
    create_vsc(net, 3, 4, 0.1, 5, 0.15,
            control_mode_ac="vm_pu", control_value_ac=1.03,
            control_mode_dc="vm_pu", control_value_dc=1.02)
    """
    runpp_with_consistency_checks(net, max_iteration=1000)


def test_line_dc_bus_dc_structures():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)

    create_ext_grid(net, 0)
    create_load(net, 2, 10, 5)

    runpp(net)

    # DC part
    ## bus structure 1
    # create_bus_dc(net, 110, 'A')
    # create_bus_dc(net, 110, 'B')
    ## bus structure 2
    create_buses_dc(net, 2, 110)

    ## line structure 1
    # create_line_dc_from_parameters(net, 0, 1, 100, 0.1, 1)

    ## line structure 2
    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1.,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 1, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1.,
               control_mode_dc="p_mw", control_value_dc=5)

    runpp_with_consistency_checks(net)


def test_line_dc_bus_dc_structures2():
    net = create_empty_network()
    # AC part
    create_buses(net, 5, 110, geodata=[(0, 50), (50, 100), (200, 100), (50, 0), (200, 0)])
    create_lines_from_parameters(net, [0, 1, 0, 1, 3], [1, 2, 3, 3, 4], 30, 0.0487, 0.13823, 160, 0.664)

    create_ext_grid(net, 0)
    create_load(net, 2, 10, 5)

    # DC part
    create_bus_dc(net, 320, 'A', geodata=(50, 100))
    create_bus_dc(net, 320, 'B', geodata=(200, 50))
    create_bus_dc(net, 320, 'C', geodata=(200, 100))
    create_bus_dc(net, 320, 'D', geodata=(200, 0))
    create_bus_dc(net, 320, 'E', geodata=(50, 0))

    ## line structure 3
    # create_lines_dc_from_parameters(net,[0,1,1,1],[1,2,3,4],100,0.01,1)
    # create_lines_from_parameters()
    ## line structure 4
    create_lines_dc(net, [0, 1, 1, 1], [1, 2, 3, 4], 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1.,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 2, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1.,
               control_mode_dc="p_mw", control_value_dc=5)
    create_vsc(net, 4, 3, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1.,
               control_mode_dc="p_mw", control_value_dc=15)
    create_vsc(net, 3, 4, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1.,
               control_mode_dc="vm_pu", control_value_dc=1.02)

    runpp_with_consistency_checks(net)


def test_vsc_hvdc_structure1():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 2, 10, 5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 2, 1, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=5)
    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)

    runpp_with_consistency_checks(net)


def test_vsc_hvdc_structure1_alternate():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 2, 10, 5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=5)
    create_vsc(net, 2, 1, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)

    runpp_with_consistency_checks(net)


def test_setting_of_dc_out_of_service():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 2, 10, 5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')
    create_bus_dc(net, 110, 'T', in_service=False)  # todo results for this dc bus must be NaN

    create_vsc(net, 1, 0, 0, 5, 0.15, control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 1, 0, 5, 0.15, control_value_dc=5)
    create_vsc(net, 2, 2, 0, 5, 0.15, control_value_dc=5)

    runpp_with_consistency_checks(net)  ## does the not in_service dc bus set the vsc out of service?


def test_setting_of_dc_vsc_out_of_service():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 2, 10, 5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')
    create_bus_dc(net, 110, 'T', in_service=False)  # todo results for this dc bus must be NaN

    create_vsc(net, 1, 0, 0, 5, 0.15, control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 1, 0, 5, 0.15, control_value_dc=5)
    create_vsc(net, 2, 2, 0, 5, 0.15, control_value_dc=5, in_service=False)

    runpp_with_consistency_checks(net)  ## does the not in_service dc bus set the vsc out of service?


def test_vsc_hvdc_structure2():
    net = create_empty_network()
    # AC part
    create_buses(net, 5, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 2, 3, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 3, 4, 30, 0.0487, 0.13823, 160, 0.664)

    create_ext_grid(net, 0)
    create_load(net, 2, 10, 3)
    create_load(net, 4, 10, 4)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")
    create_line_dc(net, 1, 2, 100, std_type="2400-CU")
    create_line_dc(net, 2, 3, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 1, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=5)

    create_vsc(net, 3, 2, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 4, 3, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=5)

    runpp_with_consistency_checks(net)


def test_vsc_hvdc_mode0_without_dc_line():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    create_ext_grid(net, 0)
    create_load(net, 2, 10, 5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1)
    create_vsc(net, 2, 1, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1.02,
               control_mode_dc="vm_pu", control_value_dc=1.02)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'vm_pu'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'vm_pu'], net.vsc.at[1, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_dc_line():
    net = create_empty_network()
    # AC part
    create_buses(net, 2, 110)

    create_ext_grid(net, 0)
    create_load(net, 1, 10, 5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 0, 0, 0.1, 5, 0.15,
               control_mode_ac="q_mvar", control_value_ac=0,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 1, 1, 0.1, 5, 0.15,
               control_mode_ac="slack", control_value_ac=1.02,
               control_mode_dc="p_mw", control_value_dc=0)

    runpp_with_consistency_checks(net)


def test_vsc_hvdc_mode1():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    create_ext_grid(net, 0)
    create_load(net, 2, 10, 5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15, control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 1, 0.1, 5, 0.15, control_mode_ac="vm_pu", control_value_ac=1.02,
               control_mode_dc="vm_pu", control_value_dc=1)

    runpp(net)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'vm_pu'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'vm_pu'], net.vsc.at[1, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode2():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    create_ext_grid(net, 0)
    create_load(net, 2, 10, 5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=5)
    create_vsc(net, 2, 1, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1.02,
               control_mode_dc="vm_pu", control_value_dc=1)

    # runpp(net)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'vm_pu'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'vm_pu'], net.vsc.at[1, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'p_mw'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode2_without_dc_line():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    create_ext_grid(net, 0)
    create_load(net, 2, 10, 5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_vsc(net, 1, 0, 0.1, 5, 0.15, control_mode_ac="vm_pu", control_value_ac=1, control_mode_dc="p_mw",
               control_value_dc=5)
    create_vsc(net, 2, 1, 0.1, 5, 0.15, control_mode_ac="vm_pu", control_value_ac=1.02, control_mode_dc="vm_pu",
               control_value_dc=1)

    runpp_with_consistency_checks(net)

    # todo one of the vsc is out of service, so only 1 dc bus will have correct result

    # assert np.isclose(net.res_bus.at[1, 'vm_pu'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'vm_pu'], net.vsc.at[1, 'control_value_ac'], rtol=0, atol=1e-6)
    # assert np.isclose(net.res_bus_dc.at[0, 'p_mw'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode3():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    create_ext_grid(net, 0)
    create_load(net, 2, 10, 5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1)
    create_vsc(net, 2, 1, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1.02,
               control_mode_dc="p_mw", control_value_dc=5)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'vm_pu'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'vm_pu'], net.vsc.at[1, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'p_mw'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode4():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    create_ext_grid(net, 0)
    create_load(net, 2, 10, 5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15, control_mode_ac="vm_pu", control_value_ac=1, control_mode_dc="p_mw",
               control_value_dc=5)
    create_vsc(net, 2, 1, 0.1, 5, 0.15, control_mode_ac="vm_pu", control_value_ac=1.02, control_mode_dc="p_mw",
               control_value_dc=5)

    # all DC buses are set out of service, so there is no error in this case and bus_dc results are NaN
    # with pytest.raises(UserWarning, match="reference bus for the dc grid"):
    #     runpp(net)
    runpp_with_consistency_checks(net)


def test_vsc_hvdc_mode5():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    create_ext_grid(net, 0)
    create_load(net, 2, 10, 5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=4)
    create_vsc(net, 2, 1, 0.1, 5, 0.15,
               control_mode_ac="q_mvar", control_value_ac=3,
               control_mode_dc="vm_pu", control_value_dc=1)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'vm_pu'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'q_mvar'], net.vsc.at[1, 'control_value_ac'] + net.load.at[0, "q_mvar"], rtol=0,
                      atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'p_mw'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode6():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    create_ext_grid(net, 0)
    create_load(net, 2, 10, 5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 1, 0.1, 5, 0.15,
               control_mode_ac="q_mvar", control_value_ac=3,
               control_mode_dc="vm_pu", control_value_dc=1)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'vm_pu'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'q_mvar'], net.vsc.at[1, 'control_value_ac'] + net.load.at[0, "q_mvar"], rtol=0,
                      atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode7():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    create_ext_grid(net, 0)
    create_load(net, 2, 10, 5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1)
    create_vsc(net, 2, 1, 0.1, 5, 0.15,
               control_mode_ac="q_mvar", control_value_ac=3,
               control_mode_dc="p_mw", control_value_dc=4)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'vm_pu'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'q_mvar'], net.vsc.at[1, 'control_value_ac'] + net.load.at[0, "q_mvar"], rtol=0,
                      atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'p_mw'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode9():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    create_ext_grid(net, 0)
    create_load(net, 2, 10, 4)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="q_mvar", control_value_ac=5,
               control_mode_dc="p_mw", control_value_dc=4)
    create_vsc(net, 2, 1, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'q_mvar'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'vm_pu'], net.vsc.at[1, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'p_mw'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode10():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    create_ext_grid(net, 0)
    create_load(net, 2, 10, 5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="q_mvar", control_value_ac=3,
               control_mode_dc="vm_pu", control_value_dc=1)
    create_vsc(net, 2, 1, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'q_mvar'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'vm_pu'], net.vsc.at[1, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode11():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    create_ext_grid(net, 0)
    create_load(net, 2, 10, 5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="q_mvar", control_value_ac=3,
               control_mode_dc="vm_pu", control_value_dc=1)
    create_vsc(net, 2, 1, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=4)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'q_mvar'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'vm_pu'], net.vsc.at[1, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'p_mw'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode13():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    create_ext_grid(net, 0)
    create_load(net, 2, 10, 4)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="q_mvar", control_value_ac=5,
               control_mode_dc="p_mw", control_value_dc=4)
    create_vsc(net, 2, 1, 0.1, 5, 0.15,
               control_mode_ac="q_mvar", control_value_ac=5,
               control_mode_dc="vm_pu", control_value_dc=1)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'q_mvar'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'q_mvar'], net.vsc.at[1, 'control_value_ac'] + net.load.at[0, "q_mvar"], rtol=0,
                      atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'p_mw'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode14():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    create_ext_grid(net, 0)
    create_load(net, 2, 10, 5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15, control_mode_ac="q_mvar", control_value_ac=3, control_mode_dc="vm_pu",
               control_value_dc=1)
    create_vsc(net, 2, 1, 0.1, 5, 0.15, control_mode_ac="q_mvar", control_value_ac=3, control_mode_dc="vm_pu",
               control_value_dc=1)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'q_mvar'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'q_mvar'], net.vsc.at[1, 'control_value_ac'] + net.load.at[0, "q_mvar"], rtol=0,
                      atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_vsc_hvdc_mode15():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)

    create_ext_grid(net, 0)
    create_load(net, 2, 10, 5)

    # DC part
    create_bus_dc(net, 110, 'A', index=1)
    create_bus_dc(net, 110, 'B', index=0)

    create_line_dc(net, 0, 1, 100, std_type="2400-CU", index=6)

    create_vsc(net, 1, 1, 0.1, 5, 0.15,
               control_mode_ac="q_mvar", control_value_ac=3,
               control_mode_dc="vm_pu", control_value_dc=1, index=3)
    create_vsc(net, 2, 0, 0.1, 5, 0.15,
               control_mode_ac="q_mvar", control_value_ac=4,
               control_mode_dc="p_mw", control_value_dc=4, index=2)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[1, 'q_mvar'], net.vsc.at[3, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus.at[2, 'q_mvar'], net.vsc.at[2, 'control_value_ac'] + net.load.at[0, "q_mvar"], rtol=0,
                      atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[3, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'p_mw'], net.vsc.at[2, 'control_value_dc'], rtol=0, atol=1e-6)


# TODO : indexing tests for VSC and bus_dc

@pytest.mark.xfail
def test_minimal_ac():
    """
    This test is not documented properly and it is not clear why it fails.
    It checks, if two slacks with different setpoints are connected together, but then impedances are checked and
    for some reason after the copy it fails.
    """
    net = create_empty_network()
    # AC part
    create_bus(net, 110)
    create_ext_grid(net, 0, vm_pu=1.02)

    # DC part
    create_bus_dc(net, 110, 'A')

    create_vsc(net, 0, 0, 0.1, 6, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1.01,
               control_mode_dc="vm_pu", control_value_dc=1.02)

    with pytest.raises(UserWarning, match="Voltage controlling elements"):
        runpp(net)

    net.vsc.at[0, "control_value_ac"] = 1.02

    net_copy = copy_with_impedance(net)
    runpp(net_copy)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)

    compare_ssc_impedance_gen(net, net_copy, "vsc")


def test_minimal_vsc_hvdc():
    net = create_empty_network()
    # AC part
    create_bus(net, 110)
    create_ext_grid(net, 0)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_vsc(net, 0, 0, 0.1, 5, 0.15, control_mode_dc="vm_pu", control_value_dc=1.02)

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    # assert pd.isnull(net.res_bus_dc.at[1, 'vm_pu'])  #todo
    # assert pd.isnull(net.res_bus_dc.at[1, 'p_mw'])   #todo

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    runpp_with_consistency_checks(net)

    assert np.isclose(net.res_bus.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_ac'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[1, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_line_dc.at[0, 'vm_from_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_line_dc.at[0, 'vm_to_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_line_dc.at[0, 'loading_percent'], 0, rtol=0, atol=1e-6)

    # create_bus(net, 110)
    # create_load(net, 2, 10)
    # create_vsc(net, 1, 1, 0.1, 5, control_mode_ac="slack", control_value_ac=1.03, control_mode_dc="p_mw", control_value_dc=10)  #todo


def test_simple_vsc_hvdc():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 2, 10, q_mvar=5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)

    runpp_with_consistency_checks(net)

    create_vsc(net, 2, 1, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=1)
    runpp_with_consistency_checks(net)


def test_simple_2vsc_hvdc1():
    # np.set_printoptions(linewidth=1000, suppress=True, precision=3)
    net = create_empty_network()

    # AC part
    create_buses(net, 4, 110, geodata=[(0, 0), (100, 0), (200, 0), (300, 0)])
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 2, 3, 30, 0.0487, 0.13823, 160, 0.664)
    # create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 3, 10, q_mvar=5)

    # DC part
    create_bus_dc(net, 110, 'A', geodata=(100, 10))  # 0
    create_bus_dc(net, 110, 'B', geodata=(200, 10))  # 1

    create_bus_dc(net, 110, 'C', geodata=(100, -10)) # 2
    create_bus_dc(net, 110, 'D', geodata=(200, -10)) # 3

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")
    create_line_dc(net, 2, 3, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac='vm_pu', control_value_ac=1.,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 1, 2, 0.1, 5, 0.15,
               control_mode_ac='vm_pu', control_value_ac=1.,
               control_mode_dc="vm_pu", control_value_dc=1.02)

    create_vsc(net, 2, 1, 0.1, 5, 0.15,
               control_mode_ac='slack', control_value_ac=1.,
               control_mode_dc="p_mw", control_value_dc=-5)
    create_vsc(net, 2, 3, 0.1, 5, 0.15,
               control_mode_ac='slack', control_value_ac=1.,
               control_mode_dc="p_mw", control_value_dc=5.007)

    runpp_with_consistency_checks(net)


def test_simple_2vsc_hvdc2():
    # np.set_printoptions(linewidth=1000, suppress=True, precision=3)
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110, geodata=[(0, 0), (100, 0), (200, 0)])
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 1, 10, q_mvar=5)

    # DC part
    create_bus_dc(net, 110, 'A', geodata=(100, 10))
    create_bus_dc(net, 110, 'B', geodata=(200, 10))

    create_bus_dc(net, 110, 'C', geodata=(100, -10))
    create_bus_dc(net, 110, 'D', geodata=(200, -10))

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")
    create_line_dc(net, 2, 3, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac='vm_pu', control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=10)
    create_vsc(net, 2, 1, 0.1, 5, 0.15,
               control_mode_ac='vm_pu', control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)

    # create_vsc(net, 1, 2, 0.1, 5, control_mode_ac='vm_pu', control_value_ac=1, control_mode_dc="p_mw", control_value_dc=10)
    # create_vsc(net, 2, 3, 0.1, 5, control_mode_ac='vm_pu', control_value_ac=1, control_mode_dc="vm_pu", control_value_dc=1.02)

    create_buses(net, 2, 110, geodata=[(100, -5), (200, -5)])
    create_line_from_parameters(net, 1, 3, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 2, 4, 30, 0.0487, 0.13823, 160, 0.664)

    create_vsc(net, 3, 2, 0.1, 5, 0.15,
               control_mode_ac='vm_pu', control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=10)
    create_vsc(net, 4, 3, 0.1, 5, 0.15,
               control_mode_ac='vm_pu', control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)

    runpp_with_consistency_checks(net)


def test_b2b_vsc_1():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 2, 10, q_mvar=5)

    # DC part
    create_bus_dc(net, 110, 'A')

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=2)

    runpp_with_consistency_checks(net)
    assert np.isclose(net.res_bus_dc.at[0, 'p_mw'], 0, rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[0, 'p_dc_mw'], -net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[1, 'p_dc_mw'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)


def test_multiple_b2b_vsc_1():
    net = create_empty_network()
    # AC part
    create_buses(net, 5, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 3, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 2, 4, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 2, 10, q_mvar=5)

    # first B2B converter
    create_bus_dc(net, 110, 'A')

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=2)

    # second B2B converter
    create_bus_dc(net, 110, 'B')

    create_vsc(net, 3, 1, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 4, 1, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=4)

    runpp_with_consistency_checks(net)
    assert np.isclose(net.res_bus_dc.at[0, 'p_mw'], 0, rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[0, 'p_dc_mw'], -net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[1, 'p_dc_mw'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)

    assert np.isclose(net.res_bus_dc.at[1, 'p_mw'], 0, rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[2, 'p_dc_mw'], -net.vsc.at[3, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[3, 'p_dc_mw'], net.vsc.at[3, 'control_value_dc'], rtol=0, atol=1e-6)


def test_tres_amigas_b2b_vsc_1():
    net = create_empty_network()
    # AC part
    create_buses(net, 5, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 0, 3, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 4, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 2, 10, q_mvar=5)

    # DC system
    create_bus_dc(net, 110, 'A')

    create_vsc(net, 4, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=2)
    create_vsc(net, 3, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=4)

    runpp_with_consistency_checks(net)
    assert net.res_ext_grid.p_mw.at[0] > 10
    assert np.isclose(net.res_bus_dc.at[0, 'p_mw'], 0, rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[0, 'p_dc_mw'],
                      -net.vsc.at[1, 'control_value_dc'] - net.vsc.at[2, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[1, 'p_dc_mw'], net.vsc.at[1, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[2, 'p_dc_mw'], net.vsc.at[2, 'control_value_dc'], rtol=0, atol=1e-6)


def test_tres_amigas_b2b_vsc_2():
    net = create_empty_network()
    # AC part
    create_buses(net, 5, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 0, 3, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 1, 4, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 2, 10, q_mvar=5)

    # DC system
    create_bus_dc(net, 110, 'A')

    create_vsc(net, 4, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 3, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=4)

    runpp_with_consistency_checks(net)
    assert net.res_ext_grid.p_mw.at[0] > 10
    assert np.isclose(net.res_bus_dc.at[0, 'p_mw'], 0, rtol=0, atol=1e-6)
    assert np.isclose(net.res_bus_dc.at[0, 'vm_pu'], net.vsc.at[0, 'control_value_dc'], rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[0, 'p_dc_mw'], -net.vsc.at[2, 'control_value_dc'] / 2, rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[1, 'p_dc_mw'], -net.vsc.at[2, 'control_value_dc'] / 2, rtol=0, atol=1e-6)
    assert np.isclose(net.res_vsc.at[2, 'p_dc_mw'], net.vsc.at[2, 'control_value_dc'], rtol=0, atol=1e-6)


def test_b2b_vsc_2():
    net = create_empty_network()
    # AC part
    create_buses(net, 4, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 2, 3, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 0, 3, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 3, 10, q_mvar=5)

    # DC part
    create_bus_dc(net, 110, 'A')

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=2)

    runpp_with_consistency_checks(net)
    assert net.res_ext_grid.p_mw.at[0] > 10


def test_b2b_vsc_2a():
    net = create_empty_network()
    # AC part
    create_buses(net, 4, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 2, 3, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 3, 10, q_mvar=5)

    # DC part
    create_bus_dc(net, 110, 'A')

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 0, 0.1, 5, 0.15,
               control_mode_ac="slack", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=0)

    runpp_with_consistency_checks(net)
    assert net.res_ext_grid.p_mw.at[0] > 10


def test_b2b_vsc_3():
    net = create_empty_network()
    # AC part
    create_buses(net, 4, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 2, 3, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 3, 10, q_mvar=5)

    # DC part
    create_bus_dc(net, 110, 'A')

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.01)
    create_vsc(net, 2, 0, 0.1, 5, 0.15,
               control_mode_ac="slack", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.01)
    # whatever we put in control_mode_dc should actually be ignored

    runpp_with_consistency_checks(net)
    assert net.res_ext_grid.p_mw.at[0] > 10


@pytest.mark.xfail
def test_b2b_vsc_4():
    """
    For reasons I do not understand, this test fails on the github server, but runs locally.
    """
    net = create_empty_network()
    # AC part
    create_buses(net, 4, 110)
    create_line_from_parameters(net, 1, 3, 30, 0.0487, 0.13823, 160, 0.664)
    create_load(net, 0, 40, 10)
    create_ext_grid(net, 3)
    create_load(net, 2, 80, 20)

    # DC part
    create_bus_dc(net, 150, 'A')

    create_vsc(net, 0, 0, 0.1, 5, 0.15,
               control_mode_ac="slack", control_value_ac=1.,
               control_mode_dc="p_mw", control_value_dc=0.)
    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="q_mvar", control_value_ac=10.,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 0, 0.1, 5, 0.15,
               control_mode_ac="slack", control_value_ac=1.,
               control_mode_dc="p_mw", control_value_dc=0.)

    runpp_with_consistency_checks(net)


def test_b2b_vsc_5():
    net = create_empty_network()
    # AC part
    create_buses(net, 5, 110)
    create_line_from_parameters(net, 1, 3, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 2, 4, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 3)
    create_ext_grid(net, 4)
    create_load(net, 0, 40, 10)
    create_load(net, 2, 80, 20)

    # DC part
    create_bus_dc(net, 150, 'A')

    create_vsc(net, 0, 0, 0.1, 5, 0.15,
               control_mode_ac="slack", control_value_ac=1.,
               control_mode_dc="p_mw", control_value_dc=0.)
    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="q_mvar", control_value_ac=10.,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 0, 0.1, 5, 0.15,
               control_mode_ac="q_mvar", control_value_ac=4.,
               control_mode_dc="p_mw", control_value_dc=5.)
    runpp_with_consistency_checks(net, max_iteration=1000)


def test_b2b_vsc_6():
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 1, 2, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 2)
    create_load(net, 0, 20, 5)

    # DC part
    create_bus_dc(net, 150, 'A')
    create_vsc(net, 0, 0, 0.1, 5, 0.15,
               control_mode_ac="slack", control_value_ac=1.,
               control_mode_dc="p_mw", control_value_dc=0.)
    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="q_mvar", control_value_ac=0.,
               control_mode_dc="vm_pu", control_value_dc=1.)

    # with pytest.raises(NotImplementedError):
    #     runpp(net)
    runpp_with_consistency_checks(net)


def test_b2b_vsc_7():
    net = create_empty_network()
    # AC part
    create_buses(net, 2, 110)
    create_ext_grid(net, 1)  # todo: why is it not working when ext_grid is connected to the VSC AC bus?
    create_load(net, 0, 20, 5)

    # DC part
    create_bus_dc(net, 150, 'A')
    create_vsc(net, 0, 0, 0.1, 5, 0.15,
               control_mode_ac="slack", control_value_ac=1.,
               control_mode_dc="p_mw", control_value_dc=0.)
    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="q_mvar", control_value_ac=0.,
               control_mode_dc="vm_pu", control_value_dc=1.)

    runpp_with_consistency_checks(net)


def test_b2b_line_dc_raise():
    net = create_empty_network()
    # AC part
    create_buses(net, 4, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 2, 3, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 2, 10, q_mvar=5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=2)
    create_vsc(net, 3, 1, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=2)

    runpp_with_consistency_checks(net)

    # with pytest.raises(NotImplementedError, match="Back-To-Back"):
    #     runpp(net)


def test_line_dc_and_2_vsc1():
    net = create_empty_network()
    # AC part
    create_buses(net, 4, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 2, 3, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 2, 10, q_mvar=5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 3, 1, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=2)

    runpp_with_consistency_checks(net)


def test_line_dc_and_2_vsc2():
    net = create_empty_network()
    # AC part
    create_buses(net, 4, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 2, 3, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 3, 10, q_mvar=5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=-2)
    create_vsc(net, 2, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=-2)
    create_vsc(net, 3, 1, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)

    runpp_with_consistency_checks(net)


@pytest.mark.parametrize("control_mode_ac", list(product(['vm_pu', 'q_mvar'], repeat=2)))
@pytest.mark.parametrize("control_mode_dc", list(product(['vm_pu', 'p_mw'], repeat=2)))
def test_2vsc_1ac_2dc(control_mode_ac, control_mode_dc):
    if control_mode_dc[0] == control_mode_dc[1] == 'p_mw':
        pytest.skip("Skipping test with two 'p_mw' in control_mode_dc")
    print(f"{control_mode_dc=}, {control_mode_ac=}")

    val_ac = {"vm_pu": 1, "q_mvar": -5}
    val_dc = {"vm_pu": 1, "p_mw": 10}

    net = create_empty_network()
    # AC part
    create_buses(net, 2, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 1, 10, 3)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac=control_mode_ac[0], control_value_ac=val_ac[control_mode_ac[0]],
               control_mode_dc=control_mode_dc[0], control_value_dc=val_dc[control_mode_dc[0]])
    create_vsc(net, 1, 1, 0.1, 5, 0.15,
               control_mode_ac=control_mode_ac[1], control_value_ac=val_ac[control_mode_ac[1]],
               control_mode_dc=control_mode_dc[1], control_value_dc=val_dc[control_mode_dc[1]])

    if control_mode_ac[0] == control_mode_ac[1]:
        runpp_with_consistency_checks(net)
    else:
        with pytest.raises(NotImplementedError, match="share the same AC bus"):
            runpp(net)


@pytest.mark.skip(reason="DC line connected to D2B VSC configuration not implemented")
@pytest.mark.parametrize("control_mode_ac", list(product(['vm_pu', 'q_mvar'], repeat=2)))
@pytest.mark.parametrize("control_mode_dc", list(product(['vm_pu', 'p_mw'], repeat=2)))
def test_2vsc_2ac_1dc(control_mode_ac, control_mode_dc):
    if control_mode_dc[0] == control_mode_dc[1] == 'p_mw':
        pytest.skip("Skipping test with two 'p_mw' in control_mode_dc")

    val_ac = {"vm_pu": 1, "q_mvar": -5}
    val_dc = {"vm_pu": 1, "p_mw": 10}

    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)

    create_ext_grid(net, 0)
    create_load(net, 1, 10, 3)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'A')

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15, control_mode_ac=control_mode_ac[0],
               control_value_ac=val_ac[control_mode_ac[0]], control_mode_dc=control_mode_dc[0],
               control_value_dc=val_dc[control_mode_dc[0]])
    create_vsc(net, 2, 0, 0.1, 5, 0.15, control_mode_ac=control_mode_ac[1],
               control_value_ac=val_ac[control_mode_ac[1]], control_mode_dc=control_mode_dc[1],
               control_value_dc=val_dc[control_mode_dc[1]])

    runpp_with_consistency_checks(net)


@pytest.mark.skip(reason="DC line connected to D2B VSC configuration not implemented")
@pytest.mark.parametrize("control_mode_ac", list(product(['vm_pu', 'q_mvar'], repeat=2)))
@pytest.mark.parametrize("control_mode_dc", list(product(['vm_pu', 'p_mw'], repeat=2)))
def test_2vsc_1ac_1dc(control_mode_ac, control_mode_dc):
    if control_mode_dc[0] == control_mode_dc[1] == 'p_mw':
        pytest.skip("Skipping test with two 'p_mw' in control_mode_dc")

    val_ac = {"vm_pu": 1, "q_mvar": -5}
    val_dc = {"vm_pu": 1, "p_mw": 10}

    net = create_empty_network()
    # AC part
    create_buses(net, 2, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)

    create_ext_grid(net, 0)
    create_load(net, 1, 10, q_mvar=5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'A')

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15, control_mode_ac=control_mode_ac[0],
               control_value_ac=val_ac[control_mode_ac[0]], control_mode_dc=control_mode_dc[0],
               control_value_dc=val_dc[control_mode_dc[0]])
    create_vsc(net, 1, 0, 0.1, 5, 0.15, control_mode_ac=control_mode_ac[1],
               control_value_ac=val_ac[control_mode_ac[1]], control_mode_dc=control_mode_dc[1],
               control_value_dc=val_dc[control_mode_dc[1]])

    if control_mode_ac[0] == control_mode_ac[1]:
        runpp_with_consistency_checks(net)
    else:
        with pytest.raises(NotImplementedError, match="share the same AC bus"):
            runpp(net)


def test_vsc_slack_minimal_wrong():
    # np.set_printoptions(linewidth=1000, suppress=True, precision=2)
    # from pandapower.test.loadflow.test_facts import *
    net = create_empty_network()
    # AC part
    create_buses(net, 2, 110, geodata=[(200, 0), (400, 0)])
    create_load(net, 1, 10, 4)
    create_gen(net, 0, 0)

    # DC part
    create_bus_dc(net, 110, 'A', geodata=(210, 0))
    create_bus_dc(net, 110, 'B', geodata=(390, 0))

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 0, 0, 0.1, 5, 0.15,
               control_mode_ac="slack", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 1, 1, 0.1, 5, 0.15,
               control_mode_ac="slack", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=1)

    # VSC as slack cannot have DC bus as vm_pu, therefore must be excluded from DC slacks
    # Then the DC buses are set out of service, and the corresponding VSC are also set out of service
    # Then the corresponding AC buses are changed from type REF to type PQ, which is valid because type PV is set later
    # Then runpp raises "no slacks" error:
    with pytest.raises(UserWarning, match="No reference bus is available"):
        runpp(net)


def test_vsc_slack_minimal_wrong2():
    # np.set_printoptions(linewidth=1000, suppress=True, precision=2)
    # from pandapower.test.loadflow.test_facts import *
    net = create_empty_network()
    # AC part
    create_buses(net, 2, 110, geodata=[(200, 0), (400, 0)])
    create_load(net, 1, 10, 4)

    # DC part
    create_bus_dc(net, 110, 'A', geodata=(210, 0))
    create_bus_dc(net, 110, 'B', geodata=(390, 0))

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 0, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 1, 1, 0.1, 5, 0.15,
               control_mode_ac="slack", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=1)

    # VSC that defines AC slack cannot define DC slack at the same time
    # DC slack buses that are only connected to VSC AC slacks are converted to type P buses
    # Then runpp raises "no DC slacks" error:
    with pytest.raises(UserWarning, match="No reference bus is available."):
        runpp(net)


@pytest.mark.xfail(reason="AC bus same as ext_grid bus not implemented")
def test_vsc_slack_minimal():  # todo: fix that FACTS elements can be connected to ext_grid buses
    # np.set_printoptions(linewidth=1000, suppress=True, precision=2)
    # from pandapower.test.loadflow.test_facts import *
    net = create_empty_network()
    # AC part
    create_buses(net, 2, 110, geodata=[(200, 0), (400, 0)])
    create_load(net, 1, 10, 4)
    create_ext_grid(net, 0)

    # DC part
    create_bus_dc(net, 110, 'A', geodata=(210, 0))
    create_bus_dc(net, 110, 'B', geodata=(390, 0))

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 0, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 1, 1, 0.1, 5, 0.15,
               control_mode_ac="slack", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=1)

    # runpp(net)

    runpp_with_consistency_checks(net)

    # plotting.simple_plot(net, plot_loads=True, load_size=5)


def test_vsc_slack():
    # np.set_printoptions(linewidth=1000, suppress=True, precision=3)
    # from pandapower.test.loadflow.test_facts import *
    net = create_empty_network()
    # AC part
    create_buses(net, 4, 110, geodata=[(0, 0), (200, 0), (400, 0), (600, 0)])
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 2, 3, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 3, 10, 4)

    # DC part
    create_bus_dc(net, 110, 'A', geodata=(210, 0))
    create_bus_dc(net, 110, 'B', geodata=(390, 0))

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 1, 0.1, 5, 0.15,
               control_mode_ac="slack", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=0.)

    runpp_with_consistency_checks(net)

    # plotting.simple_plot(net, plot_loads=True)


def test_vsc_slack2():
    # np.set_printoptions(linewidth=1000, suppress=True, precision=3)
    # from pandapower.test.loadflow.test_facts import *
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110, geodata=[(0, 0), (200, 0), (400, 0)])
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 2, 10, 4)

    # DC part
    create_bus_dc(net, 110, 'A', geodata=(210, 0))
    create_bus_dc(net, 110, 'B', geodata=(390, 0))

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 1, 0.1, 5, 0.15,
               control_mode_ac="slack", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=10)

    runpp_with_consistency_checks(net)

    # plotting.simple_plot(net, plot_loads=True)


def test_vsc_slack_oos():
    # np.set_printoptions(linewidth=1000, suppress=True, precision=3)
    # from pandapower.test.loadflow.test_facts import *
    net = create_empty_network()
    # AC part
    create_buses(net, 4, 110, geodata=[(0, 0), (200, 0), (400, 0), (600, 0)])
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 2, 3, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 3, 10, 4)
    net.bus.loc[[2, 3], "in_service"] = False

    # DC part
    create_bus_dc(net, 110, 'A', geodata=(210, 0))
    create_bus_dc(net, 110, 'B', geodata=(390, 0))

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 1, 0.1, 5, 0.15,
               control_mode_ac="slack", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=0.)

    runpp_with_consistency_checks(net)


def test_vsc_dc_r():
    # np.set_printoptions(linewidth=1000, suppress=True, precision=3)
    # from pandapower.test.loadflow.test_facts import *
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110, geodata=[(0, 0), (200, 0), (400, 0)])
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 2, 10, 4)

    # DC part
    create_bus_dc(net, 110, 'A', geodata=(210, 0))
    create_bus_dc(net, 110, 'B', geodata=(390, 0))

    create_line_dc(net, 0, 1, 100, std_type="2400-CU")

    create_vsc(net, 1, 0, 0.1, 5, 0.15,
               control_mode_ac="vm_pu", control_value_ac=1,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 1, 0.1, 5, 0.15,
               control_mode_ac="slack", control_value_ac=1,
               control_mode_dc="p_mw", control_value_dc=10)

    runpp_with_consistency_checks(net)


def test_vsc_hvdc_dc_rl():
    # from pandapower.test.loadflow.test_facts import *
    net = create_empty_network()
    # AC part
    create_buses(net, 3, 110)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)
    create_line_from_parameters(net, 0, 2, 30, 0.0487, 0.13823, 160, 0.664)
    create_ext_grid(net, 0)
    create_load(net, 2, 10, 5)

    # DC part
    create_bus_dc(net, 110, 'A')
    create_bus_dc(net, 110, 'B')

    create_line_dc_from_parameters(net, 0, 1, 100, 0.1, 1)

    create_vsc(net, 1, 0, 0.1, 5, 0.5, pl_dc_mw=0.5,
               control_mode_ac="vm_pu", control_value_ac=1.,
               control_mode_dc="vm_pu", control_value_dc=1.02)
    create_vsc(net, 2, 1, 0.1, 5, 0.5, pl_dc_mw=0.75,
               control_mode_ac="vm_pu", control_value_ac=1.,
               control_mode_dc="p_mw", control_value_dc=5)

    runpp_with_consistency_checks(net)


def test_results_pf_grid():
    # todo: improve accuracy of DC losses and reduce tolerances
    path = os.path.join(pp_dir, "test", "test_files", "test_ac_dc.json")
    net = from_json(path)
    res = validate_pf_conversion(net)
    assert np.max(np.abs(res['diff_vm']['diff'])) < 1e-6
    assert np.max(np.abs(res['diff_va']['diff'])) < 1e-3
    assert np.max(np.abs(res['bus_dc_diff'])) < 5e-6
    assert np.max(np.abs(res['line_diff'])) < 0.02
    assert np.max(np.abs(res['line_dc_diff'])) < 1e-3
    assert np.max(np.abs(res['vsc_p_diff_is'])) < 0.02
    assert np.max(np.abs(res['vsc_q_diff_is'])) < 0.01
    assert np.max(np.abs(res['vsc_p_dc_diff_is'])) < 1e-3
    assert np.max(np.abs(res['ext_grid_p_diff'])) < 0.05
    assert np.max(np.abs(res['ext_grid_q_diff'])) < 0.01


# TODO test for when the VSC, SSC, TCSC, connect to same buses

if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import pandapower as pp
try:
    import pplog as logging
except:
    import logging
logger = logging.getLogger(__name__)


def create_cigre_network_hv(length_km_6a_6b=0.1):
    net_cigre_hv = pp.create_empty_network()

    # Linedata
    # Line220kV
    line_data = {'c_nf_per_km': 9.08, 'r_ohm_per_km': 0.0653,
                 'x_ohm_per_km': 0.398, 'max_i_ka': 1.14,
                 'type': 'ol'}

    pp.create_std_type(net_cigre_hv, line_data, 'Line220kV', element='line')

    # Line380kV
    line_data = {'c_nf_per_km': 11.5, 'r_ohm_per_km': 0.0328,
                 'x_ohm_per_km': 0.312, 'max_i_ka': 1.32,
                 'type': 'ol'}

    pp.create_std_type(net_cigre_hv, line_data, 'Line380kV', element='line')

    # Busses
    bus1 = pp.create_bus(net_cigre_hv, name='Bus 1', vn_kv=220, type='b', zone='CIGRE_HV')
    bus2 = pp.create_bus(net_cigre_hv, name='Bus 2', vn_kv=220, type='b', zone='CIGRE_HV')
    bus3 = pp.create_bus(net_cigre_hv, name='Bus 3', vn_kv=220, type='b', zone='CIGRE_HV')
    bus4 = pp.create_bus(net_cigre_hv, name='Bus 4', vn_kv=220, type='b', zone='CIGRE_HV')
    bus5 = pp.create_bus(net_cigre_hv, name='Bus 5', vn_kv=220, type='b', zone='CIGRE_HV')
    bus6a = pp.create_bus(net_cigre_hv, name='Bus 6a', vn_kv=220, type='b', zone='CIGRE_HV')
    bus6b = pp.create_bus(net_cigre_hv, name='Bus 6b', vn_kv=220, type='b', zone='CIGRE_HV')
    bus7 = pp.create_bus(net_cigre_hv, name='Bus 7', vn_kv=380, type='b', zone='CIGRE_HV')
    bus8 = pp.create_bus(net_cigre_hv, name='Bus 8', vn_kv=380, type='b', zone='CIGRE_HV')
    bus9 = pp.create_bus(net_cigre_hv, name='Bus 9', vn_kv=22, type='b', zone='CIGRE_HV')
    bus10 = pp.create_bus(net_cigre_hv, name='Bus 10', vn_kv=22, type='b', zone='CIGRE_HV')
    bus11 = pp.create_bus(net_cigre_hv, name='Bus 11', vn_kv=22, type='b', zone='CIGRE_HV')
    bus12 = pp.create_bus(net_cigre_hv, name='Bus 12', vn_kv=22, type='b', zone='CIGRE_HV')

    # Lines
    pp.create_line(net_cigre_hv, bus1, bus2, length_km=100,
                   std_type='Line220kV', name='Line 1-2')
    pp.create_line(net_cigre_hv, bus1, bus6a, length_km=300,
                   std_type='Line220kV', name='Line 1-6a')
    pp.create_line(net_cigre_hv, bus2, bus5, length_km=300,
                   std_type='Line220kV', name='Line 2-5')
    pp.create_line(net_cigre_hv, bus3, bus4, length_km=100,
                   std_type='Line220kV', name='Line 3-4')
    pp.create_line(net_cigre_hv, bus3, bus4, length_km=100,
                   std_type='Line220kV', name='Line 3-4_2')
    pp.create_line(net_cigre_hv, bus4, bus5, length_km=300,
                   std_type='Line220kV', name='Line 4-5')
    pp.create_line(net_cigre_hv, bus4, bus6a, length_km=300,
                   std_type='Line220kV', name='Line 4-6a')
    pp.create_line(net_cigre_hv, bus7, bus8, length_km=600,
                   std_type='Line380kV', name='Line 7-8')
    pp.create_line(net_cigre_hv, bus6a, bus6b, length_km=length_km_6a_6b,
                   std_type='Line220kV', name='Line 6a-6b')

    # Trafos
    pp.create_transformer_from_parameters(net_cigre_hv, bus7, bus1, sn_kva=1000000,
                                          vn_hv_kv=380, vn_lv_kv=220, vscr_percent=0.0,
                                          vsc_percent=13.0, pfe_kw=0, i0_percent=0,
                                          shift_degree=0.0, name='Trafo 1-7')
    pp.create_transformer_from_parameters(net_cigre_hv, bus8, bus3, sn_kva=1000000,
                                          vn_hv_kv=380, vn_lv_kv=220, vscr_percent=0.0,
                                          vsc_percent=13.0, pfe_kw=0, i0_percent=0,
                                          shift_degree=0.0, name='Trafo 3-8')

    pp.create_transformer_from_parameters(net_cigre_hv, bus1, bus9, sn_kva=1000000,
                                          vn_hv_kv=220, vn_lv_kv=22, vscr_percent=0.0,
                                          vsc_percent=13.0, pfe_kw=0, i0_percent=0,
                                          shift_degree=330.0, name='Trafo 9-1')
    pp.create_transformer_from_parameters(net_cigre_hv, bus2, bus10, sn_kva=1000000,
                                          vn_hv_kv=220, vn_lv_kv=22, vscr_percent=0.0,
                                          vsc_percent=13.0, pfe_kw=0, i0_percent=0,
                                          shift_degree=330.0, name='Trafo 10-2')
    pp.create_transformer_from_parameters(net_cigre_hv, bus3, bus11, sn_kva=1000000,
                                          vn_hv_kv=220, vn_lv_kv=22, vscr_percent=0.0,
                                          vsc_percent=13.0, pfe_kw=0, i0_percent=0,
                                          shift_degree=330.0, name='Trafo 11-3')
    pp.create_transformer_from_parameters(net_cigre_hv, bus6b, bus12, sn_kva=500000,
                                          vn_hv_kv=220, vn_lv_kv=22, vscr_percent=0.0,
                                          vsc_percent=13.0, pfe_kw=0, i0_percent=0,
                                          shift_degree=330.0, name='Trafo 12-6b')

    # Loads
    pp.create_load(net_cigre_hv, bus2, p_kw=285000, q_kvar=200000, name='Load 2')
    pp.create_load(net_cigre_hv, bus3, p_kw=325000, q_kvar=244000, name='Load 3')
    pp.create_load(net_cigre_hv, bus4, p_kw=326000, q_kvar=244000, name='Load 4')
    pp.create_load(net_cigre_hv, bus5, p_kw=103000, q_kvar=62000, name='Load 5')
    pp.create_load(net_cigre_hv, bus6a, p_kw=435000, q_kvar=296000, name='Load 6a')

    # External grid
    pp.create_ext_grid(net_cigre_hv, bus9, vm_pu=1.03, va_degree=0, name='Generator 9')

    # Generators
    pp.create_gen(net_cigre_hv, bus10, vm_pu=1.03, p_kw=-5e5, name='Generator 10')
    pp.create_gen(net_cigre_hv, bus11, vm_pu=1.03, p_kw=-2e5, name='Generator 11')
    pp.create_gen(net_cigre_hv, bus12, vm_pu=1.03, p_kw=-3e5, name='Generator 12')

    # Shunts
    pp.create_shunt(net_cigre_hv, bus4, p_kw=0.0, q_kvar=-160000, name='Shunt 4')
    pp.create_shunt(net_cigre_hv, bus5, p_kw=0.0, q_kvar=-80000, name='Shunt 5')
    pp.create_shunt(net_cigre_hv, bus6a, p_kw=0.0, q_kvar=-180000, name='Shunt 6a')

    return net_cigre_hv


def create_cigre_network_mv(with_der=False):
    if with_der is True:
        raise ValueError("'with_der=True' is deprecated. Please use 'with_der=pv_wind'")
    if with_der not in [False, "pv_wind", "all"]:
        raise ValueError("'with_der' is unknown. It should be in [False, 'pv_wind', 'all'].")

    net_cigre_mv = pp.create_empty_network()

    # Linedata
    line_data = {'c_nf_per_km': 151.1749, 'r_ohm_per_km': 0.501,
                 'x_ohm_per_km': 0.716, 'max_i_ka': 0.145,
                 'type': 'cs'}
    pp.create_std_type(net_cigre_mv, line_data, name='CABLE_CIGRE_MV', element='line')

    line_data = {'c_nf_per_km': 10.09679, 'r_ohm_per_km': 0.510,
                 'x_ohm_per_km': 0.366, 'max_i_ka': 0.195,
                 'type': 'ol'}
    pp.create_std_type(net_cigre_mv, line_data, name='OHL_CIGRE_MV', element='line')

    # Busses
    bus0 = pp.create_bus(net_cigre_mv, name='Bus 0', vn_kv=110, type='b', zone='CIGRE_MV')
    bus1 = pp.create_bus(net_cigre_mv, name='Bus 1', vn_kv=20, type='b', zone='CIGRE_MV')
    bus2 = pp.create_bus(net_cigre_mv, name='Bus 2', vn_kv=20, type='b', zone='CIGRE_MV')
    bus3 = pp.create_bus(net_cigre_mv, name='Bus 3', vn_kv=20, type='b', zone='CIGRE_MV')
    bus4 = pp.create_bus(net_cigre_mv, name='Bus 4', vn_kv=20, type='b', zone='CIGRE_MV')
    bus5 = pp.create_bus(net_cigre_mv, name='Bus 5', vn_kv=20, type='b', zone='CIGRE_MV')
    bus6 = pp.create_bus(net_cigre_mv, name='Bus 6', vn_kv=20, type='b', zone='CIGRE_MV')
    bus7 = pp.create_bus(net_cigre_mv, name='Bus 7', vn_kv=20, type='b', zone='CIGRE_MV')
    bus8 = pp.create_bus(net_cigre_mv, name='Bus 8', vn_kv=20, type='b', zone='CIGRE_MV')
    bus9 = pp.create_bus(net_cigre_mv, name='Bus 9', vn_kv=20, type='b', zone='CIGRE_MV')
    bus10 = pp.create_bus(net_cigre_mv, name='Bus 10', vn_kv=20, type='b', zone='CIGRE_MV')
    bus11 = pp.create_bus(net_cigre_mv, name='Bus 11', vn_kv=20, type='b', zone='CIGRE_MV')
    bus12 = pp.create_bus(net_cigre_mv, name='Bus 12', vn_kv=20, type='b', zone='CIGRE_MV')
    bus13 = pp.create_bus(net_cigre_mv, name='Bus 13', vn_kv=20, type='b', zone='CIGRE_MV')
    bus14 = pp.create_bus(net_cigre_mv, name='Bus 14', vn_kv=20, type='b', zone='CIGRE_MV')

    # Lines
    pp.create_line(net_cigre_mv, bus1, bus2, length_km=2.82,
                   std_type='CABLE_CIGRE_MV', name='Line 1-2')
    pp.create_line(net_cigre_mv, bus2, bus3, length_km=4.42,
                   std_type='CABLE_CIGRE_MV', name='Line 2-3')
    pp.create_line(net_cigre_mv, bus3, bus4, length_km=0.61,
                   std_type='CABLE_CIGRE_MV', name='Line 3-4')
    pp.create_line(net_cigre_mv, bus4, bus5, length_km=0.56,
                   std_type='CABLE_CIGRE_MV', name='Line 4-5')
    pp.create_line(net_cigre_mv, bus5, bus6, length_km=1.54,
                   std_type='CABLE_CIGRE_MV', name='Line 5-6')
    pp.create_line(net_cigre_mv, bus7, bus8, length_km=1.67,
                   std_type='CABLE_CIGRE_MV', name='Line 7-8')
    pp.create_line(net_cigre_mv, bus8, bus9, length_km=0.32,
                   std_type='CABLE_CIGRE_MV', name='Line 8-9')
    pp.create_line(net_cigre_mv, bus9, bus10, length_km=0.77,
                   std_type='CABLE_CIGRE_MV', name='Line 9-10')
    pp.create_line(net_cigre_mv, bus10, bus11, length_km=0.33,
                   std_type='CABLE_CIGRE_MV', name='Line 10-11')
    pp.create_line(net_cigre_mv, bus3, bus8, length_km=1.3,
                   std_type='CABLE_CIGRE_MV', name='Line 3-8')
    pp.create_line(net_cigre_mv, bus12, bus13, length_km=4.89,
                   std_type='OHL_CIGRE_MV', name='Line 12-13')
    pp.create_line(net_cigre_mv, bus13, bus14, length_km=2.99,
                   std_type='OHL_CIGRE_MV', name='Line 13-14')

    line6_7 = pp.create_line(net_cigre_mv, bus6, bus7, length_km=0.24,
                             std_type='CABLE_CIGRE_MV', name='Line 6-7')
    line4_11 = pp.create_line(net_cigre_mv, bus11, bus4, length_km=0.49,
                              std_type='CABLE_CIGRE_MV', name='Line 11-4')
    line8_14 = pp.create_line(net_cigre_mv, bus14, bus8, length_km=2.,
                              std_type='OHL_CIGRE_MV', name='Line 14-8')

    # Ext-Grid
    pp.create_ext_grid(net_cigre_mv, bus0, vm_pu=1.03, va_degree=0.,
                       s_sc_max_mva=5000, s_sc_min_mva=5000, rx_max=0.1, rx_min=0.1)

    # Trafos
    trafo0 = pp.create_transformer_from_parameters(net_cigre_mv, bus0, bus1, sn_kva=25000,
                                                   vn_hv_kv=110, vn_lv_kv=20, vscr_percent=0.16,
                                                   vsc_percent=12.00107, pfe_kw=0, i0_percent=0,
                                                   shift_degree=30.0, name='Trafo 0-1')
    trafo1 = pp.create_transformer_from_parameters(net_cigre_mv, bus0, bus12, sn_kva=25000,
                                                   vn_hv_kv=110, vn_lv_kv=20, vscr_percent=0.16,
                                                   vsc_percent=12.00107, pfe_kw=0, i0_percent=0,
                                                   shift_degree=30.0, name='Trafo 0-12')

    # Switches
    # S2
    pp.create_switch(net_cigre_mv, bus6, line6_7, et='l', closed=True, type='LBS')
    pp.create_switch(net_cigre_mv, bus7, line6_7, et='l', closed=False, type='LBS', name='S2')
    # S3
    pp.create_switch(net_cigre_mv, bus4, line4_11, et='l', closed=False, type='LBS', name='S3')
    pp.create_switch(net_cigre_mv, bus11, line4_11, et='l', closed=True, type='LBS')
    # S1
    pp.create_switch(net_cigre_mv, bus8, line8_14, et='l', closed=False, type='LBS', name='S1')
    pp.create_switch(net_cigre_mv, bus14, line8_14, et='l', closed=True, type='LBS')
    # trafos
    pp.create_switch(net_cigre_mv, bus0, trafo0, et='t', closed=True, type='CB')
    pp.create_switch(net_cigre_mv, bus0, trafo1, et='t', closed=True, type='CB')

    # Loads
    # Residential
    pp.create_load(net_cigre_mv, bus1, p_kw=14994.0, q_kvar=3044.66156, name='Load R1')
    pp.create_load(net_cigre_mv, bus3, p_kw=276.45, q_kvar=69.28490, name='Load R3')
    pp.create_load(net_cigre_mv, bus4, p_kw=431.65, q_kvar=108.18169, name='Load R4')
    pp.create_load(net_cigre_mv, bus5, p_kw=727.5, q_kvar=182.32869, name='Load R5')
    pp.create_load(net_cigre_mv, bus6, p_kw=548.05, q_kvar=137.35428, name='Load R6')
    pp.create_load(net_cigre_mv, bus8, p_kw=586.85, q_kvar=147.07847, name='Load R8')
    pp.create_load(net_cigre_mv, bus10, p_kw=475.3, q_kvar=119.12141, name='Load R10')
    pp.create_load(net_cigre_mv, bus11, p_kw=329.8, q_kvar=82.65567, name='Load R11')
    pp.create_load(net_cigre_mv, bus12, p_kw=14994.0, q_kvar=3044.66156, name='Load R12')
    pp.create_load(net_cigre_mv, bus14, p_kw=208.55, q_kvar=52.26756, name='Load R14')

    # Commercial / Industrial
    pp.create_load(net_cigre_mv, bus1, p_kw=4845.0, q_kvar=1592.47449, name='Load CI1')
    pp.create_load(net_cigre_mv, bus3, p_kw=225.25, q_kvar=139.59741, name='Load CI3')
    pp.create_load(net_cigre_mv, bus7, p_kw=76.5, q_kvar=47.41044, name='Load CI7')
    pp.create_load(net_cigre_mv, bus9, p_kw=573.75, q_kvar=355.57831, name='Load CI9')
    pp.create_load(net_cigre_mv, bus10, p_kw=68.0, q_kvar=42.14262, name='Load CI10')
    pp.create_load(net_cigre_mv, bus12, p_kw=5016.0, q_kvar=1648.67947, name='Load CI12')
    pp.create_load(net_cigre_mv, bus13, p_kw=34.0, q_kvar=21.07131, name='Load CI13')
    pp.create_load(net_cigre_mv, bus14, p_kw=331.5, q_kvar=205.44525, name='Load CI14')

    # Optional distributed energy recources
    if with_der in ["pv_wind", "all"]:
        pp.create_sgen(net_cigre_mv, bus3, p_kw=-20, q_kvar=0, sn_kva=20, name='PV 3', type='PV')
        pp.create_sgen(net_cigre_mv, bus4, p_kw=-20, q_kvar=0, sn_kva=20, name='PV 4', type='PV')
        pp.create_sgen(net_cigre_mv, bus5, p_kw=-30, q_kvar=0, sn_kva=30, name='PV 5', type='PV')
        pp.create_sgen(net_cigre_mv, bus6, p_kw=-30, q_kvar=0, sn_kva=30, name='PV 6', type='PV')
        pp.create_sgen(net_cigre_mv, bus8, p_kw=-30, q_kvar=0, sn_kva=30, name='PV 8', type='PV')
        pp.create_sgen(net_cigre_mv, bus9, p_kw=-30, q_kvar=0, sn_kva=30, name='PV 9', type='PV')
        pp.create_sgen(net_cigre_mv, bus10, p_kw=-40, q_kvar=0, sn_kva=40, name='PV 10', type='PV')
        pp.create_sgen(net_cigre_mv, bus11, p_kw=-10, q_kvar=0, sn_kva=10, name='PV 11', type='PV')
        pp.create_sgen(net_cigre_mv, bus7, p_kw=-1500, q_kvar=0, sn_kva=1500, name='WKA 7',
                       type='WP')
        if with_der == "all":
            pp.create_sgen(net_cigre_mv, bus=bus5, p_kw=-600, sn_kva=600, name='Battery 1',
                           type='Battery', max_p_kw=-600, min_p_kw=600)
            pp.create_sgen(net_cigre_mv, bus=bus5, p_kw=-33, sn_kva=33,
                           name='Residential fuel cell 1', type='Residential fuel cell')
            pp.create_sgen(net_cigre_mv, bus=bus9, p_kw=-310, sn_kva=310, name='CHP diesel 1',
                           type='CHP diesel')
            pp.create_sgen(net_cigre_mv, bus=bus9, p_kw=-212, sn_kva=212, name='Fuel cell 1',
                           type='Fuel cell')
            pp.create_sgen(net_cigre_mv, bus=bus10, p_kw=0, sn_kva=200, name='Battery 2',
                           type='Battery', max_p_kw=-200, min_p_kw=200)
            pp.create_sgen(net_cigre_mv, bus=bus10, p_kw=-14, sn_kva=14,
                           name='Residential fuel cell 2', type='Residential fuel cell')
    return net_cigre_mv


def create_cigre_network_lv():
    net_cigre_lv = pp.create_empty_network()

    # Linedata
    # UG1
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.162,
                 'x_ohm_per_km': 0.0832, 'max_i_ka': 1.0,
                 'type': 'cs'}
    pp.create_std_type(net_cigre_lv, line_data, name='UG1', element='line')

    # UG2
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.2647,
                 'x_ohm_per_km': 0.0823, 'max_i_ka': 1.0,
                 'type': 'cs'}
    pp.create_std_type(net_cigre_lv, line_data, name='UG2', element='line')

    # UG3
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.822,
                 'x_ohm_per_km': 0.0847, 'max_i_ka': 1.0,
                 'type': 'cs'}
    pp.create_std_type(net_cigre_lv, line_data, name='UG3', element='line')

    # OH1
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 0.4917,
                 'x_ohm_per_km': 0.2847, 'max_i_ka': 1.0,
                 'type': 'ol'}
    pp.create_std_type(net_cigre_lv, line_data, name='OH1', element='line')

    # OH2
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 1.3207,
                 'x_ohm_per_km': 0.321, 'max_i_ka': 1.0,
                 'type': 'ol'}
    pp.create_std_type(net_cigre_lv, line_data, name='OH2', element='line')

    # OH3
    line_data = {'c_nf_per_km': 0.0, 'r_ohm_per_km': 2.0167,
                 'x_ohm_per_km': 0.3343, 'max_i_ka': 1.0,
                 'type': 'ol'}
    pp.create_std_type(net_cigre_lv, line_data, name='OH3', element='line')

    # Busses
    bus0 = pp.create_bus(net_cigre_lv, name='Bus 0', vn_kv=20.0, type='b', zone='CIGRE_LV')
    busR0 = pp.create_bus(net_cigre_lv, name='Bus R0', vn_kv=20.0, type='b', zone='CIGRE_LV')
    busR1 = pp.create_bus(net_cigre_lv, name='Bus R1', vn_kv=0.4, type='b', zone='CIGRE_LV')
    busR2 = pp.create_bus(net_cigre_lv, name='Bus R2', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR3 = pp.create_bus(net_cigre_lv, name='Bus R3', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR4 = pp.create_bus(net_cigre_lv, name='Bus R4', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR5 = pp.create_bus(net_cigre_lv, name='Bus R5', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR6 = pp.create_bus(net_cigre_lv, name='Bus R6', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR7 = pp.create_bus(net_cigre_lv, name='Bus R7', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR8 = pp.create_bus(net_cigre_lv, name='Bus R8', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR9 = pp.create_bus(net_cigre_lv, name='Bus R9', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR10 = pp.create_bus(net_cigre_lv, name='Bus R10', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR11 = pp.create_bus(net_cigre_lv, name='Bus R11', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR12 = pp.create_bus(net_cigre_lv, name='Bus R12', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR13 = pp.create_bus(net_cigre_lv, name='Bus R13', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR14 = pp.create_bus(net_cigre_lv, name='Bus R14', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR15 = pp.create_bus(net_cigre_lv, name='Bus R15', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR16 = pp.create_bus(net_cigre_lv, name='Bus R16', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR17 = pp.create_bus(net_cigre_lv, name='Bus R17', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busR18 = pp.create_bus(net_cigre_lv, name='Bus R18', vn_kv=0.4, type='m', zone='CIGRE_LV')

    busI0 = pp.create_bus(net_cigre_lv, name='Bus I0', vn_kv=20.0, type='b', zone='CIGRE_LV')
    busI1 = pp.create_bus(net_cigre_lv, name='Bus I1', vn_kv=0.4, type='b', zone='CIGRE_LV')
    busI2 = pp.create_bus(net_cigre_lv, name='Bus I2', vn_kv=0.4, type='m', zone='CIGRE_LV')

    busC0 = pp.create_bus(net_cigre_lv, name='Bus C0', vn_kv=20.0, type='b', zone='CIGRE_LV')
    busC1 = pp.create_bus(net_cigre_lv, name='Bus C1', vn_kv=0.4, type='b', zone='CIGRE_LV')
    busC2 = pp.create_bus(net_cigre_lv, name='Bus C2', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC3 = pp.create_bus(net_cigre_lv, name='Bus C3', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC4 = pp.create_bus(net_cigre_lv, name='Bus C4', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC5 = pp.create_bus(net_cigre_lv, name='Bus C5', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC6 = pp.create_bus(net_cigre_lv, name='Bus C6', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC7 = pp.create_bus(net_cigre_lv, name='Bus C7', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC8 = pp.create_bus(net_cigre_lv, name='Bus C8', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC9 = pp.create_bus(net_cigre_lv, name='Bus C9', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC10 = pp.create_bus(net_cigre_lv, name='Bus C10', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC11 = pp.create_bus(net_cigre_lv, name='Bus C11', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC12 = pp.create_bus(net_cigre_lv, name='Bus C12', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC13 = pp.create_bus(net_cigre_lv, name='Bus C13', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC14 = pp.create_bus(net_cigre_lv, name='Bus C14', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC15 = pp.create_bus(net_cigre_lv, name='Bus C15', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC16 = pp.create_bus(net_cigre_lv, name='Bus C16', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC17 = pp.create_bus(net_cigre_lv, name='Bus C17', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC18 = pp.create_bus(net_cigre_lv, name='Bus C18', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC19 = pp.create_bus(net_cigre_lv, name='Bus C19', vn_kv=0.4, type='m', zone='CIGRE_LV')
    busC20 = pp.create_bus(net_cigre_lv, name='Bus C20', vn_kv=0.4, type='m', zone='CIGRE_LV')

    # Lines
    pp.create_line(net_cigre_lv, busR1, busR2, length_km=0.035, std_type='UG1',
                   name='Line R1-R2')
    pp.create_line(net_cigre_lv, busR2, busR3, length_km=0.035, std_type='UG1',
                   name='Line R2-R3')
    pp.create_line(net_cigre_lv, busR3, busR4, length_km=0.035, std_type='UG1',
                   name='Line R3-R4')
    pp.create_line(net_cigre_lv, busR4, busR5, length_km=0.035, std_type='UG1',
                   name='Line R4-R5')
    pp.create_line(net_cigre_lv, busR5, busR6, length_km=0.035, std_type='UG1',
                   name='Line R5-R6')
    pp.create_line(net_cigre_lv, busR6, busR7, length_km=0.035, std_type='UG1',
                   name='Line R6-R7')
    pp.create_line(net_cigre_lv, busR7, busR8, length_km=0.035, std_type='UG1',
                   name='Line R7-R8')
    pp.create_line(net_cigre_lv, busR8, busR9, length_km=0.035, std_type='UG1',
                   name='Line R8-R9')
    pp.create_line(net_cigre_lv, busR9, busR10, length_km=0.035, std_type='UG1',
                   name='Line R9-R10')
    pp.create_line(net_cigre_lv, busR3, busR11, length_km=0.030, std_type='UG3',
                   name='Line R3-R11')
    pp.create_line(net_cigre_lv, busR4, busR12, length_km=0.035, std_type='UG3',
                   name='Line R4-R12')
    pp.create_line(net_cigre_lv, busR12, busR13, length_km=0.035, std_type='UG3',
                   name='Line R12-R13')
    pp.create_line(net_cigre_lv, busR13, busR14, length_km=0.035, std_type='UG3',
                   name='Line R13-R14')
    pp.create_line(net_cigre_lv, busR14, busR15, length_km=0.030, std_type='UG3',
                   name='Line R14-R15')
    pp.create_line(net_cigre_lv, busR6, busR16, length_km=0.030, std_type='UG3',
                   name='Line R6-R16')
    pp.create_line(net_cigre_lv, busR9, busR17, length_km=0.030, std_type='UG3',
                   name='Line R9-R17')
    pp.create_line(net_cigre_lv, busR10, busR18, length_km=0.030, std_type='UG3',
                   name='Line R10-R18')

    pp.create_line(net_cigre_lv, busI1, busI2, length_km=0.2, std_type='UG2',
                   name='Line I1-I2')

    pp.create_line(net_cigre_lv, busC1, busC2, length_km=0.030, std_type='OH1',
                   name='Line C1-C2')
    pp.create_line(net_cigre_lv, busC2, busC3, length_km=0.030, std_type='OH1',
                   name='Line C2-C3')
    pp.create_line(net_cigre_lv, busC3, busC4, length_km=0.030, std_type='OH1',
                   name='Line C3-C4')
    pp.create_line(net_cigre_lv, busC4, busC5, length_km=0.030, std_type='OH1',
                   name='Line C4-C5')
    pp.create_line(net_cigre_lv, busC5, busC6, length_km=0.030, std_type='OH1',
                   name='Line C5-C6')
    pp.create_line(net_cigre_lv, busC6, busC7, length_km=0.030, std_type='OH1',
                   name='Line C6-C7')
    pp.create_line(net_cigre_lv, busC7, busC8, length_km=0.030, std_type='OH1',
                   name='Line C7-C8')
    pp.create_line(net_cigre_lv, busC8, busC9, length_km=0.030, std_type='OH1',
                   name='Line C8-C9')
    pp.create_line(net_cigre_lv, busC3, busC10, length_km=0.030, std_type='OH2',
                   name='Line C3-C10')
    pp.create_line(net_cigre_lv, busC10, busC11, length_km=0.030, std_type='OH2',
                   name='Line C10-C11')
    pp.create_line(net_cigre_lv, busC11, busC12, length_km=0.030, std_type='OH3',
                   name='Line C11-C12')
    pp.create_line(net_cigre_lv, busC11, busC13, length_km=0.030, std_type='OH3',
                   name='Line C11-C13')
    pp.create_line(net_cigre_lv, busC10, busC14, length_km=0.030, std_type='OH3',
                   name='Line C10-C14')
    pp.create_line(net_cigre_lv, busC5, busC15, length_km=0.030, std_type='OH2',
                   name='Line C5-C15')
    pp.create_line(net_cigre_lv, busC15, busC16, length_km=0.030, std_type='OH2',
                   name='Line C15-C16')
    pp.create_line(net_cigre_lv, busC15, busC17, length_km=0.030, std_type='OH3',
                   name='Line C15-C17')
    pp.create_line(net_cigre_lv, busC16, busC18, length_km=0.030, std_type='OH3',
                   name='Line C16-C18')
    pp.create_line(net_cigre_lv, busC8, busC19, length_km=0.030, std_type='OH3',
                   name='Line C8-C19')
    pp.create_line(net_cigre_lv, busC9, busC20, length_km=0.030, std_type='OH3',
                   name='Line C9-C20')

    # Trafos
    pp.create_transformer_from_parameters(net_cigre_lv, busR0, busR1, sn_kva=500, vn_hv_kv=20.0,
                                          vn_lv_kv=0.4, vscr_percent=1.0, vsc_percent=4.123106,
                                          pfe_kw=0.0, i0_percent=0.0, shift_degree=30.0,
                                          tp_pos=0.0, name='Trafo R0-R1')

    pp.create_transformer_from_parameters(net_cigre_lv, busI0, busI1, sn_kva=150, vn_hv_kv=20.0,
                                          vn_lv_kv=0.4, vscr_percent=1.003125, vsc_percent=4.126896,
                                          pfe_kw=0.0, i0_percent=0.0, shift_degree=30.0,
                                          tp_pos=0.0, name='Trafo I0-I1')

    pp.create_transformer_from_parameters(net_cigre_lv, busC0, busC1, sn_kva=300, vn_hv_kv=20.0,
                                          vn_lv_kv=0.4, vscr_percent=0.993750, vsc_percent=4.115529,
                                          pfe_kw=0.0, i0_percent=0.0, shift_degree=30.0,
                                          tp_pos=0.0, name='Trafo C0-C1')

    # External grid
    pp.create_ext_grid(net_cigre_lv, bus0, vm_pu=1.0, va_degree=0.0, s_sc_max_mva=100.0,
                       s_sc_min_mva=100.0, rx_max=1.0, rx_min=1.0)

    # Loads
    pp.create_load(net_cigre_lv, busR1, p_kw=190.0, q_kvar=62.449980, name='Load R1')
    pp.create_load(net_cigre_lv, busR11, p_kw=14.25, q_kvar=4.683748, name='Load R11')
    pp.create_load(net_cigre_lv, busR15, p_kw=49.4, q_kvar=16.236995, name='Load R15')
    pp.create_load(net_cigre_lv, busR16, p_kw=52.25, q_kvar=17.173744, name='Load R16')
    pp.create_load(net_cigre_lv, busR17, p_kw=33.25, q_kvar=10.928746, name='Load R17')
    pp.create_load(net_cigre_lv, busR18, p_kw=44.65, q_kvar=14.675745, name='Load R18')
    pp.create_load(net_cigre_lv, busI2, p_kw=85.0, q_kvar=52.678269, name='Load I2')
    pp.create_load(net_cigre_lv, busC1, p_kw=108.0, q_kvar=52.306787, name='Load C1')
    pp.create_load(net_cigre_lv, busC12, p_kw=18.0, q_kvar=8.717798, name='Load C12')
    pp.create_load(net_cigre_lv, busC13, p_kw=18.0, q_kvar=8.717798, name='Load C13')
    pp.create_load(net_cigre_lv, busC14, p_kw=22.5, q_kvar=10.897247, name='Load C14')
    pp.create_load(net_cigre_lv, busC17, p_kw=22.5, q_kvar=10.897247, name='Load C17')
    pp.create_load(net_cigre_lv, busC18, p_kw=7.2, q_kvar=3.487119, name='Load C18')
    pp.create_load(net_cigre_lv, busC19, p_kw=14.4, q_kvar=6.974238, name='Load C19')
    pp.create_load(net_cigre_lv, busC20, p_kw=7.2, q_kvar=3.487119, name='Load C20')

    # Switches
    pp.create_switch(net_cigre_lv, bus0, busR0, et='b', closed=True, type='CB', name='S1')
    pp.create_switch(net_cigre_lv, bus0, busI0, et='b', closed=True, type='CB', name='S2')
    pp.create_switch(net_cigre_lv, bus0, busC0, et='b', closed=True, type='CB', name='S3')

    return net_cigre_lv

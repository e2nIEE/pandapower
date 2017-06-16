# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import pandapower as pp
import pandas as pd
from numpy import nan, append


def create_synthetic_voltage_control_lv_network(network_class="rural_1"):
    """
    This function creates a LV network from M. Lindner, C. Aigner, R. Witzmann, F. Wirtz, \
    I. Berber, M. Gödde and R. Frings. "Aktuelle Musternetze zur Untersuchung von \
    Spannungsproblemen in der Niederspannung". 14. Symposium Energieinnovation TU Graz. 2014
    which are representative, synthetic grids for voltage control analysis. According to Lindner \
    the household loads are 5.1 kW and the special loads are 7.9 kW. The user is suggested to \
    assume load distribution and load profile generation.

    OPTIONAL:

        **network_class** (str, 'rural_1') - specify which type of network will be created. Must \
            be in ['rural_1', 'rural_2', 'village_1', 'village_2', 'suburb_1'].

    OUTPUT:

        **net** - returns the required synthetic voltage control lv network

    EXAMPLE:

        import networks as nw

        net = nw.create_synthetic_voltage_control_lv_network()
    """
    # process network choosing input data
    if network_class not in ['rural_1', 'rural_2', 'village_1', 'village_2', 'suburb_1']:
        raise ValueError("network_class is not in ['rural_1', 'rural_2', 'village_1', 'village_2',"
                         " 'suburb_1']")
    n_feeder = {'rural_1': [1, 4, 7], 'rural_2': [1, 3, 3, 1], 'village_1': [9, 16, 5, 9],
                'village_2': [9, 12, 5, 10], 'suburb_1': [9, 17, 5, 13, 8, 1, 9, 17, 5, 13, 4]}
    l_lines = {'rural_1': [0.26, 0.133, 0.068], 'rural_2': [3e-3, 0.076, 0.076, 0.076],
               'village_1': [0.053, 0.034, 0.08, 0.04], 'village_2': [0.021, 0.033, 0.023, 0.052],
               'suburb_1': [0.041, 0.017, 0.056, 0.018, 0.035, 0.103, 0.046, 0.019, 0.065, 0.026,
                            0.031]}
    line_types = {'rural_1': ['NAYY 4x150 SE']*2 + [('NAYY 4x150 SE', 'NAYY 4x120 SE')],
                  'rural_2': ['NAYY 4x35'] + ['NF 4x70']*3,
                  'village_1': ['NAYY 4x150 SE']*4,
                  'village_2': [('NAYY 4x70', 'NF 4x50'), ('NAYY 4x120 SE', 'NF 4x95'),
                                ('NAYY 4x95', 'NF 4x70'), ('NAYY 4x150 SE', 'NF 4x120')],
                  'suburb_1': [('NAYY 4x150 SE', 'NAYY 4x120 SE')]*3 +
                              [('NAYY 4x120 SE', 'NAYY 4x95'), 'NAYY 4x150 SE', 'NAYY 4x95'] +
                              ['NAYY 4x150 SE']*5}
    line_type_change_at = {'rural_1': [nan]*2 + [5], 'rural_2': [nan]*4, 'village_1': [nan]*4,
                           'village_2': [8, 3, 4, 5], 'suburb_1': [5, 10, 3, 6] + [nan]*7}
    trafo_type = {'rural_1': '0.16 MVA 20/0.4 kV vc', 'rural_2': '0.25 MVA 20/0.4 kV vc',
                  'village_1': '0.25 MVA 20/0.4 kV vc', 'village_2': '0.4 MVA 20/0.4 kV vc',
                  'suburb_1': '0.4 MVA 20/0.4 kV vc'}
    house_connection_length = {'rural_1': 29e-3, 'rural_2': 4e-3, 'village_1': 21e-3,
                               'village_2': 17e-3, 'suburb_1': 18e-3}
    house_connection_type = {'rural_1': 'NAYY 4x50 SE', 'rural_2': 'NAYY 4x35',
                             'village_1': 'NAYY 4x50 SE', 'village_2': 'NAYY 4x35',
                             'suburb_1': 'NAYY 4x35'}

    # create network
    net = pp.create_empty_network(name='synthetic_voltage_control_lv_network: ' + network_class)

    # create std_types
    # cable data (r, x, i_max) from www.faberkabel.de
    if network_class in ["rural_2", "village_2", "suburb_1"]:
        pp.create_std_type(net, {
                "c_nf_per_km": 202, "r_ohm_per_km": 0.869, "x_ohm_per_km": 0.085,
                "max_i_ka": 0.123, "type": "cs", "q_mm2": 35}, name="NAYY 4x35", element="line")
        if network_class != "suburb_1":
            pp.create_std_type(net, {
                    "c_nf_per_km": 17.8, "r_ohm_per_km": 0.439, "x_ohm_per_km": 0.295,
                    "max_i_ka": 0.28, "type": "ol", "q_mm2": 70}, name="NF 4x70", element="line")
            if network_class == "village_2":
                pp.create_std_type(net, {
                        "c_nf_per_km": 230, "r_ohm_per_km": 0.443, "x_ohm_per_km": 0.0823,
                        "max_i_ka": 0.179, "type": "cs", "q_mm2": 70}, name="NAYY 4x70",
                        element="line")
                data = net.std_types['line']['48-AL1/8-ST1A 0.4']
                data['q_mm2'] = 50
                pp.create_std_type(net, data, name="NF 4x50", element="line")
                data = net.std_types['line']['94-AL1/15-ST1A 0.4']
                data['q_mm2'] = 95
                pp.create_std_type(net, data, name="NF 4x95", element="line")
                pp.create_std_type(net, {
                        "c_nf_per_km": 16.2, "r_ohm_per_km": 0.274, "x_ohm_per_km": 0.31,
                        "max_i_ka": 0.4, "type": "ol", "q_mm2": 120}, name="NF 4x120", element="line")
        if network_class != "rural_2":
            pp.create_std_type(net, {
                    "c_nf_per_km": 240, "r_ohm_per_km": 0.32, "x_ohm_per_km": 0.082,
                    "max_i_ka": 0.215, "type": "cs", "q_mm2": 95}, name="NAYY 4x95", element="line")
    # trafos
    if network_class == "rural_1":
        data = net.std_types['trafo']['0.25 MVA 20/0.4 kV']
        data['sn_kva'] = 160
        data['pfe_kw'] = 0.62
        data['i0_percent'] = 0.31
        data['vscr_percent'] = data['vscr_percent'] * 4 / data['vsc_percent']
        data['vsc_percent'] = 4
        pp.create_std_type(net, data, name=trafo_type[network_class], element="trafo")
    elif network_class in ["rural_2", "village_1"]:
        data = net.std_types['trafo']['0.25 MVA 20/0.4 kV']
        data['vscr_percent'] = data['vscr_percent'] * 4 / data['vsc_percent']
        data['vsc_percent'] = 4
        pp.create_std_type(net, data, name=trafo_type[network_class], element="trafo")
    elif network_class in ["suburb_1", "village_2"]:
        data = net.std_types['trafo']['0.4 MVA 20/0.4 kV']
        data['vscr_percent'] = data['vscr_percent'] * 4 / data['vsc_percent']
        data['vsc_percent'] = 4
        pp.create_std_type(net, data, name=trafo_type[network_class], element="trafo")

    # create mv connection
    mv_bus = pp.create_bus(net, 20, name='mv bus')
    bb = pp.create_bus(net, 0.4, name='busbar')
    pp.create_ext_grid(net, mv_bus)
    pp.create_transformer(net, mv_bus, bb, std_type=trafo_type[network_class])

    # create lv network
    idx_feeder = range(len(n_feeder[network_class]))
    lv_buses = {}
    house_buses = {}
    for i in idx_feeder:
        # buses
        lv_buses[i] = pp.create_buses(net, n_feeder[network_class][i], 0.4, zone='Feeder'+str(i+1),
                                      type='m')
        house_buses[i] = pp.create_buses(net, n_feeder[network_class][i], 0.4,
                                         zone='Feeder'+str(i+1), type='n')
        # lines
        lines = pd.DataFrame()
        lines['from_bus'] = append(bb, append(lv_buses[i][:-1], lv_buses[i]))
        lines['to_bus'] = append(lv_buses[i], house_buses[i])
        if line_type_change_at[network_class][i] is nan:
            lines['std_type'] = [line_types[network_class][i]]*n_feeder[network_class][i] + \
                 [house_connection_type[network_class]]*n_feeder[network_class][i]
        else:
            lines['std_type'] = \
                 [line_types[network_class][i][0]]*line_type_change_at[network_class][i] + \
                 [line_types[network_class][i][1]]*(n_feeder[network_class][i] -
                                                    line_type_change_at[network_class][i]) + \
                 [house_connection_type[network_class]]*n_feeder[network_class][i]
        lines['length'] = [l_lines[network_class][i]]*n_feeder[network_class][i] + \
             [house_connection_length[network_class]]*n_feeder[network_class][i]

        for _, lines in lines.iterrows():
            pp.create_line(net, lines.from_bus, lines.to_bus, length_km=lines.length,
                           std_type=lines.std_type)
        # load
        for i in house_buses[i]:
            pp.create_load(net, i, p_kw=5.1)

    # direct loads and DEA
    if network_class == "rural_1":
        special_load = [(2, 4), (3, 2)]
        DER = [(2, 1, 6.9), (2, 2, 15.3), (2, 4, 29.6), (3, 4, 15.8), (3, 5, 25.3)]
    elif network_class == "rural_2":
        special_load = [(1, 1), (2, 3), (3, 2)]
        DER = [(1, 1, 29.6), (2, 3, 25.4), (3, 2, 25), (3, 3, 10)]
    elif network_class == "village_1":
        special_load = [(2, 9), (2, 12), (2, 14), (2, 16), (3, 5), (4, 3), (4, 6), (4, 8)]
        DER = [(1, 6, 29.8), (1, 8, 22.8), (2, 3, 7.9), (2, 5, 4.2), (2, 11, 16.7), (2, 15, 7.3),
               (3, 1, 31.9), (3, 3, 17.4), (3, 5, 15), (4, 1, 8.8), (4, 3, 19.6), (4, 5, 9.3),
               (4, 6, 13)]
    elif network_class == "village_2":
        special_load = []
        DER = [(1, 6, 29.8), (1, 2, 16), (1, 3, 4.6), (1, 6, 19), (1, 8, 29), (2, 1, 16),
               (2, 2, 5.2), (2, 3, 19), (2, 5, 12), (2, 10, 10), (2, 12, 8), (3, 1, 12.63),
               (3, 2, 30), (4, 3, 10), (4, 4, 33), (4, 10, 8)]
    elif network_class == "suburb_1":
        special_load = [(6, 1), (1, 4), (2, 17), (3, 5), (4, 5), (6, 1), (7, 7), (8, 17)]
        DER = [(1, 1, 9.36), (1, 2, 79.12), (7, 7, 30), (8, 7, 18.47), (8, 15, 9.54),
               (10, 10, 14.4)]
    for i in special_load:
        pp.create_load(net, lv_buses[i[0]-1][i[1]-1], p_kw=7.9)
    for i in DER:
        pp.create_sgen(net, house_buses[i[0]-1][i[1]-1], p_kw=-i[2])

    return net

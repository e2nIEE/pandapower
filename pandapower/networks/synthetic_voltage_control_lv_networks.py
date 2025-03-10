# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pandapower as pp
import pandas as pd
from numpy import nan, append
from geojson import Point, dumps


def create_synthetic_voltage_control_lv_network(network_class="rural_1"):
    """
    This function creates a LV network from M. Lindner, C. Aigner, R. Witzmann, F. Wirtz, \
    I. Berber, M. Gödde and R. Frings. "Aktuelle Musternetze zur Untersuchung von \
    Spannungsproblemen in der Niederspannung". 14. Symposium Energieinnovation TU Graz. 2014
    which are representative, synthetic grids for voltage control analysis.

    Neccessary assumptions, in addition to the paper above:

    According to Lindner, the household loads are 5.1 kw and the special loads are 7.9 kW. \
    The user is suggested to assume load distribution and load profile generation. The line
    parameters according to the given types are received from pandapower standard types and
    literatur (as stated in the code). Transformer parameters, except the given 'vk_percent',
    'sn_mva' and voltage levels, are based the pandapower standard type data.

    OPTIONAL:

        **network_class** (str, 'rural_1') - specify which type of network will be created. Must \
            be in ['rural_1', 'rural_2', 'village_1', 'village_2', 'suburb_1'].

    OUTPUT:

        **net** - returns the required synthetic voltage control lv network

    EXAMPLE:

        import pandapower.networks as pn

        net = pn.create_synthetic_voltage_control_lv_network()
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
                        "max_i_ka": 0.4, "type": "ol", "q_mm2": 120}, name="NF 4x120",
                                   element="line")
        if network_class != "rural_2":
            pp.create_std_type(net, {
                    "c_nf_per_km": 240, "r_ohm_per_km": 0.32, "x_ohm_per_km": 0.082,
                    "max_i_ka": 0.215, "type": "cs", "q_mm2": 95}, name="NAYY 4x95", element="line")
    # trafos
    if network_class == "rural_1":
        data = net.std_types['trafo']['0.25 MVA 20/0.4 kV']
        data['sn_mva'] = 0.16
        data['pfe_kw'] = 0.62
        data['i0_percent'] = 0.31
        data['vkr_percent'] = data['vkr_percent'] * 4 / data['vk_percent']
        data['vk_percent'] = 4
        pp.create_std_type(net, data, name=trafo_type[network_class], element="trafo")
    elif network_class in ["rural_2", "village_1"]:
        data = net.std_types['trafo']['0.25 MVA 20/0.4 kV']
        data['vkr_percent'] = data['vkr_percent'] * 4 / data['vk_percent']
        data['vk_percent'] = 4
        pp.create_std_type(net, data, name=trafo_type[network_class], element="trafo")
    elif network_class in ["suburb_1", "village_2"]:
        data = net.std_types['trafo']['0.4 MVA 20/0.4 kV']
        data['vkr_percent'] = data['vkr_percent'] * 4 / data['vk_percent']
        data['vk_percent'] = 4
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
            pp.create_load(net, i, p_mw=5.1e-3)

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
        pp.create_load(net, lv_buses[i[0]-1][i[1]-1], p_mw=7.9e-3)
    for i in DER:
        pp.create_sgen(net, house_buses[i[0]-1][i[1]-1], p_mw=i[2]*1e-3)

    # set bus geo data
    bus_geo = {
        "rural_1": {0: [0.0, 0.0], 1: [0.0, 1.0], 2: [-1.6666666667, 2.0], 3: [-1.6666666667, 3.0],
                    4: [-0.1666666667, 2.0], 5: [-0.6666666667, 3.0], 6: [-1.1666666667, 4.0], 7: [-1.6666666667, 5.0],
                    8: [0.3333333333, 3.0], 9: [-0.1666666667, 4.0], 10: [-0.6666666667, 5.0], 11: [-1.6666666667, 6.0],
                    12: [1.8333333333, 2.0], 13: [1.3333333333, 3.0], 14: [0.8333333333, 4.0], 15: [0.3333333333, 5.0],
                    16: [-0.1666666667, 6.0], 17: [-0.6666666667, 7.0], 18: [-1.1666666667, 8.0],
                    19: [2.3333333333, 3.0], 20: [1.8333333333, 4.0], 21: [1.3333333333, 5.0], 22: [0.8333333333, 6.0],
                    23: [0.3333333333, 7.0], 24: [-0.1666666667, 8.0], 25: [-1.1666666667, 9.0]
                    },
        "rural_2": {0: [0.0, 0.0], 1: [0.0, 1.0], 2: [-2.5, 2.0], 3: [-2.5, 3.0], 4: [-1.0, 2.0], 5: [-1.5, 3.0],
                    6: [-2.0, 4.0], 7: [-0.5, 3.0], 8: [-1.0, 4.0], 9: [-2.0, 5.0], 10: [1.0, 2.0], 11: [0.5, 3.0],
                    12: [0.0, 4.0], 13: [1.5, 3.0], 14: [1.0, 4.0], 15: [0.0, 5.0], 16: [2.5, 2.0], 17: [2.5, 3.0]
                    },
        "village_1": {0: [0.0, 0.0], 1: [0.0, 1.0], 2: [-3.0, 2.0], 3: [-3.5, 3.0], 4: [-4.0, 4.0], 5: [-4.5, 5.0],
                      6: [-5.0, 6.0], 7: [-5.5, 7.0], 8: [-6.0, 8.0], 9: [-6.5, 9.0], 10: [-7.0, 10.0], 11: [-2.5, 3.0],
                      12: [-3.0, 4.0], 13: [-3.5, 5.0], 14: [-4.0, 6.0], 15: [-4.5, 7.0], 16: [-5.0, 8.0],
                      17: [-5.5, 9.0], 18: [-6.0, 10.0], 19: [-7.0, 11.0], 20: [-1.0, 2.0], 21: [-1.5, 3.0],
                      22: [-2.0, 4.0], 23: [-2.5, 5.0], 24: [-3.0, 6.0], 25: [-3.5, 7.0], 26: [-4.0, 8.0],
                      27: [-4.5, 9.0], 28: [-5.0, 10.0], 29: [-5.5, 11.0], 30: [-6.0, 12.0], 31: [-6.5, 13.0],
                      32: [-7.0, 14.0], 33: [-7.5, 15.0], 34: [-8.0, 16.0], 35: [-8.5, 17.0], 36: [-0.5, 3.0],
                      37: [-1.0, 4.0], 38: [-1.5, 5.0], 39: [-2.0, 6.0], 40: [-2.5, 7.0], 41: [-3.0, 8.0],
                      42: [-3.5, 9.0], 43: [-4.0, 10.0], 44: [-4.5, 11.0], 45: [-5.0, 12.0], 46: [-5.5, 13.0],
                      47: [-6.0, 14.0], 48: [-6.5, 15.0], 49: [-7.0, 16.0], 50: [-7.5, 17.0], 51: [-8.5, 18.0],
                      52: [1.0, 2.0], 53: [0.5, 3.0], 54: [0.0, 4.0], 55: [-0.5, 5.0], 56: [-1.0, 6.0], 57: [1.5, 3.0],
                      58: [1.0, 4.0], 59: [0.5, 5.0], 60: [0.0, 6.0], 61: [-1.0, 7.0], 62: [3.0, 2.0], 63: [2.5, 3.0],
                      64: [2.0, 4.0], 65: [1.5, 5.0], 66: [1.0, 6.0], 67: [0.5, 7.0], 68: [0.0, 8.0], 69: [-0.5, 9.0],
                      70: [-1.0, 10.0], 71: [3.5, 3.0], 72: [3.0, 4.0], 73: [2.5, 5.0], 74: [2.0, 6.0], 75: [1.5, 7.0],
                      76: [1.0, 8.0], 77: [0.5, 9.0], 78: [0.0, 10.0], 79: [-1.0, 11.0]
                      },
        "village_2": {0: [0.0, 0.0], 1: [0.0, 1.0], 2: [-3.0, 2.0], 3: [-3.5, 3.0], 4: [-4.0, 4.0], 5: [-4.5, 5.0],
                      6: [-5.0, 6.0], 7: [-5.5, 7.0], 8: [-6.0, 8.0], 9: [-6.5, 9.0], 10: [-7.0, 10.0], 11: [-2.5, 3.0],
                      12: [-3.0, 4.0], 13: [-3.5, 5.0], 14: [-4.0, 6.0], 15: [-4.5, 7.0], 16: [-5.0, 8.0],
                      17: [-5.5, 9.0], 18: [-6.0, 10.0], 19: [-7.0, 11.0], 20: [-1.0, 2.0], 21: [-1.5, 3.0],
                      22: [-2.0, 4.0], 23: [-2.5, 5.0], 24: [-3.0, 6.0], 25: [-3.5, 7.0], 26: [-4.0, 8.0],
                      27: [-4.5, 9.0], 28: [-5.0, 10.0], 29: [-5.5, 11.0], 30: [-6.0, 12.0], 31: [-6.5, 13.0],
                      32: [-0.5, 3.0], 33: [-1.0, 4.0], 34: [-1.5, 5.0], 35: [-2.0, 6.0], 36: [-2.5, 7.0],
                      37: [-3.0, 8.0], 38: [-3.5, 9.0], 39: [-4.0, 10.0], 40: [-4.5, 11.0], 41: [-5.0, 12.0],
                      42: [-5.5, 13.0], 43: [-6.5, 14.0], 44: [1.0, 2.0], 45: [0.5, 3.0], 46: [0.0, 4.0],
                      47: [-0.5, 5.0], 48: [-1.0, 6.0], 49: [1.5, 3.0], 50: [1.0, 4.0], 51: [0.5, 5.0], 52: [0.0, 6.0],
                      53: [-1.0, 7.0], 54: [3.0, 2.0], 55: [2.5, 3.0], 56: [2.0, 4.0], 57: [1.5, 5.0], 58: [1.0, 6.0],
                      59: [0.5, 7.0], 60: [0.0, 8.0], 61: [-0.5, 9.0], 62: [-1.0, 10.0], 63: [-1.5, 11.0],
                      64: [3.5, 3.0], 65: [3.0, 4.0], 66: [2.5, 5.0], 67: [2.0, 6.0], 68: [1.5, 7.0], 69: [1.0, 8.0],
                      70: [0.5, 9.0], 71: [0.0, 10.0], 72: [-0.5, 11.0], 73: [-1.5, 12.0]
                      },
        "suburb_1": {0: [0.0, 0.0], 1: [0.0, 1.0], 2: [-9.5, 2.0], 3: [-10.0, 3.0], 4: [-10.5, 4.0], 5: [-11.0, 5.0],
                     6: [-11.5, 6.0], 7: [-12.0, 7.0], 8: [-12.5, 8.0], 9: [-13.0, 9.0], 10: [-13.5, 10.0],
                     11: [-9.0, 3.0], 12: [-9.5, 4.0], 13: [-10.0, 5.0], 14: [-10.5, 6.0], 15: [-11.0, 7.0],
                     16: [-11.5, 8.0], 17: [-12.0, 9.0], 18: [-12.5, 10.0], 19: [-13.5, 11.0], 20: [-7.5, 2.0],
                     21: [-8.0, 3.0], 22: [-8.5, 4.0], 23: [-9.0, 5.0], 24: [-9.5, 6.0], 25: [-10.0, 7.0],
                     26: [-10.5, 8.0], 27: [-11.0, 9.0], 28: [-11.5, 10.0], 29: [-12.0, 11.0], 30: [-12.5, 12.0],
                     31: [-13.0, 13.0], 32: [-13.5, 14.0], 33: [-14.0, 15.0], 34: [-14.5, 16.0], 35: [-15.0, 17.0],
                     36: [-15.5, 18.0], 37: [-7.0, 3.0], 38: [-7.5, 4.0], 39: [-8.0, 5.0], 40: [-8.5, 6.0],
                     41: [-9.0, 7.0], 42: [-9.5, 8.0], 43: [-10.0, 9.0], 44: [-10.5, 10.0], 45: [-11.0, 11.0],
                     46: [-11.5, 12.0], 47: [-12.0, 13.0], 48: [-12.5, 14.0], 49: [-13.0, 15.0], 50: [-13.5, 16.0],
                     51: [-14.0, 17.0], 52: [-14.5, 18.0], 53: [-15.5, 19.0], 54: [-5.5, 2.0], 55: [-6.0, 3.0],
                     56: [-6.5, 4.0], 57: [-7.0, 5.0], 58: [-7.5, 6.0], 59: [-5.0, 3.0], 60: [-5.5, 4.0],
                     61: [-6.0, 5.0], 62: [-6.5, 6.0], 63: [-7.5, 7.0], 64: [-3.5, 2.0], 65: [-4.0, 3.0],
                     66: [-4.5, 4.0], 67: [-5.0, 5.0], 68: [-5.5, 6.0], 69: [-6.0, 7.0], 70: [-6.5, 8.0],
                     71: [-7.0, 9.0], 72: [-7.5, 10.0], 73: [-8.0, 11.0], 74: [-8.5, 12.0], 75: [-9.0, 13.0],
                     76: [-9.5, 14.0], 77: [-3.0, 3.0], 78: [-3.5, 4.0], 79: [-4.0, 5.0], 80: [-4.5, 6.0],
                     81: [-5.0, 7.0], 82: [-5.5, 8.0], 83: [-6.0, 9.0], 84: [-6.5, 10.0], 85: [-7.0, 11.0],
                     86: [-7.5, 12.0], 87: [-8.0, 13.0], 88: [-8.5, 14.0], 89: [-9.5, 15.0], 90: [-1.5, 2.0],
                     91: [-2.0, 3.0], 92: [-2.5, 4.0], 93: [-3.0, 5.0], 94: [-3.5, 6.0], 95: [-4.0, 7.0],
                     96: [-4.5, 8.0], 97: [-5.0, 9.0], 98: [-1.0, 3.0], 99: [-1.5, 4.0], 100: [-2.0, 5.0],
                     101: [-2.5, 6.0], 102: [-3.0, 7.0], 103: [-3.5, 8.0], 104: [-4.0, 9.0], 105: [-5.0, 10.0],
                     106: [0.0, 2.0], 107: [0.0, 3.0], 108: [1.5, 2.0], 109: [1.0, 3.0], 110: [0.5, 4.0],
                     111: [0.0, 5.0], 112: [-0.5, 6.0], 113: [-1.0, 7.0], 114: [-1.5, 8.0], 115: [-2.0, 9.0],
                     116: [-2.5, 10.0], 117: [2.0, 3.0], 118: [1.5, 4.0], 119: [1.0, 5.0], 120: [0.5, 6.0],
                     121: [0.0, 7.0], 122: [-0.5, 8.0], 123: [-1.0, 9.0], 124: [-1.5, 10.0], 125: [-2.5, 11.0],
                     126: [3.5, 2.0], 127: [3.0, 3.0], 128: [2.5, 4.0], 129: [2.0, 5.0], 130: [1.5, 6.0],
                     131: [1.0, 7.0], 132: [0.5, 8.0], 133: [0.0, 9.0], 134: [-0.5, 10.0], 135: [-1.0, 11.0],
                     136: [-1.5, 12.0], 137: [-2.0, 13.0], 138: [-2.5, 14.0], 139: [-3.0, 15.0], 140: [-3.5, 16.0],
                     141: [-4.0, 17.0], 142: [-4.5, 18.0], 143: [4.0, 3.0], 144: [3.5, 4.0], 145: [3.0, 5.0],
                     146: [2.5, 6.0], 147: [2.0, 7.0], 148: [1.5, 8.0], 149: [1.0, 9.0], 150: [0.5, 10.0],
                     151: [0.0, 11.0], 152: [-0.5, 12.0], 153: [-1.0, 13.0], 154: [-1.5, 14.0], 155: [-2.0, 15.0],
                     156: [-2.5, 16.0], 157: [-3.0, 17.0], 158: [-3.5, 18.0], 159: [-4.5, 19.0], 160: [5.5, 2.0],
                     161: [5.0, 3.0], 162: [4.5, 4.0], 163: [4.0, 5.0], 164: [3.5, 6.0], 165: [6.0, 3.0],
                     166: [5.5, 4.0], 167: [5.0, 5.0], 168: [4.5, 6.0], 169: [3.5, 7.0], 170: [7.5, 2.0],
                     171: [7.0, 3.0], 172: [6.5, 4.0], 173: [6.0, 5.0], 174: [5.5, 6.0], 175: [5.0, 7.0],
                     176: [4.5, 8.0], 177: [4.0, 9.0], 178: [3.5, 10.0], 179: [3.0, 11.0], 180: [2.5, 12.0],
                     181: [2.0, 13.0], 182: [1.5, 14.0], 183: [8.0, 3.0], 184: [7.5, 4.0], 185: [7.0, 5.0],
                     186: [6.5, 6.0], 187: [6.0, 7.0], 188: [5.5, 8.0], 189: [5.0, 9.0], 190: [4.5, 10.0],
                     191: [4.0, 11.0], 192: [3.5, 12.0], 193: [3.0, 13.0], 194: [2.5, 14.0], 195: [1.5, 15.0],
                     196: [9.5, 2.0], 197: [9.0, 3.0], 198: [8.5, 4.0], 199: [8.0, 5.0], 200: [10.0, 3.0],
                     201: [9.5, 4.0], 202: [9.0, 5.0], 203: [8.0, 6.0]
                     }
    }
    net.bus.geo = net.bus.apply(
        lambda row: dumps(Point(bus_geo[network_class][row.name])),
        axis=1
    )
    return net

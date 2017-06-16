# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import copy
import warnings

import numpy as np
import pytest

import pandapower as pp
import pandapower.networks as nw
from pandapower.diagnostic_reports import DiagnosticReports


@pytest.fixture(scope='module')
def test_net():
    net = nw.example_multivoltage()
    return net


@pytest.fixture(scope='module')
def diag_params():
    diag_params = {
        "overload_scaling_factor": 0.001,
        "lines_min_length_km": 0,
        "lines_min_z_ohm": 0,
        "nom_voltage_tolerance": 0.3,
        "numba_tolerance": 1e-5}
    return diag_params


@pytest.fixture(scope='module')
def report_methods():
    report_methods = {
    "missing_bus_indeces": "diag_report.report_missing_bus_indeces()",
    "disconnected_elements": "diag_report.report_disconnected_elements()",
    "different_voltage_levels_connected": "diag_report.report_different_voltage_levels_connected()",
    "lines_with_impedance_close_to_zero": "diag_report.report_lines_with_impedance_close_to_zero()",
    "nominal_voltages_dont_match": "diag_report.report_nominal_voltages_dont_match()",
    "invalid_values": "diag_report.report_invalid_values()",
    "overload": "diag_report.report_overload()",
    "multiple_voltage_controlling_elements_per_bus" : "diag_report.report_multiple_voltage_controlling_elements_per_bus()",
    "wrong_switch_configuration": "diag_report.report_wrong_switch_configuration()",
    "no_ext_grid": "diag_report.report_no_ext_grid()",
    "wrong_reference_system": "diag_report.report_wrong_reference_system()",
    "deviation_from_std_type": "diag_report.report_deviation_from_std_type()",
    "numba_comparison": "diag_report.report_numba_comparison()",
    "parallel_switches": "diag_report.report_parallel_switches()"}
    return report_methods


def test_no_issues(diag_params, report_methods):
    net = nw.example_simple()
    diag_params = copy.deepcopy(diag_params)
    report_methods = copy.deepcopy(report_methods)

    diag_results = pp.diagnostic(net, report_style=None)
    assert diag_results == {}
    for bool_value in [True, False]:
        for check_function in report_methods.keys():
            diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
            report_check = None
            try:
                eval(report_methods[check_function])
                report_check = True
            except:
                report_check = False
            assert report_check


class TestInvalidValues:

    def test_greater_zero(self, test_net, diag_params, report_methods):
        net = copy.deepcopy(test_net)
        check_function = 'invalid_values'
        diag_params = copy.deepcopy(diag_params)
        report_methods = copy.deepcopy(report_methods)
        net.bus.loc[42, 'vn_kv'] = '-1'
        net.line.loc[7, 'length_km'] = -1
        net.line.loc[8, 'max_i_ka'] = 0
        net.trafo.loc[0, 'vsc_percent'] = 0.0
        net.trafo.loc[0, 'sn_kva'] = None
        net.trafo.loc[0, 'vn_hv_kv'] = -1.5
        net.trafo.loc[0, 'vn_lv_kv'] = False
        net.trafo3w.loc[0, 'vsc_hv_percent'] = 2.3
        net.trafo3w.loc[0, 'vsc_mv_percent'] = np.nan
        net.trafo3w.loc[0, 'vsc_lv_percent'] = 0.0
        net.trafo3w.loc[0, 'sn_hv_kva'] = 11
        net.trafo3w.loc[0, 'sn_mv_kva'] = 'a'
        net.trafo3w.loc[0, 'vn_hv_kv'] = -1.5
        net.trafo3w.loc[0, 'vn_mv_kv'] = -1.5
        net.trafo3w.loc[0, 'vn_lv_kv'] = False
        net.ext_grid.loc[0, 'vm_pu'] = True

        check_result = pp.invalid_values(net)
        if check_result:
            diag_results = {check_function: check_result}
        else:
            diag_results = {}

        assert diag_results[check_function] == \
        {'bus': [(42, 'vn_kv', '-1', '>0')],
         'ext_grid': [(0, 'vm_pu', True, '>0')],
         'line': [(7, 'length_km', -1.0, '>0'), (8, 'max_i_ka', 0.0, '>0')],
         'trafo': [(0, 'sn_kva', 'nan', '>0'), (0, 'vn_hv_kv', -1.5, '>0'),
                   (0, 'vn_lv_kv', False, '>0'), (0, 'vsc_percent', 0.0, '>0')],
         'trafo3w': [(0, 'sn_mv_kva', 'a', '>0'), (0, 'vn_hv_kv', -1.5, '>0'),
                     (0, 'vn_mv_kv', -1.5, '>0'), (0, 'vn_lv_kv', False, '>0'),
                     (0, 'vsc_mv_percent', 'nan', '>0'), (0, 'vsc_lv_percent', 0.0, '>0')]}

        for bool_value in [True, False]:
            diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
            report_check = None
            try:
                eval(report_methods[check_function])
                report_check = True
            except:
                report_check = False
            assert report_check


    def test_greater_equal_zero(self, test_net, diag_params, report_methods):
        net = copy.deepcopy(test_net)
        check_function = 'invalid_values'
        diag_params = copy.deepcopy(diag_params)
        report_methods = copy.deepcopy(report_methods)
        net.line.loc[7, 'r_ohm_per_km'] = -1
        net.line.loc[8, 'x_ohm_per_km'] = None
        net.line.loc[8, 'c_nf_per_km'] = '0'
        net.trafo.loc[0, 'vscr_percent'] = '-1'
        net.trafo.loc[0, 'pfe_kw'] = -1.5
        net.trafo.loc[0, 'i0_percent'] = -0.001
        net.trafo3w.loc[0, 'vscr_hv_percent'] = True
        net.trafo3w.loc[0, 'vscr_mv_percent'] = False
        net.trafo3w.loc[0, 'vscr_lv_percent'] = 1
        net.trafo3w.loc[0, 'pfe_kw'] = '2'
        net.trafo3w.loc[0, 'i0_percent'] = 10
        net.load.loc[0, 'scaling'] = -0.1
        net.load.loc[1, 'scaling'] = 0
        net.load.loc[2, 'scaling'] = 1
        net.load.loc[3, 'scaling'] = '1'
        net.gen.loc[0, 'scaling'] = None
        net.sgen.loc[0, 'scaling'] = False

        check_result = pp.invalid_values(net)
        if check_result:
            diag_results = {check_function: check_result}
        else:
            diag_results = {}
        assert diag_results[check_function] == \
        {'line': [(7, 'r_ohm_per_km', -1.0, '>=0'), (8, 'x_ohm_per_km', 'nan', '>=0'),
                  (8, 'c_nf_per_km', '0', '>=0')],
         'trafo': [(0, 'vscr_percent', '-1', '>=0'), (0, 'pfe_kw', -1.5, '>=0'),
                   (0, 'i0_percent', -0.001, '>=0')],
         'trafo3w': [(0, 'vscr_hv_percent', True, '>=0'), (0, 'vscr_mv_percent', False, '>=0'),
                     (0, 'pfe_kw', '2', '>=0')],
         'gen': [(0, 'scaling', 'nan', '>=0')],
         'load': [(0, 'scaling', -0.1, '>=0'), (3, 'scaling', '1', '>=0')],
         'sgen': [(0, 'scaling', False, '>=0')]}

        for bool_value in [True, False]:
            diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
            report_check = None
            try:
                eval(report_methods[check_function])
                report_check = True
            except:
                report_check = False
            assert report_check
#
#
#    #def test_smaller_zero(self, net):           # check_smaller_zero currently not in use
#        #pass
#
#    #def test_smaller_equal_zero(self, net):     # check_smaller_equal_zero currently not in use
#        #pass
#
#
    def test_boolean(self, test_net, diag_params, report_methods):
        net = copy.deepcopy(test_net)
        check_function = 'invalid_values'
        diag_params = copy.deepcopy(diag_params)
        report_methods = copy.deepcopy(report_methods)
        net.sgen.loc[0, 'in_service'] = 0
        net.sgen.loc[1, 'in_service'] = 0.0
        net.sgen.loc[2, 'in_service'] = '0'
        net.sgen.loc[3, 'in_service'] = '0.0'
        net.sgen.loc[4, 'in_service'] = 1
        net.gen.loc[0, 'in_service'] = '1'
        net.load.loc[0, 'in_service'] = 10
        net.line.loc[0, 'in_service'] = -1
        net.bus.loc[0, 'in_service'] = 'no'
        net.trafo.loc[0, 'in_service'] = 'True'
        net.trafo3w.loc[0, 'in_service'] = None
        net.switch.loc[0, 'closed'] = 0
        net.switch.loc[1, 'closed'] = 'False'
        net.switch.loc[2, 'closed'] = False
        net.switch.loc[3, 'closed'] = 'False'
        net.switch.loc[4, 'closed'] = None
        net.switch.loc[5, 'closed'] = 10

        check_result = pp.invalid_values(net)
        if check_result:
            diag_results = {check_function: check_result}
        else:
            diag_results = {}
        assert diag_results[check_function] == \
        {'bus': [(0, 'in_service', 'no', 'boolean')],
         'gen': [(0, 'in_service', '1', 'boolean')],
         'sgen': [(2, 'in_service', '0', 'boolean'), (3, 'in_service', '0.0', 'boolean')],
         'switch': [(1, 'closed', 'False', 'boolean'), (3, 'closed', 'False', 'boolean'),
                    (4, 'closed', 'None', 'boolean'), (5, 'closed', 10, 'boolean')],
         'trafo': [(0, 'in_service', 'True', 'boolean')],
         'trafo3w': [(0, 'in_service', 'nan', 'boolean')]}

        for bool_value in [True, False]:
            diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
            report_check = None
            try:
                eval(report_methods[check_function])
                report_check = True
            except:
                report_check = False
            assert report_check


    def test_pos_int(self, test_net, diag_params, report_methods):
        net = copy.deepcopy(test_net)
        check_function = 'invalid_values'
        diag_params = copy.deepcopy(diag_params)
        report_methods = copy.deepcopy(report_methods)
        net.line.loc[7, 'from_bus'] = 1
        net.line.loc[8, 'to_bus'] = '2'
        net.trafo.loc[0, 'hv_bus'] = False
        net.trafo.loc[0, 'lv_bus'] = None
        net.trafo3w.loc[0, 'hv_bus'] = False
        net.trafo3w.loc[0, 'mv_bus'] = 0.5
        net.trafo3w.loc[0, 'lv_bus'] = 2
        net.load.loc[0, 'bus'] = True
        net.sgen.loc[0, 'bus'] = 1.5
        net.gen.loc[0, 'bus'] = np.nan
        net.ext_grid.loc[0, 'bus'] = -2.5
        net.switch.loc[0, 'bus'] = None
        net.switch.loc[0, 'element'] = -1.5

        check_result = pp.invalid_values(net)
        if check_result:
            diag_results = {check_function: check_result}
        else:
            diag_results = {}
        assert diag_results[check_function] == \
        {'ext_grid': [(0, 'bus', -2.5, 'positive_integer')],
         'gen': [(0, 'bus', 'nan', 'positive_integer')],
         'line': [(8, 'to_bus', '2', 'positive_integer')],
         'load': [(0, 'bus', True, 'positive_integer')],
         'sgen': [(0, 'bus', 1.5, 'positive_integer')],
         'switch': [(0, 'bus', 'nan', 'positive_integer'),
                    (0, 'element', -1.5, 'positive_integer')],
         'trafo': [(0, 'hv_bus', False, 'positive_integer'),
                   (0, 'lv_bus', 'nan', 'positive_integer')],
         'trafo3w': [(0, 'hv_bus', False, 'positive_integer'),
                     (0, 'mv_bus', 0.5, 'positive_integer')]}

        for bool_value in [True, False]:
            diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
            report_check = None
            try:
                eval(report_methods[check_function])
                report_check = True
            except:
                report_check = False
            assert report_check


    def test_number(self, test_net, diag_params, report_methods):
        net = copy.deepcopy(test_net)
        check_function = 'invalid_values'
        diag_params = copy.deepcopy(diag_params)
        report_methods = copy.deepcopy(report_methods)
        net.load.loc[0, 'p_kw'] = '1000'
        net.load.loc[1, 'q_kvar'] = None
        net.gen.loc[0, 'p_kw'] = False
        net.sgen.loc[0, 'p_kw'] = -1.5
        net.sgen.loc[1, 'q_kvar'] = np.nan
        net.ext_grid.loc[0, 'va_degree'] = 13.55

        check_result = pp.invalid_values(net)
        if check_result:
            diag_results = {check_function: check_result}
        else:
            diag_results = {}
        assert diag_results[check_function] == \
        {'gen': [(0, 'p_kw', False, 'number')],
         'load': [(0, 'p_kw', '1000', 'number'), (1, 'q_kvar', 'nan', 'number')],
         'sgen': [(1, 'q_kvar', 'nan', 'number')]}

        for bool_value in [True, False]:
            diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
            report_check = None
            try:
                eval(report_methods[check_function])
                report_check = True
            except:
                report_check = False
            assert report_check


    def test_between_zero_and_one(self, test_net, diag_params, report_methods):
        net = copy.deepcopy(test_net)
        check_function = 'invalid_values'
        diag_params = copy.deepcopy(diag_params)
        report_methods = copy.deepcopy(report_methods)
        net.line.loc[0, 'df'] = 1.5

        check_result = pp.invalid_values(net)
        if check_result:
            diag_results = {check_function: check_result}
        else:
            diag_results = {}
        assert diag_results[check_function] == \
        {'line': [(0, 'df', 1.5, '0to1')]}

        for bool_value in [True, False]:
            diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
            report_check = None
            try:
                eval(report_methods[check_function])
                report_check = True
            except:
                report_check = False
            assert report_check


    def test_switch_type(self, test_net, diag_params, report_methods):
        net = copy.deepcopy(test_net)
        check_function = 'invalid_values'
        diag_params = copy.deepcopy(diag_params)
        report_methods = copy.deepcopy(report_methods)
        net.switch.loc[0, 'et'] = 'bus'
        net.switch.loc[1, 'et'] = 1
        net.switch.loc[2, 'et'] = None
        net.switch.loc[3, 'et'] = True
        net.switch.loc[4, 'et'] = 't'

        check_result = pp.invalid_values(net)
        if check_result:
            diag_results = {check_function: check_result}
        else:
            diag_results = {}
        assert diag_results[check_function] == \
        {'switch': [(0, 'et', 'bus', 'switch_type'),
                    (1, 'et', 1, 'switch_type'),
                    (2, 'et', 'None', 'switch_type'),
                    (3, 'et', True, 'switch_type')]}

        for bool_value in [True, False]:
            diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
            report_check = None
            try:
                eval(report_methods[check_function])
                report_check = True
            except:
                report_check = False
            assert report_check


def test_no_ext_grid(test_net, diag_params, report_methods):
    net = copy.deepcopy(test_net)
    check_function = 'no_ext_grid'
    diag_params = copy.deepcopy(diag_params)
    report_methods = copy.deepcopy(report_methods)
    net.ext_grid = net.ext_grid.drop(0)
    check_result = pp.no_ext_grid(net)
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert diag_results[check_function] == True


def test_multiple_voltage_controlling_elements_per_bus(test_net, diag_params, report_methods):
    net = copy.deepcopy(test_net)
    check_function = 'multiple_voltage_controlling_elements_per_bus'
    diag_params = copy.deepcopy(diag_params)
    report_methods = copy.deepcopy(report_methods)
    net.gen.bus.at[0] = 0
    pp.create_ext_grid(net, 1)
    net.ext_grid.bus.at[1] = 0

    check_result = pp.multiple_voltage_controlling_elements_per_bus(net)
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert diag_results[check_function] == \
    {'buses_with_gens_and_ext_grids': [0],
     'buses_with_mult_ext_grids': [0]}

    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check


def test_overload(test_net, diag_params, report_methods):
    net = copy.deepcopy(test_net)
    check_function = 'overload'
    diag_params = copy.deepcopy(diag_params)
    report_methods = copy.deepcopy(report_methods)
    net.load.p_kw.at[4] *= 1000

    check_result = pp.overload(net, diag_params['overload_scaling_factor'])
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert diag_results[check_function] == \
    {'generation': 'uncertain', 'load': True}

    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check

    net.load.p_kw.at[4] /= 1000
    net.gen.p_kw *= 1000
    check_result = pp.overload(net, diag_params['overload_scaling_factor'])
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert diag_results[check_function] == \
    {'generation': True, 'load': 'uncertain'}

    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check


def test_switch_configuration(test_net, diag_params, report_methods):
    net = copy.deepcopy(test_net)
    check_function = 'wrong_switch_configuration'
    diag_params = copy.deepcopy(diag_params)
    report_methods = copy.deepcopy(report_methods)
    net.switch.closed.at[0] = 0
    net.switch.closed.at[2] = 0
    check_result = pp.wrong_switch_configuration(net)
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert diag_results[check_function] == True
    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check

    net.switch.closed.at[0] = 1
    net.switch.closed.at[2] = 1
    net.switch.closed = 0
    # this will raise the warning "Matrix is exactly singular" -> ignore
    warnings.simplefilter("ignore")
    check_result = pp.wrong_switch_configuration(net)
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert diag_results[check_function] == True
    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check
    net.switch.closed = 1
    net.load.p_kw.at[4] *= 1000
    check_result = pp.wrong_switch_configuration(net)
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert diag_results[check_function] == 'uncertain'
    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check


def test_different_voltage_levels_connected(test_net, diag_params, report_methods):
    net = copy.deepcopy(test_net)
    check_function = 'different_voltage_levels_connected'
    diag_params = copy.deepcopy(diag_params)
    report_methods = copy.deepcopy(report_methods)
    pp.create_switch(net, 41, 45, et = 'b')
    net.bus.vn_kv.loc[38] = 30
    check_result = pp.different_voltage_levels_connected(net)
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert diag_results[check_function] == \
    {'lines': [6, 7], 'switches': [88]}

    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check


def test_lines_with_impedance_close_to_zero(test_net, diag_params, report_methods):
    net = copy.deepcopy(test_net)
    check_function = 'lines_with_impedance_close_to_zero'
    diag_params = copy.deepcopy(diag_params)
    report_methods = copy.deepcopy(report_methods)
    net.line.length_km.at[0] = 0
    net.line.r_ohm_per_km.at[1] = 0
    net.line.x_ohm_per_km.at[1] = 0
    net.line.r_ohm_per_km.at[2] = 0
    net.line.x_ohm_per_km.at[3] = 0
    net.line.length_km.at[4] = 0
    net.line.r_ohm_per_km.at[4] = 0
    net.line.x_ohm_per_km.at[4] = 0
    check_result = pp.lines_with_impedance_close_to_zero(net,
                                                         diag_params['lines_min_length_km'],
                                                         diag_params['lines_min_z_ohm'])
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert diag_results[check_function] == [0, 1, 4]

    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check

    net.line.length_km.at[0] = 0.001
    net.line.r_ohm_per_km.at[1] = 0.001
    net.line.x_ohm_per_km.at[1] = 0.001
    net.line.r_ohm_per_km.at[2] = 0.001
    net.line.x_ohm_per_km.at[3] = 0.001
    net.line.length_km.at[4] = 1
    net.line.r_ohm_per_km.at[4] = 0.001
    net.line.x_ohm_per_km.at[4] = 0.001
    check_result = pp.lines_with_impedance_close_to_zero(net,
                                                         diag_params['lines_min_length_km'],
                                                         diag_params['lines_min_z_ohm'])
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert check_function not in diag_results

    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check


def test_nominal_voltages_dont_match(test_net, diag_params, report_methods):
    net = copy.deepcopy(test_net)
    check_function = 'nominal_voltages_dont_match'
    diag_params = copy.deepcopy(diag_params)
    report_methods = copy.deepcopy(report_methods)
    trafo_copy = copy.deepcopy(net.trafo)
    trafo3w_copy = copy.deepcopy(net.trafo3w)
    net.trafo.hv_bus.at[0] = trafo_copy.lv_bus.at[0]
    net.trafo.lv_bus.at[0] = trafo_copy.hv_bus.at[0]
    check_result = pp.nominal_voltages_dont_match(net, diag_params['nom_voltage_tolerance'])
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert diag_results[check_function] == \
    {'trafo': {'hv_lv_swapped': [0]}}

    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check

    net.trafo = copy.deepcopy(trafo_copy)
    net.trafo.vn_hv_kv.at[0] *= 1.31
    net.trafo.vn_lv_kv.at[0] *= 1.31
    check_result = pp.nominal_voltages_dont_match(net, diag_params['nom_voltage_tolerance'])
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert diag_results[check_function] == \
    {'trafo': {'hv_bus': [0], 'lv_bus': [0]}}

    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check

    net.trafo = copy.deepcopy(trafo_copy)
    net.trafo.vn_hv_kv.at[0] *= 0.69
    net.trafo.vn_lv_kv.at[0] *= 0.69
    check_result = pp.nominal_voltages_dont_match(net, diag_params['nom_voltage_tolerance'])
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert diag_results[check_function] == \
    {'trafo': {'hv_bus': [0], 'lv_bus': [0]}}

    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check

    net.trafo = copy.deepcopy(trafo_copy)
    net.trafo.vn_hv_kv.at[0] *= 1.29
    net.trafo.vn_lv_kv.at[0] *= 1.29
    check_result = pp.nominal_voltages_dont_match(net, diag_params['nom_voltage_tolerance'])
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert check_function not in diag_results

    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check

    net.trafo = copy.deepcopy(trafo_copy)
    net.trafo.vn_hv_kv.at[0] *= 0.71
    net.trafo.vn_lv_kv.at[0] *= 0.71
    check_result = pp.nominal_voltages_dont_match(net, diag_params['nom_voltage_tolerance'])
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert check_function not in diag_results

    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check

    net.trafo = copy.deepcopy(trafo_copy)

    net.trafo3w.hv_bus.at[0] = trafo3w_copy.mv_bus.at[0]
    net.trafo3w.mv_bus.at[0] = trafo3w_copy.lv_bus.at[0]
    net.trafo3w.lv_bus.at[0] = trafo3w_copy.hv_bus.at[0]
    check_result = pp.nominal_voltages_dont_match(net, diag_params['nom_voltage_tolerance'])
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert diag_results[check_function] == \
    {'trafo3w': {'connectors_swapped_3w': [0]}}

    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check

    net.trafo3w = copy.deepcopy(trafo3w_copy)
    net.trafo3w.vn_hv_kv.at[0] *= 1.31
    net.trafo3w.vn_mv_kv.at[0] *= 1.31
    net.trafo3w.vn_lv_kv.at[0] *= 1.31
    check_result = pp.nominal_voltages_dont_match(net, diag_params['nom_voltage_tolerance'])
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert diag_results[check_function] == \
    {'trafo3w': {'hv_bus': [0], 'lv_bus': [0], 'mv_bus': [0]}}

    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check

    net.trafo3w = copy.deepcopy(trafo3w_copy)
    net.trafo3w.vn_hv_kv.at[0] *= 0.69
    net.trafo3w.vn_mv_kv.at[0] *= 0.69
    net.trafo3w.vn_lv_kv.at[0] *= 0.69
    check_result = pp.nominal_voltages_dont_match(net, diag_params['nom_voltage_tolerance'])
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert diag_results[check_function] == \
    {'trafo3w': {'hv_bus': [0], 'lv_bus': [0], 'mv_bus': [0]}}

    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check

    net.trafo3w = copy.deepcopy(trafo3w_copy)
    net.trafo3w.vn_hv_kv.at[0] *= 1.29
    net.trafo3w.vn_mv_kv.at[0] *= 1.29
    net.trafo3w.vn_lv_kv.at[0] *= 1.29
    check_result = pp.nominal_voltages_dont_match(net, diag_params['nom_voltage_tolerance'])
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert check_function not in diag_results

    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check

    net.trafo3w = copy.deepcopy(trafo3w_copy)
    net.trafo3w.vn_hv_kv.at[0] *= 0.71
    net.trafo3w.vn_mv_kv.at[0] *= 0.71
    net.trafo3w.vn_lv_kv.at[0] *= 0.71
    check_result = pp.nominal_voltages_dont_match(net, diag_params['nom_voltage_tolerance'])
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert check_function not in diag_results

    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check




def test_wrong_reference_system(test_net, diag_params, report_methods):
    net = copy.deepcopy(test_net)
    check_function = 'wrong_reference_system'
    diag_params = copy.deepcopy(diag_params)
    report_methods = copy.deepcopy(report_methods)
    net.load.p_kw.at[0] = -1
    net.gen.p_kw.at[0] = 1
    net.sgen.p_kw.at[0] = 1
    check_result = pp.wrong_reference_system(net)
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert diag_results[check_function] == {'gens': [0], 'loads': [0], 'sgens': [0]}

    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check


def test_disconnected_elements(test_net, diag_params, report_methods):
    net = copy.deepcopy(test_net)
    check_function = 'disconnected_elements'
    diag_params = copy.deepcopy(diag_params)
    report_methods = copy.deepcopy(report_methods)
    net.switch.closed.loc[37,38] = False
    pp.drop_trafos(net, [1])
    check_result = pp.disconnected_elements(net)
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    expected_disconnect = [{'buses': [33, 36, 37, 38, 39, 40, 41, 42, 43, 44],
      'lines': [6, 7, 8, 9, 11, 12, 13],
      'loads': [2, 5, 6, 7, 8, 9, 10, 11, 12],
      'sgens': [1, 2, 3, 4],
      'switches': [37, 38, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61],
      'trafos3w': [0]},
     {'buses': [45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56],
      'lines': [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
      'loads': [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
      'sgens': [5, 6, 7, 8, 9, 10],
      'switches': [62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                   82, 83]}]
    for area, expected_area in zip(diag_results[check_function], expected_disconnect):
        assert area == expected_area

    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check



def test_deviation_from_std_type(test_net, diag_params, report_methods):
    net = copy.deepcopy(test_net)
    check_function = 'deviation_from_std_type'
    diag_params = copy.deepcopy(diag_params)
    report_methods = copy.deepcopy(report_methods)
    net.line.r_ohm_per_km.loc[0] += 1
    net.line.x_ohm_per_km.loc[6] -= 1
    net.line.c_nf_per_km.loc[14] *= -1
    net.line.max_i_ka.loc[21] = '5'
    pp.change_std_type(net, 0, element='trafo', name='160 MVA 380/110 kV')
    net.trafo.vsc_percent.loc[0] *= 2
    check_result = pp.deviation_from_std_type(net)
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert diag_results[check_function] == \
    {'line': {0: {'e_value': 1.1571, 'param': 'r_ohm_per_km', 'std_type_in_lib': True,
                  'std_type_value': 0.1571},
              6: {'e_value': -0.883, 'param': 'x_ohm_per_km', 'std_type_in_lib': True,
                  'std_type_value': 0.117},
              14: {'e_value': -264.0, 'param': 'c_nf_per_km', 'std_type_in_lib': True,
                   'std_type_value': 264},
              21: {'e_value': '5', 'param': 'max_i_ka', 'std_type_in_lib': True,
                   'std_type_value': 0.105}},
    'trafo': {0: {'e_value': 24.4, 'param': 'vsc_percent', 'std_type_in_lib': True,
                  'std_type_value': 12.2}}
    }

    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check


def test_numba_comparison(test_net, diag_params, report_methods):
    net = copy.deepcopy(test_net)
    check_function = 'numba_comparison'
    diag_params = copy.deepcopy(diag_params)
    report_methods = copy.deepcopy(report_methods)
    diag_params['numba_tolerance'] = 1e-10
    check_result = pp.numba_comparison(net, numba_tolerance=diag_params['numba_tolerance'])
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    for element_type in diag_results[check_function]:
        check_results = diag_results[check_function]
        for result_type in check_results[element_type]:
            for result in check_results[element_type][result_type]:
                assert result > diag_params['numba_tolerance']

    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check


def test_parallel_switches(test_net, diag_params, report_methods):
    net = copy.deepcopy(test_net)
    check_function = 'parallel_switches'
    diag_params = copy.deepcopy(diag_params)
    report_methods = copy.deepcopy(report_methods)
    pp.create_switch(net, 1, 2, et='b')
    pp.create_switch(net, 13, 0, et='t', closed=False)
    pp.create_switch(net, 13, 0, et='t')
    pp.create_switch(net, 47, 16, et='l')

    check_result = pp.parallel_switches(net)
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert diag_results[check_function] == [[0, 88], [84, 89, 90], [66, 91]]

    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check


def test_missing_bus_indeces(test_net, diag_params, report_methods):
    net = copy.deepcopy(test_net)
    check_function = 'missing_bus_indeces'
    diag_params = copy.deepcopy(diag_params)
    report_methods = copy.deepcopy(report_methods)
    net.line.from_bus.iloc[0] = 10000
    net.trafo.lv_bus.iloc[0] = 10001
    net.trafo3w.mv_bus.iloc[0] = 10002
    net.switch.bus.iloc[0] = 10003
    net.switch.element.iloc[0] = 10004
    net.ext_grid.bus.iloc[0] = 10005
    check_result = pp.missing_bus_indeces(net)
    if check_result:
        diag_results = {check_function: check_result}
    else:
        diag_results = {}
    assert diag_results[check_function] == {'ext_grid': [(0, 'bus', 10005)],
                                            'line': [(0, 'from_bus', 10000)],
                                            'switch': [(0, 'bus', 10003), (0, 'element', 10004)],
                                            'trafo': [(0, 'lv_bus', 10001)],
                                            'trafo3w': [(0, 'mv_bus', 10002)]}

    for bool_value in [True, False]:
        diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=bool_value)
        report_check = None
        try:
            eval(report_methods[check_function])
            report_check = True
        except:
            report_check = False
        assert report_check


#def test_mixed():
#    net = networks.mv_network("ring")
#    net.switch.loc[6, "closed"] = 0
#    net.switch.closed.at[8] = 0
#    net.switch.closed.at[5] = 0
#    net.switch.closed.at[21] = 0
#    net.switch.closed.at[19] = 0
#    net.switch.closed.at[18] = 0
#    net.trafo.vn_hv_kv.at[0] = 20
#    net.trafo.vn_lv_kv.at[0] = 110
#    net.trafo.vn_lv_kv.at[1] = 50
#    net.bus.vn_kv.at[52] = 0.4
#    net.bus.vn_kv.at[53] = 30
#    net.load.p_kw.at[5] = -5
#    net.bus.in_service.at[15] = 0
#    net.switch.closed.iloc[0] = 1.5
#    net.bus.vn_kv.at[16] = -20
#    net.line.length_km.at[0] = -10
#    net.line.length_km.at[8] = -10
#    net.line.length_km.at[9] = 0
#    pp.create_gen(net, 1, p_kw=0)
#    pp.create_gen(net, 2, p_kw=0)
#    net.gen.bus.loc[0] = 35
#    net.gen.bus.loc[1] = 35
#
#    assert pp.diagnostic(net, report_style='None') == \
#    {'closed_switches_between_oos_and_is_buses': [284, 298],
#     'different_voltage_levels_connected': {'lines': [5, 6, 7],
#                                            'switches': [284, 299, 300]},
#     'disconnected_elements': [{'buses': [49], 'loads': [4], 'switches': [5, 6]},
#                               {'buses': [50], 'lines': [3], 'loads': [5], 'switches': [7, 8]},
#                               {'buses': [51, 52, 53, 54, 55], 'lines': [4, 5, 6, 7, 8],
#                                'loads': [6, 7, 8, 9, 10],
#                                'switches': [9, 10, 11, 12, 13, 14, 15, 16, 17, 18]},
#                               {'buses': [56], 'lines': [10], 'loads': [11], 'switches': [19, 20]},
#                               {'buses': [57, 58], 'lines': [11], 'loads': [12, 13],
#                                'switches': [21, 22, 23, 42]},
#                               {'isolated_lines': [9]}],
#    'invalid_values': {'bus': [(16, 'vn_kv', -20.0, '>0')], 'line': [(0, 'length_km', -10.0, '>0'),
#                               (8, 'length_km', -10.0, '>0'), (9, 'length_km', 0.0, '>0')],
#                       'switch': [(0, 'closed', 1.5, 'boolean')]},
#    'lines_with_impedance_close_to_zero': [0, 8, 9],
#    'multiple_voltage_controlling_elements_per_bus': {'buses_with_gens_and_ext_grids': [35],
#                                                      'buses_with_mult_gens': [35]},
#    'nominal_voltages_dont_match': {'trafo': {'hv_lv_swapped': [0], 'lv_bus': [1]}},
#    'overload': {'generation': 'uncertain', 'load': 'uncertain'},
#    'wrong_reference_system': {'loads': [5]},
#    'wrong_switch_configuration': 'uncertain'}

if __name__ == "__main__":
    pytest.main(["test_diagnostic.py", "-xs"])

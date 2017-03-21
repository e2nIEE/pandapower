# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

import pytest
import pandapower as pp
import pandapower.networks as nw
from pandapower.diagnostic_reports import DiagnosticReports

def test_diagnostic_report():
    net = nw.example_multivoltage()
    net.line.loc[7, 'length_km'] = -1
    net.gen.bus.at[0] = 0
    net.load.p_kw.at[4] *= 1000
    net.switch.closed.at[0] = 0
    net.switch.closed.at[2] = 0
    pp.create_switch(net, 41, 45, et = 'b')
    net.line.r_ohm_per_km.at[1] = 0
    net.load.p_kw.at[0] = -1
    net.switch.closed.loc[37,38] = False
    net.line.x_ohm_per_km.loc[6] -= 1
    net.line.from_bus.iloc[0] = 10000
    pp.create_switch(net, 1, 2, et='b')
    
    diag_results = pp.diagnostic(net)
    diag_params = {
        "overload_scaling_factor": 0.001,
        "lines_min_length_km": 0,
        "lines_min_z_ohm": 0,
        "nom_voltage_tolerance": 0.3,
        "numba_tolerance": 1e-5
    }
    diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=False)
    report_methods = {
        "missing_bus_indeces": diag_report.report_missing_bus_indeces,
        "disconnected_elements": diag_report.report_disconnected_elements,
        "different_voltage_levels_connected": diag_report.report_different_voltage_levels_connected,
        "lines_with_impedance_close_to_zero": diag_report.report_lines_with_impedance_close_to_zero,
        "nominal_voltages_dont_match": diag_report.report_nominal_voltages_dont_match,
        "invalid_values": diag_report.report_invalid_values,
        "overload": diag_report.report_overload,
        "multiple_voltage_controlling_elements_per_bus" : diag_report.report_multiple_voltage_controlling_elements_per_bus,
        "wrong_switch_configuration": diag_report.report_wrong_switch_configuration,
        "no_ext_grid": diag_report.report_no_ext_grid,
        "wrong_reference_system": diag_report.report_wrong_reference_system,
        "deviation_from_std_type": diag_report.report_deviation_from_std_type,
        "numba_comparison": diag_report.report_numba_comparison,
        "parallel_switches": diag_report.report_parallel_switches
                    }

    for key in diag_results:
        report_check = None
        try:
            report_methods[key]()
            report_check = True
        except:
            report_check = False
        assert report_check
        
    
    diag_report = DiagnosticReports(net, diag_results, diag_params, compact_report=True)
    for key in diag_results:
        report_check = None
        try:
            report_methods[key]()
            report_check = True
        except:
            report_check = False
        assert report_check

if __name__ == "__main__":
    pytest.main(["test_diagnostic_report.py", "-xs"])
import sys
from pandapower.converter.powerfactory.export_pfd_to_pp import from_pfd
import pandas as pd
import numpy as np
from pandapower.converter.powerfactory.pf_export_functions import run_short_circuit

sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP1\Python\3.11")
import powerfactory as pf

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

app = pf.GetApplication()
app.ActivateProject("Short_Circuit_Test_Case_SCE")
active_project = app.GetActiveProject()

# activate study case
study_case_folder = active_project.GetContents("Study Cases")[0]
study_cases = study_case_folder.GetContents()
study_case = study_cases[0]
study_case.Activate()


## functions to get short circuit results from powerfactory
def get_pf_sc_bus_results(app, fault_type='lll', calc_mode='max', fault_impedance_rf=0, fault_impedance_xf=0):

    res = run_short_circuit(app, fault_type=fault_type, calc_mode=calc_mode,
                            fault_impedance_rf=fault_impedance_rf, fault_impedance_xf=fault_impedance_xf)
    if res == 1:
        raise UserWarning("short circuit results could not be calculated in powerfactory")

    bus_results = []
    bus_elements = app.GetCalcRelevantObjects('*.ElmTerm')

    result_variables_3ph = {
        "pf_ikss_ka": "m:Ikss",
        "pf_skss_mw": "m:Skss",
        "pf_rk_ohm": "m:R",
        "pf_xk_ohm": "m:X"  # ,"pf_ip_ka": "m:ip"
    }

    result_variables = {
        "pf_ikss_a_ka": "m:Ikss:A",
        "pf_ikss_b_ka": "m:Ikss:B",
        "pf_ikss_c_ka": "m:Ikss:C",
        "pf_skss_a_mw": "m:Skss:A",
        "pf_skss_b_mw": "m:Skss:B",
        "pf_skss_c_mw": "m:Skss:C",
        "pf_rk0_ohm": "m:R0",
        "pf_xk0_ohm": "m:X0",
        "pf_rk1_ohm": "m:R1",
        "pf_xk1_ohm": "m:X1",
        "pf_rk2_ohm": "m:R2",
        "pf_xk2_ohm": "m:X2"
    }

    if fault_type == 'lll':
        result_variables = result_variables_3ph

    for bus in bus_elements:
        if bus.HasResults(0):
            bus_data = {'name': bus.loc_name}
            for col_name, pf_attribute in result_variables.items():
                try:
                    value = bus.GetAttribute(pf_attribute)
                except Exception:
                    value = np.nan
                bus_data[col_name] = value
            bus_results.append(bus_data)

    df_bus_results = pd.DataFrame(bus_results)
    return df_bus_results


def get_pf_sc_line_results(app, fault_type='lll', calc_mode='max', fault_impedance_rf=0, fault_impedance_xf=0):

    res = run_short_circuit(app, fault_type=fault_type, calc_mode=calc_mode,
                            fault_impedance_rf=fault_impedance_rf, fault_impedance_xf=fault_impedance_xf)
    if res == 1:
        raise UserWarning("short circuit results could not be calculated in powerfactory")

    line_results = []
    line_elements = app.GetCalcRelevantObjects('*.ElmLne')

    result_variables_lines_3ph = {
        "pf_ikss_from_ka": "m:Ikss:bus1",
        "pf_ikss_to_ka": "m:Ikss:bus2",
        "pf_ip_from_ka": "m:Ip:bus1",
        "pf_ip_to_ka": "m:Ip:bus2",
        "pf_skss_from_mw": "m:Skss:bus1",
        "pf_skss_to_mw": "m:Skss:bus2",
        "pf_p_from_mw": "m:P:bus1",
        "pf_p_to_mw": "m:P:bus2",
        "pf_q_from_mvar": "m:Q:bus1",
        "pf_q_to_mvar": "m:Q:bus2"
    }

    result_variables_lines = {
        "pf_ikss_a_from_ka": "m:Ikss:bus1:A",
        "pf_ikss_b_from_ka": "m:Ikss:bus1:B",
        "pf_ikss_c_from_ka": "m:Ikss:bus1:C",
        "pf_ikss_a_to_ka": "m:Ikss:bus2:A",
        "pf_ikss_b_to_ka": "m:Ikss:bus2:B",
        "pf_ikss_c_to_ka": "m:Ikss:bus2:C",
        "pf_skss_a_from_mw": "m:Skss:bus1:A",
        "pf_skss_b_from_mw": "m:Skss:bus1:B",
        "pf_skss_c_from_mw": "m:Skss:bus1:C",
        "pf_skss_a_to_mw": "m:Skss:bus2:A",
        "pf_skss_b_to_mw": "m:Skss:bus2:B",
        "pf_skss_c_to_mw": "m:Skss:bus2:C",
        "pf_p_a_from_mw": "m:P:bus1:A",
        "pf_p_b_from_mw": "m:P:bus1:B",
        "pf_p_c_from_mw": "m:P:bus1:C",
        "pf_p_a_to_mw": "m:P:bus2:A",
        "pf_p_b_to_mw": "m:P:bus2:B",
        "pf_p_c_to_mw": "m:P:bus2:C",
        "pf_q_a_from_mw": "m:Q:bus1:A",
        "pf_q_b_from_mw": "m:Q:bus1:B",
        "pf_q_c_from_mw": "m:Q:bus1:C",
        "pf_q_a_to_mw": "m:Q:bus2:A",
        "pf_q_b_to_mw": "m:Q:bus2:B",
        "pf_q_c_to_mw": "m:Q:bus2:C"
    }

    if fault_type == 'lll':
        result_variables_lines = result_variables_lines_3ph

    for line in line_elements:
        if line.HasResults(0):
            line_data = {'name': line.loc_name}
            for col_name, pf_attribute in result_variables_lines.items():
                try:
                    value = line.GetAttribute(pf_attribute)
                except Exception:
                    value = np.nan
                line_data[col_name] = value
            line_results.append(line_data)

    df_line_results = pd.DataFrame(line_results)
    return df_line_results


## get results for all fault types and cases and write to excel
fault_types = ['lll', 'll', 'llg', 'lg']
cases = ['max', 'min']
fault_impedances = [(0, 0), (5, 5)]

with pd.ExcelWriter('pf_bus_sc_results_all_cases.xlsx') as writer:
    for fault_type in fault_types:
        for case in cases:
            for fault_impedance in fault_impedances:
                df = get_pf_sc_bus_results(app, fault_type=fault_type, calc_mode=case,
                                           fault_impedance_rf=fault_impedance[0], fault_impedance_xf=fault_impedance[1])
                if fault_impedance[0] > 0:
                    sheet_name = f"{fault_type.upper()}_{case}_fault"
                else:
                    sheet_name = f"{fault_type.upper()}_{case}"
                sheet_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)

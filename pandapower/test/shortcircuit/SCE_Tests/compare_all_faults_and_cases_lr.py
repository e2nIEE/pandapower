# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pandapower.shortcircuit.calc_sc import calc_sc
from pandapower.file_io import from_json
from pandapower.test.shortcircuit.SCE_Tests.test_all_faults_and_cases import (check_pattern,
                                                                              modify_impedance_values_with_fault_value,
                                                                              load_pf_results, get_columns_to_check)

def compare_sc_results(net, excel_file):

    pf_dataframes = load_pf_results(excel_file)

    # Definiere absolute Toleranzen
    tolerances = {
        "ikss_ka": 1e-6, "skss_mw": 1e-5, "rk_ohm": 1e-6, "xk_ohm": 1e-6,
        "ikss0_ka": 1e-6, "ikss1_ka": 1e-6, "skss0_mw": 1e-5, "skss1_mw": 1e-5,
        "rk0_ohm": 1e-6, "xk0_ohm": 1e-6, "rk1_ohm": 1e-6, "xk1_ohm": 1e-6,
        "rk2_ohm": 1e-6, "xk2_ohm": 1e-6
    }

    faults = ["LLL", "LL", "LG", "LLG"]
    cases = ["max", "min"]
    fault_ohm_values = [(0.0, 0.0), (5.0, 5.0)]

    all_differences = []

    for r_fault_ohm, x_fault_ohm in fault_ohm_values:
        for fault in faults:
            columns_to_check = get_columns_to_check(fault)
            for case in cases:
                selected_sheet = f"{fault}_{case}_10"
                if r_fault_ohm != 0.0 and x_fault_ohm != 0.0:
                    selected_sheet = f"{fault}_{case}_fault_10"

                selected_pf_results = pf_dataframes[selected_sheet]
                modified_pf_results = modify_impedance_values_with_fault_value(selected_pf_results, r_fault_ohm, x_fault_ohm)

                # Kurzschlussberechnung
                calc_sc(net, fault=fault, case=case, branch_results=True, ip=False,
                        r_fault_ohm=r_fault_ohm, x_fault_ohm=x_fault_ohm)


                net.res_bus_sc["name"] = net.bus.name
                net.res_bus_sc = net.res_bus_sc[['name'] + [col for col in net.res_bus_sc.columns if col != 'name']]
                net.res_bus_sc.sort_values(by='name', inplace=True)
                result_df = net.res_bus_sc
                compare_ids = net.bus.name

                element_id_column = "name"

                # Vergleich pro Element und Spalte
                for element in compare_ids:
                    for column in columns_to_check:
                        column_key = check_pattern(column)

                        pandapower_value = result_df.loc[result_df[element_id_column] == element, column].values[0]
                        pf_value = modified_pf_results.loc[modified_pf_results[element_id_column] == element, column].values[0]

                        diff = pandapower_value - pf_value
                        diff_abs = 1 - pandapower_value / pf_value if pf_value != 0 else np.nan
                        tol = tolerances.get(column_key, 1e-6)
                        status = "OK" if abs(diff) <= tol else "Exceeds tolerance"

                        all_differences.append({
                            "Element": element,
                            "Fault Type": fault,
                            "Case": case,
                            "r_fault_ohm": r_fault_ohm,
                            "x_fault_ohm": x_fault_ohm,
                            "Quantity": column,
                            "Pandapower Result": pandapower_value,
                            "PowerFactory Result": pf_value,
                            "Difference": diff,
                            "Difference_perc": diff_abs,
                            "Status": status
                        })

    diff_df = pd.DataFrame(all_differences)
    return diff_df


##
net = from_json('test_case_1_four_bus_radial_grid.json')
excel_file = r'C:\Users\lriedl\PycharmProjects\pandapower\pandapower\test\shortcircuit\SCE_Tests\sc_result_comparison\test_case_1_four_bus_radial_grid_pf_sc_results_bus.xlsx'
diff_df = compare_sc_results(net, excel_file)


##
excel_file = r'C:\Users\lriedl\PycharmProjects\pandapower\pandapower\test\shortcircuit\SCE_Tests\sc_result_comparison\test_case_1_four_bus_radial_grid_pf_sc_results_branch.xlsx'
df = load_pf_results(excel_file)
calc_sc(net, fault='3ph', case='max', branch_results=True)
pp_branch_results = net.res_line_sc

##


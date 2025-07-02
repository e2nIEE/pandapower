# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pandapower.shortcircuit.calc_sc import calc_sc
from pandapower.file_io import from_json
from pandapower.test.shortcircuit.sce_tests.test_all_faults_and_cases import (check_pattern,
                                                                              modify_impedance_values_with_fault_value,
                                                                              load_pf_results)


def compare_sc_results(net, excel_file, branch=False, fault_location=None):
    pf_dataframes = load_pf_results(excel_file)

    # Toleranzen für relevante Größen
    tolerances = {"ikss_ka": 1e-6, "skss_mw": 1e-4, "rk_ohm": 1e-5, "xk_ohm": 1e-5,
                  "vm_pu": 1e-4, "va_degree": 1e-2, "p_mw": 1e-4, "q_mvar": 1e-4, "ikss_degree": 1e-4}

    faults = ["LLL","LL", "LG", "LLG"]
    cases = ["max"]
    fault_ohm_values = [(0.0, 0.0), (5.0, 5.0)]

    all_differences = []

    for r_fault_ohm, x_fault_ohm in fault_ohm_values:
        for fault in faults:
            for case in cases:
                selected_sheet = f"{fault}_{case}_10"
                if r_fault_ohm != 0.0 and x_fault_ohm != 0.0:
                    selected_sheet = f"{fault}_{case}_fault_10"

                selected_pf_results = pf_dataframes[selected_sheet]
                modified_pf_results = modify_impedance_values_with_fault_value(selected_pf_results, r_fault_ohm,
                                                                               x_fault_ohm)

                try:
                    calc_sc(net, fault=fault, case=case, branch_results=branch, ip=False,
                            r_fault_ohm=r_fault_ohm, x_fault_ohm=x_fault_ohm, bus=fault_location, return_all_currents=False)
                except KeyError as e:
                    print(f"KeyError for fault={fault}, case={case}: {e}")
                    continue
                except UnboundLocalError as e:
                    print(f"UnboundLocalError for fault={fault}, case={case}: {e}")
                    continue

                if branch:
                    net.res_line_sc["name"] = net.line.name
                    net.res_line_sc.sort_values(by="name", inplace=True)
                    result_df = net.res_line_sc
                    compare_ids = net.line.name
                    element_id_column = "name"
                    element_type = "line"
                else:
                    net.res_bus_sc["name"] = net.bus.name
                    net.res_bus_sc.sort_values(by="name", inplace=True)
                    result_df = net.res_bus_sc
                    compare_ids = net.bus.name
                    element_id_column = "name"
                    element_type = "bus"

                for element in compare_ids:
                    for column in result_df.columns:
                        if column == element_id_column:
                            continue
                        column_key = check_pattern(column)
                        # if column_key in ['p_mw', 'q_mvar', 'vm_pu', 'va_degree']:
                        #     continue
                        if column_key not in tolerances:
                            continue
                        try:
                            pp_val = result_df.loc[result_df[element_id_column] == element, column].values[0]
                            pf_val = \
                            modified_pf_results.loc[modified_pf_results[element_id_column] == element, column].values[0]

                            diff = pp_val - pf_val
                            diff_perc = 1 - pp_val / pf_val if pf_val != 0 else np.nan
                            diff_perc = diff_perc * 100
                            tol = tolerances[column_key]
                            status = "OK" if abs(diff) <= tol else "Exceeds tolerance"

                            all_differences.append({
                                "Element": element,
                                "Fault Type": fault,
                                "Case": case,
                                "r_fault_ohm": r_fault_ohm,
                                "x_fault_ohm": x_fault_ohm,
                                "Quantity": column,
                                "Pandapower Result": pp_val,
                                "PowerFactory Result": pf_val,
                                "Difference": np.round(diff, 4),
                                "Difference_perc": np.round(diff_perc, 4),
                                "Status": status
                            })
                        except Exception as e:
                            print(f"Error at {element_type} {element}, column {column}: {e}")

    return pd.DataFrame(all_differences)


##
# net = from_json(r"/pandapower/test/shortcircuit/sce_tests/test_grids/test_case_2_five_bus_radial_grid_dyn.json")
# excel_file = r"/pandapower/test/shortcircuit/sce_tests/sc_result_comparison/test_case_2_five_bus_radial_grid_dyn_pf_sc_results_1_bus.xlsx"
# diff_df = compare_sc_results(net, excel_file, fault_location=1)
# #
# ##
# net = from_json(r"/pandapower/test/shortcircuit/sce_tests/test_grids/test_case_2_five_bus_radial_grid_dyn.json")
# excel_file = r"/pandapower/test/shortcircuit/sce_tests/sc_result_comparison/test_case_2_five_bus_radial_grid_dyn_pf_sc_results_0_branch.xlsx"
# diff_df_branch = compare_sc_results(net, excel_file, branch=True, fault_location=0)
#
# ##
# net = from_json(r"/pandapower/test/shortcircuit/sce_tests/test_grids/test_case_1_four_bus_radial_grid.json")
# fault = 'LL'
# branch= True
# case = 'max'
# r_fault_ohm = 0
# x_fault_ohm = 0
# fault_location = 1
#
# calc_sc(net, fault=fault, case=case, branch_results=branch, ip=False,
#                             r_fault_ohm=r_fault_ohm, x_fault_ohm=x_fault_ohm, bus=fault_location, return_all_currents=False)

## sgen
net = from_json(r"C:\Users\lriedl\PycharmProjects\pandapower\pandapower\test\shortcircuit\sce_tests\test_grids\wp_2.2\1_four_bus_radial_grid_sgen.json")
#net = from_json(r"C:\Users\lriedl\PycharmProjects\pandapower\pandapower\test\shortcircuit\sce_tests\test_grids\wp_2.2\2_five_bus_radial_grid_dyn_sgen.json")
net.sgen['k'] = 1.2
net.sgen['active_current'] = False
net.sgen.loc[net.sgen.bus == 1, 'in_service'] = True
net.sgen.loc[net.sgen.bus == 2, 'in_service'] = False
net.sgen.loc[net.sgen.bus == 3, 'in_service'] = False
fault = 'LL'
branch= False
case = 'max'
r_fault_ohm = 0
x_fault_ohm = 0
fault_location = 0
calc_sc(net, fault=fault, case=case, branch_results=branch, ip=False,
                            r_fault_ohm=r_fault_ohm, x_fault_ohm=x_fault_ohm, bus=fault_location, return_all_currents=False)

## sgen bus
excel_file = r"C:\Users\lriedl\PycharmProjects\pandapower\pandapower\test\shortcircuit\sce_tests\sc_result_comparison\1_four_bus_radial_grid_sgen_pf_sc_results_0_bus_sgen1.xlsx"
#excel_file = r"C:\Users\lriedl\PycharmProjects\pandapower\pandapower\test\shortcircuit\sce_tests\sc_result_comparison\2_five_bus_radial_grid_dyn_sgen_pf_sc_results_1_bus_sgenNone.xlsx"
diff_df = compare_sc_results(net, excel_file, fault_location=0)

## sgen branch
excel_file = r"C:\Users\lriedl\PycharmProjects\pandapower\pandapower\test\shortcircuit\sce_tests\sc_result_comparison\1_four_bus_radial_grid_sgen_pf_sc_results_0_branch_sgen1.xlsx"
diff_df_branch = compare_sc_results(net, excel_file, branch=True, fault_location=0)



##


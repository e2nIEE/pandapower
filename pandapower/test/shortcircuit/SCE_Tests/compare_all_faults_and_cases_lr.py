# -*- coding: utf-8 -*-

import os

import pandas as pd
import numpy as np
from pandapower import pp_dir

from pandapower.shortcircuit.calc_sc import calc_sc
from pandapower.test.shortcircuit.sce_tests.test_all_faults_and_cases import (
    load_test_case,
    check_pattern,
    modify_impedance_values_with_fault_value,
    load_pf_results
)


def compare_sc_results(net, excel_file, branch=False, fault_location=None):
    pf_dataframes = load_pf_results(excel_file)

    # Toleranzen für relevante Größen
    tolerances = {"ikss_ka": 1e-4, "skss_mw": 1e-4, "rk_ohm": 1e-5, "xk_ohm": 1e-5,
                  "vm_pu": 1e-4, "va_degree": 1e-2, "p_mw": 1e-4, "q_mvar": 1e-4, "ikss_degree": 1e-3}

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
                            continue
                            # print(f"Error at {element_type} {element}, column {column}: {e}")

    return pd.DataFrame(all_differences)


## without sgen
net_name = "test_case_1_four_bus_radial_grid"
net_name = "test_case_3_five_bus_meshed_grid_yyn"
result_files_path = os.path.join(pp_dir, 'test', 'shortcircuit', 'sce_tests', 'sc_result_comparison')

net = load_test_case(net_name)
fault_location = 0

excel_file = f"wp_2.1/{net_name}_pf_sc_results_{fault_location}_bus.xlsx"
diff_df = compare_sc_results(net, os.path.join(result_files_path, excel_file), fault_location=fault_location)

excel_file = f"wp_2.1/{net_name}_pf_sc_results_{fault_location}_branch.xlsx"
diff_df_branch = compare_sc_results(net, os.path.join(result_files_path, excel_file), fault_location=fault_location, branch=True)

## sgen
#net_name = "1_four_bus_radial_grid_sgen"
# net_name = "2_five_bus_radial_grid_yyn_sgen"
# net_name = "3_five_bus_meshed_grid_ynyn_sgen"
net_name = "4_twenty_bus_radial_grid_yyn_sgen"
result_files_path = os.path.join(pp_dir, 'test', 'shortcircuit', 'sce_tests', 'sc_result_comparison')

net = load_test_case(net_name)

net.sgen['k'] = 1.2
net.sgen['active_current'] = False
if net_name.startswith('1_'):
    net.sgen.loc[net.sgen.bus == 1, 'in_service'] = True
    net.sgen.loc[net.sgen.bus == 2, 'in_service'] = False
    net.sgen.loc[net.sgen.bus == 3, 'in_service'] = True
# net.line["c0_nf_per_km"] = 0
# net.line["c_nf_per_km"] = 0
fault = 'LLL'
branch = True
case = 'max'
r_fault_ohm = 0
x_fault_ohm = 0
fault_location = 18

calc_sc(
      net, fault=fault, case=case, branch_results=branch, ip=False,
      r_fault_ohm=r_fault_ohm, x_fault_ohm=x_fault_ohm, bus=fault_location, return_all_currents=False
)

if net_name.startswith('1_'):
    sgen_loc = '13'
elif net_name.startswith('2_') or net_name.startswith('3_') :
    sgen_loc = '34'
else:
    if '_dyn_' in net_name:
        sgen_loc = '4'
    elif '_yyn_' in net_name:
        sgen_loc = '4714'
    elif '_ynyn_' in net_name:
        sgen_loc = '471419'

# sgen bus
excel_file = f"wp_2.2/{net_name}_pf_sc_results_{fault_location}_bus_sgen{sgen_loc}.xlsx"
diff_df = compare_sc_results(net, os.path.join(result_files_path, excel_file), fault_location=fault_location)
diff_df_nom = diff_df.copy()

# sgen branch
excel_file = f"wp_2.2/{net_name}_pf_sc_results_{fault_location}_branch_sgen{sgen_loc}.xlsx"
diff_df_branch = compare_sc_results(
    net, os.path.join(result_files_path, excel_file), branch=True, fault_location=fault_location
)

## sgen with active current
# net_name = '1_four_bus_radial_grid_sgen_act'
# net_name = "2_five_bus_radial_grid_ynyn_sgen_act"
# net_name = "3_five_bus_meshed_grid_yyn_sgen_act"
net_name = "4_twenty_bus_radial_grid_ynyn_sgen_act"

result_files_path = os.path.join(pp_dir, 'test', 'shortcircuit', 'sce_tests', 'sc_result_comparison')

net = load_test_case(net_name)

net.sgen['active_current'] = True
if net_name.startswith('1_'):
    net.sgen.loc[net.sgen.bus == 1, 'in_service'] = True
    net.sgen.loc[net.sgen.bus == 2, 'in_service'] = False
    net.sgen.loc[net.sgen.bus == 3, 'in_service'] = True

fault = 'LLL'
branch = True
case = 'max'
r_fault_ohm = 0
x_fault_ohm = 0
fault_location = 18

calc_sc(
      net, fault=fault, case=case, branch_results=branch, ip=False,
      r_fault_ohm=r_fault_ohm, x_fault_ohm=x_fault_ohm, bus=fault_location, return_all_currents=False
)
#
if net_name.startswith('1_'):
    sgen_loc = '13'
elif net_name.startswith('2_') or net_name.startswith('3_') :
    sgen_loc = '34'
else:
    if '_dyn_' in net_name:
        sgen_loc = '4'
    elif '_yyn_' in net_name:
        sgen_loc = '4714'
    elif '_ynyn_' in net_name:
        sgen_loc = '471419'

# sgen bus
excel_file = f"wp_2.2/{net_name}_pf_sc_results_{fault_location}_bus_sgen{sgen_loc}.xlsx"
diff_df = compare_sc_results(net, os.path.join(result_files_path, excel_file), fault_location=fault_location)

# sgen branch
excel_file = f"wp_2.2/{net_name}_pf_sc_results_{fault_location}_branch_sgen{sgen_loc}.xlsx"
diff_df_branch = compare_sc_results(
    net, os.path.join(result_files_path, excel_file), branch=True, fault_location=fault_location
)

##


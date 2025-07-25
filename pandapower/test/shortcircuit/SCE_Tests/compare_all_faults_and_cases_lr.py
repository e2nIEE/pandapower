# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
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
    cases = ["min", "max"]
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
                        if branch and column_key == "ikss_ka" and column == "ikss_ka":
                            continue
                        try:
                            pp_val = result_df.loc[result_df[element_id_column] == element, column].values[0]
                            pf_val = \
                            modified_pf_results.loc[modified_pf_results[element_id_column] == element, column].values[0]

                            if column_key.endswith("degree"):
                                diff = (pp_val - pf_val + 180) % 360 - 180
                            else:
                                diff = pp_val - pf_val

                            # diff = pp_val - pf_val
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


def get_result_dfs(net_name, fault_location):
    result_files_path = os.path.join(pp_dir, 'test', 'shortcircuit', 'sce_tests', 'sc_result_comparison')
    net = load_test_case(net_name)
    if net_name.endswith('_sgen') or net_name.endswith('_gen') or net_name.endswith('_sgen_act'):
        wp_folder = 'wp_2.2_2.4'

        if net_name.endswith('_sgen') or net_name.endswith('_sgen_act'):
            net.sgen['k'] = 1.2
            net.sgen['active_current'] = False if net_name.endswith('_sgen') else True
            elm_name = '_sgen'
            if net_name.startswith('1_'):
                net.sgen.loc[net.sgen.bus == 1, 'in_service'] = True
                net.sgen.loc[net.sgen.bus == 2, 'in_service'] = False
                net.sgen.loc[net.sgen.bus == 3, 'in_service'] = True

        elif net_name.endswith('_gen'):
            net.gen['k'] = 6
            net.gen['active_current'] = False
            elm_name = '_gen'
            if net_name.startswith('1_'):
                net.gen.loc[net.gen.bus == 1, 'in_service'] = True
                net.gen.loc[net.gen.bus == 2, 'in_service'] = False
                net.gen.loc[net.gen.bus == 3, 'in_service'] = True

        if net_name.startswith('1_'):
            gen_loc = f'{elm_name}13'
        elif net_name.startswith('2_') or net_name.startswith('3_'):
            gen_loc = f'{elm_name}34'
        else:
            if '_dyn_' in net_name:
                gen_loc = f'{elm_name}4'
            elif '_yyn_' in net_name:
                gen_loc = f'{elm_name}4714'
            elif '_ynyn_' in net_name:
                gen_loc = f'{elm_name}471419'

    else:
        wp_folder = 'wp_2.1'
        gen_loc = ''

    # bus
    excel_file = f"{wp_folder}/{net_name}_pf_sc_results_{fault_location}_bus{gen_loc}.xlsx"
    diff_df = compare_sc_results(net, os.path.join(result_files_path, excel_file), fault_location=fault_location)

    # branch
    excel_file = f"{wp_folder}/{net_name}_pf_sc_results_{fault_location}_branch{gen_loc}.xlsx"
    diff_df_branch = compare_sc_results(net, os.path.join(result_files_path, excel_file), branch=True, fault_location=fault_location)

    return diff_df, diff_df_branch


def generate_summary_table(net_names, fault_locations):
    summary = []

    current_keys = ["ikss_ka", "ikss_degree"]
    impedance_keys = ["rk_ohm", "xk_ohm"]
    voltage_keys = ["vm_pu", "va_degree"]

    combinations = [(net, loc) for net in net_names for loc in fault_locations]

    for net_name, fault_location in tqdm(combinations, desc="generate_summary", unit="grid"):
        try:
            diff_df, diff_df_branch = get_result_dfs(net_name, fault_location)

            for df, scope in [(diff_df, "bus"), (diff_df_branch, "branch")]:
                if df.empty:
                    continue
                grouped = df.groupby(["Fault Type", "Case", "r_fault_ohm", "x_fault_ohm"])

                for group_keys, group_df in grouped:
                    current_ok = all(group_df[group_df["Quantity"].isin(current_keys)]["Status"] == "OK")
                    impedance_ok = all(group_df[group_df["Quantity"].isin(impedance_keys)]["Status"] == "OK")
                    voltage_ok = all(group_df[group_df["Quantity"].isin(voltage_keys)]["Status"] == "OK")

                    overall_ok = all([current_ok, impedance_ok, voltage_ok])
                    summary.append({
                        "name": net_name,
                        "location": fault_location,
                        "element": scope,
                        "fault_type": group_keys[0],
                        "case": group_keys[1],
                        "rx_fault_ohm": group_keys[2],
                        "current OK": "OK" if current_ok else "Exceeds",
                        "impedance OK": "OK" if impedance_ok else "Exceeds",
                        "voltage OK": "OK" if voltage_ok else "Exceeds",
                        "total": "OK" if overall_ok else "Exceeds"
                    })
        except Exception as e:
            print(f"Fehler bei {net_name}, Ort {fault_location}: {e}")
            continue

    return pd.DataFrame(summary)


## all net names
testfiles_path = os.path.join(pp_dir, 'test', 'shortcircuit', 'sce_tests', 'test_grids', 'wp_2.1')
testfiles_gen_path = os.path.join(pp_dir, 'test', 'shortcircuit', 'sce_tests', 'test_grids', 'wp_2.2_2.4')
net_names = [f[:-5] for f in os.listdir(testfiles_path) if f.endswith(".json")]
net_names_gen = [f[:-5] for f in os.listdir(testfiles_gen_path) if f.endswith(".json")]

## show panadpower and powerfactory results for specified grid and location
net_name = "3_five_bus_meshed_grid_ynyn_sgen"  # possible net_name in net_names and net_names_gen
fault_location = 2  # 0, 1, 2, 3 for four- and five-bus grids; 0, 8, 18 for twenty-bus grid

diff_df, diff_df_branch = get_result_dfs(net_name, fault_location)

## overview for all grids
names = net_names_gen
fault_location = [0]
df_summary = generate_summary_table(names, fault_location)

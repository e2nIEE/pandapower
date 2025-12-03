# -*- coding: utf-8 -*-

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from pandapower import pp_dir
from pandapower.shortcircuit.calc_sc import calc_sc
from pandapower.test.shortcircuit.sce_tests.functions_tests import (
    load_test_case,
    check_pattern,
    modify_impedance_values_with_fault_value,
    load_pf_results
)


def compare_sc_results(net, excel_file, branch=False, fault_location=None, lv_tol_percent=10):
    pf_dataframes = load_pf_results(excel_file)

    # Toleranzen für relevante Größen
    tolerances = {"ikss_ka": 1e-4, "skss_mw": 1e-4, "rk_ohm": 1e-5, "xk_ohm": 1e-5,
                  "vm_pu": 1e-4, "va_degree": 1e-2, "p_mw": 1e-4, "q_mvar": 1e-4, "ikss_degree": 1e-3}

    faults = ["LLL"] #,"LL", "LG", "LLG"]
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
                    calc_sc(net, fault=fault, case=case, branch_results=branch, ip=False, lv_tol_percent=lv_tol_percent,
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
                            #print(f"Error at {element_type} {element}, column {column}: {e}")

    return pd.DataFrame(all_differences)


def get_result_dfs(net_name, fault_location, lv_tol_percent):
    if 'four_bus' in net_name and fault_location not in [0, 1, 3]:
        print(f"For {net_name} only fault locations 0, 1, 3 are supported. Skipping fault location {fault_location}.")
        return None, None
    elif 'five_bus' in net_name and fault_location not in [0, 2, 4]:
        print(f"For {net_name} only fault locations 0, 2, 4 are supported. Skipping fault location {fault_location}.")
        return None, None
    elif 'eight_bus' in net_name and fault_location not in [0, 2, 4, 6]:
        print(f"For {net_name} only fault locations 0, 2, 4, 6 are supported. Skipping fault location {fault_location}.")
        return None, None

    result_files_path = os.path.join(pp_dir, 'test', 'shortcircuit', 'sce_tests', 'sc_result_comparison')
    net = load_test_case(net_name)

    # set zero values to very small values avoid division by zero in short circuit calculation #TODO: only temporary!
    x = 1e-20

    net.line.loc[net.line.name.str.contains('1ph'), "c0_nf_per_km"] = x
    net.line.loc[net.line.name.str.contains('1ph'), "r0_ohm_per_km"] = x
    net.line.loc[net.line.name.str.contains('1ph'), "x0_ohm_per_km"] = x

    net.line.loc[net.line.name.str.contains('2ph'), "c0_nf_per_km"] = x
    net.line.loc[net.line.name.str.contains('2ph'), "r0_ohm_per_km"] = x
    net.line.loc[net.line.name.str.contains('2ph'), "x0_ohm_per_km"] = x

    wp_folder = 'wp_2.3'

    # bus
    excel_file = f"{wp_folder}/{net_name}_pf_sc_results_{fault_location}_bus.xlsx"
    diff_df = compare_sc_results(net, os.path.join(result_files_path, excel_file),
                                 fault_location=fault_location, lv_tol_percent=lv_tol_percent)

    # branch
    excel_file = f"{wp_folder}/{net_name}_pf_sc_results_{fault_location}_branch.xlsx"
    diff_df_branch = compare_sc_results(net, os.path.join(result_files_path, excel_file), branch=True,
                                        fault_location=fault_location, lv_tol_percent=lv_tol_percent)

    return diff_df, diff_df_branch, net


def generate_summary_tables(net_names, fault_locations, detailed=False, lv_tol_percent=10):
    bus_summary = []
    branch_summary = []

    current_keys = ["ikss_ka", "ikss_degree", "ikss_a_ka", "ikss_b_ka", "ikss_c_ka",
                    "ikss_a_degree", "ikss_b_degree", "ikss_c_degree",
                    "ikss_a_from_ka", "ikss_b_from_ka", "ikss_c_from_ka",
                    "ikss_a_to_ka", "ikss_b_to_ka", "ikss_c_to_ka"]

    impedance_keys = ["rk_ohm", "xk_ohm", "rk0_ohm", "rk1_ohm", "rk2_ohm",
                      "xk0_ohm", "xk1_ohm", "xk2_ohm"]

    voltage_keys = ["vm_pu", "va_degree", "vm_a_from_pu", "vm_b_from_pu", "vm_c_from_pu",
                    "vm_a_to_pu", "vm_b_to_pu", "vm_c_to_pu",
                    "va_a_from_degree", "va_b_from_degree", "va_c_from_degree",
                    "va_a_to_degree", "va_b_to_degree", "va_c_to_degree"]

    combinations = [(net, loc) for net in net_names for loc in fault_locations]

    for net_name, fault_location in tqdm(combinations, desc="generate_summary", unit="grid"):
        try:
            diff_df, diff_df_branch = get_result_dfs(net_name, fault_location, lv_tol_percent)
            if diff_df is None and diff_df_branch is None:
                continue

            # bus
            if not diff_df.empty:
                if detailed:
                    grouped_bus = diff_df.groupby(["Fault Type", "Case", "r_fault_ohm", "x_fault_ohm"])
                    for group_keys, group_df in grouped_bus:
                        current_ok = all(group_df[group_df["Quantity"].isin(current_keys)]["Status"] == "OK")
                        impedance_ok = all(group_df[group_df["Quantity"].isin(impedance_keys)]["Status"] == "OK")
                        overall_ok = all([current_ok, impedance_ok])
                        bus_summary.append({
                            "name": net_name,
                            "location": fault_location,
                            "fault_type": group_keys[0],
                            "case": group_keys[1],
                            "rx_fault_ohm": str(group_keys[2]),
                            "current ok": True if current_ok else False,
                            "impedance ok": True if impedance_ok else False,
                            "total ok": True if overall_ok else False
                        })
                else:
                    for fault_type, group_df in diff_df.groupby("Fault Type"):
                        current_ok = all(group_df[group_df["Quantity"].isin(current_keys)]["Status"] == "OK")
                        impedance_ok = all(group_df[group_df["Quantity"].isin(impedance_keys)]["Status"] == "OK")
                        overall_ok = all([current_ok, impedance_ok])
                        bus_summary.append({
                            "name": net_name,
                            "location": fault_location,
                            "fault_type": fault_type,
                            "current ok": True if current_ok else False,
                            "impedance ok": True if impedance_ok else False,
                            "total ok": True if overall_ok else False
                        })

            # branch
            if not diff_df_branch.empty:
                if detailed:
                    grouped_branch = diff_df_branch.groupby(["Fault Type", "Case", "r_fault_ohm", "x_fault_ohm"])
                    for group_keys, group_df in grouped_branch:
                        current_ok = all(group_df[group_df["Quantity"].isin(current_keys)]["Status"] == "OK")
                        voltage_ok = all(group_df[group_df["Quantity"].isin(voltage_keys)]["Status"] == "OK")
                        overall_ok = all([current_ok, voltage_ok])
                        branch_summary.append({
                            "name": net_name,
                            "location": fault_location,
                            "fault_type": group_keys[0],
                            "case": group_keys[1],
                            "rx_fault_ohm": str(group_keys[2]),
                            "current ok": True if current_ok else False,
                            "voltage ok": True if voltage_ok else False,
                            "total ok": True if overall_ok else False
                        })
                else:
                    for fault_type, group_df in diff_df_branch.groupby("Fault Type"):
                        current_ok = all(group_df[group_df["Quantity"].isin(current_keys)]["Status"] == "OK")
                        voltage_ok = all(group_df[group_df["Quantity"].isin(voltage_keys)]["Status"] == "OK")
                        overall_ok = all([current_ok, voltage_ok])
                        branch_summary.append({
                            "name": net_name,
                            "location": fault_location,
                            "fault_type": fault_type,
                            "current ok": True if current_ok else False,
                            "voltage ok": True if voltage_ok else False,
                            "total ok": True if overall_ok else False
                        })

        except Exception as e:
            print(f"error for {net_name}, {fault_location}: {e}")
            continue

    return pd.DataFrame(bus_summary), pd.DataFrame(branch_summary)


##
if __name__ == "__main__":
    ## all net names
    testfiles_path = os.path.join(pp_dir, 'test', 'shortcircuit', 'sce_tests', 'test_grids', 'wp_2.3')
    net_names = [f[:-5] for f in os.listdir(testfiles_path) if f.endswith(".json")]
    net_names.append(net_names.pop(0))
    print(net_names)

    # ## show panadpower and powerfactory results for specified grid and location
    # net_name = '3_five_bus_radial_grid_1ph_dyn_MP'  # possible net_name in net_names
    # fault_location = 4  # 0, 1, 3 for four-bus grids; 0, 2, 4 for five-bus grids, 0, 2, 4, 6 for eight-bus grids
    # lv_tol_percent = 10
    #
    # diff_df, diff_df_branch, net = get_result_dfs(net_name, fault_location, lv_tol_percent)
    #
    # ## detailed overview for all grids
    # names = [name for name in net_names]
    # fault_location = [4]
    # lv_tol_percent = 10
    # df_bus, df_branch = generate_summary_tables(names, fault_location,
    #                                             detailed=True, lv_tol_percent=lv_tol_percent)
    #
    # ## simple overview for all grids
    # names = [name for name in net_names]
    # fault_location = [4]
    # lv_tol_percent = 6
    # df_bus_simple, df_branch_simple = generate_summary_tables(names, fault_location,
    #                                                           detailed=False, lv_tol_percent=lv_tol_percent)


##
import pandapower as pp
from pandapower.shortcircuit.calc_sc import calc_sc
net = pp.from_json(r"C:\Users\lriedl\PycharmProjects\sce\pandapower\test\shortcircuit\sce_tests\test_grids\wp_2.3\1_four_bus_radial_1ph_grid_MP.json")
x = 0.001

net.line.loc[net.line.name.str.contains('1ph'), "c0_nf_per_km"] = x
net.line.loc[net.line.name.str.contains('1ph'), "r0_ohm_per_km"] = x
net.line.loc[net.line.name.str.contains('1ph'), "x0_ohm_per_km"] = x

net.line.loc[net.line.name.str.contains('2ph'), "c0_nf_per_km"] = x
net.line.loc[net.line.name.str.contains('2ph'), "r0_ohm_per_km"] = x
net.line.loc[net.line.name.str.contains('2ph'), "x0_ohm_per_km"] = x

# net.line.loc[0, "in_service"] = False
# net.line.loc[1, "in_service"] = False
net.line.loc[2, "in_service"] = False

calc_sc(net, fault="LLG", case="max", branch_results=False, ip=False, lv_tol_percent=10, bus=2)

##


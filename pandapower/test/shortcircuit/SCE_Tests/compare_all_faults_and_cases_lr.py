# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import copy
from pandapower.shortcircuit.calc_sc import calc_sc
from pandapower.file_io import from_json


def check_pattern(pattern):
    if re.match(r"^rk[0-2]?_ohm$", pattern):
        return "rk_ohm"
    elif re.match(r"^xk[0-2]?_ohm$", pattern):
        return "xk_ohm"
    else:
        return pattern


def modify_impedance_values_with_fault_value(selected_results, r_ohm, x_ohm):
    modified_results = copy.deepcopy(selected_results)

    def adjust_columns(df, pattern, value):
        matching_columns = df.columns[df.columns.str.match(pattern)]
        df[matching_columns] += value

    rk_pattern = r"^rk[0-2]?_ohm$|^rk_ohm$"
    xk_pattern = r"^xk[0-2]?_ohm$|^xk_ohm$"

    adjust_columns(modified_results, rk_pattern, r_ohm)
    adjust_columns(modified_results, xk_pattern, x_ohm)

    return modified_results


def load_pf_results(excel_file):
    sheets = pd.ExcelFile(excel_file).sheet_names
    dataframes = {}

    for sheet in sheets:
        pf_results = pd.read_excel(excel_file, sheet_name=sheet)

        if sheet.startswith("LLL_") or sheet.startswith("LL_"):
            if sheet.startswith("LL_"):
                pf_results.drop(columns=['pf_ikss_a_ka', 'pf_ikss_b_ka', 'pf_skss_a_mw', 'pf_skss_b_mw',
                                         'pf_rk0_ohm', 'pf_rk1_ohm', 'pf_xk0_ohm', 'pf_xk1_ohm'], inplace=True)
            pf_results.columns = ['name', 'ikss_ka', 'skss_mw', 'rk_ohm', 'xk_ohm']

        elif sheet.startswith("LLG_"):
            pf_results.drop(columns=['pf_ikss_a_ka', 'pf_skss_a_mw'], inplace=True)
            pf_results.columns = ["name", "ikss0_ka", "ikss1_ka", 'skss0_mw', 'skss1_mw',
                                  "rk0_ohm", "xk0_ohm", "rk1_ohm", "xk1_ohm", "rk2_ohm", "xk2_ohm"]
        elif sheet.startswith("LG_"):
            pf_results.drop(columns=['pf_ikss_b_ka', 'pf_ikss_c_ka', 'pf_skss_b_mw', 'pf_skss_c_mw'], inplace=True)
            pf_results.columns = ["name", "ikss_ka", 'skss_mw', "rk0_ohm", "xk0_ohm", "rk1_ohm",
                                  "xk1_ohm", "rk2_ohm", "xk2_ohm"]

        dataframes[sheet] = pf_results

    return dataframes


def get_columns_to_check(fault):
    if fault in ["LLL", "LL"]:
        return ["ikss_ka", "skss_mw", "rk_ohm", "xk_ohm"]
    elif fault == "LG":
        return ["ikss_ka", "skss_mw", "rk0_ohm", "xk0_ohm", "rk1_ohm", "xk1_ohm", "rk2_ohm", "xk2_ohm"]
    elif fault == 'LLG':
        return ["ikss0_ka", "ikss1_ka", "skss0_mw", "skss1_mw",
                "rk0_ohm", "xk0_ohm", "rk1_ohm", "xk1_ohm", "rk2_ohm", "xk2_ohm"]
    return []


def compare_sc_results():
    # Load network and PowerFactory results
    net = from_json('4_bus_radial_grid.json')
    net.line.rename(columns={'temperature_degree_celsius': 'endtemp_degree'}, inplace=True)
    net.line["endtemp_degree"] = 250

    excel_file = r'C:\Users\lriedl\PycharmProjects\pandapower\pandapower\test\shortcircuit\SCE_Tests\pf_bus_sc_results_all_cases.xlsx'
    pf_dataframes = load_pf_results(excel_file)

    # Define absolute tolerances
    tolerances = {
        "ikss_ka": 1e-6, "skss_mw": 1e-5, "rk_ohm": 1e-6, "xk_ohm": 1e-6,
        "ikss0_ka": 1e-6, "ikss1_ka": 1e-6, "skss0_mw": 1e-5, "skss1_mw": 1e-5,
        "rk0_ohm": 1e-6, "xk0_ohm": 1e-6, "rk1_ohm": 1e-6, "xk1_ohm": 1e-6, "rk2_ohm": 1e-6, "xk2_ohm": 1e-6
    }

    faults = ["LLL", "LL", "LG"]
    cases = ["max", "min"]
    fault_ohm_values = [(0.0, 0.0), (15.0, 15.0)]

    all_differences = []

    for r_fault_ohm, x_fault_ohm in fault_ohm_values:
        for fault in faults:
            columns_to_check = get_columns_to_check(fault)
            for case in cases:
                selected_sheet = f"{fault}_{case}"
                if r_fault_ohm != 0.0 and x_fault_ohm != 0.0:
                    selected_sheet += "_fault"

                selected_pf_results = pf_dataframes[selected_sheet]
                modified_pf_results = modify_impedance_values_with_fault_value(selected_pf_results, r_fault_ohm, x_fault_ohm)

                calc_sc(net, fault=fault, case=case, branch_results=True, ip=False, r_fault_ohm=r_fault_ohm,
                        x_fault_ohm=x_fault_ohm)

                net.res_bus_sc["name"] = net.bus.name
                net.res_bus_sc = net.res_bus_sc[['name'] + [col for col in net.res_bus_sc.columns if col != 'name']]
                net.res_bus_sc.sort_values(by='name', inplace=True)

                for bus in net.bus.name:
                    for column in columns_to_check:
                        column_key = check_pattern(column)
                        pandapower_value = net.res_bus_sc.loc[net.res_bus_sc.name == bus, column].values[0]
                        pf_value = modified_pf_results.loc[modified_pf_results.name == bus, column].values[0]
                        diff = pandapower_value - pf_value
                        diff_abs = 1 - pandapower_value/pf_value

                        tol = tolerances.get(column_key, 1e-6)
                        status = "OK" if abs(diff) <= tol else "Exceeds tolerance"

                        all_differences.append({
                            "Bus": bus,
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

    # Save all comparison results into an Excel file
    diff_df = pd.DataFrame(all_differences)
    #diff_df.to_excel("comparison_results.xlsx", index=False)
    #print("Comparison finished! Results saved to 'comparison_results.xlsx'.")
    return diff_df


diff_df = compare_sc_results()


#
df = diff_df[diff_df['Fault Type'] == 'LL']
#df = df[df['r_fault_ohm'] >0]
df.head()

##


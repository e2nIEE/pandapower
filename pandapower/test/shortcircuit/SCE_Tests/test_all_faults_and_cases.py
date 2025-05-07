# -*- coding: utf-8 -*-

# Copyright (c) 2016-2025 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pandas as pd
import numpy as np
from pandapower.shortcircuit.calc_sc import calc_sc
from pandapower.file_io import from_json
import pytest
import re
import copy
import os
from pandapower import pp_dir


def check_pattern(pattern):
    """
    Checks the given pattern and returns a corresponding identifier.

    This function checks if the input pattern matches specific regular expressions
    for 'rk', 'xk', 'ikss', and 'skss' types. It returns a standardized identifier if a match is found
    or returns the original pattern if no match is found.

    Parameters:
    pattern (str): The input pattern to check.

    Returns:
    str: A standardized identifier ('rk_ohm', 'xk_ohm', 'ikss_a_ka', 'skss_a_mw') or the original pattern.
    """
    if re.match(r"^rk[0-2]?_ohm$", pattern):
        return "rk_ohm"
    elif re.match(r"^xk[0-2]?_ohm$", pattern):
        return "xk_ohm"
    elif re.match(r"^ikss_[abc]_ka$", pattern):  # Matches ikss_a_ka, ikss_b_ka, ikss_c_ka
        return "ikss_ka"
    elif re.match(r"^skss_[abc]_mw$", pattern):  # Matches skss_a_mw, skss_b_mw, skss_c_mw
        return "skss_mw"
    else:
        return pattern


def modify_impedance_values_with_fault_value(selected_results, r_ohm, x_ohm):
    """
    Modifies the impedance values in a DataFrame by subtracting r_ohm from rk columns
    and x_ohm from xk columns.

    Parameters:
    selected_results (pd.DataFrame): The input DataFrame containing impedance values.
    r_ohm (float): The value to be subtracted from the rk columns.
    x_ohm (float): The value to be subtracted from the xk columns.

    Returns:
    pd.DataFrame: The modified DataFrame with adjusted values.
    """
    # Create a deep copy of the input DataFrame
    copy_selected_results = copy.deepcopy(selected_results)

    def adjust_columns(df, pattern, value):
        """Helper function to adjust column values based on a pattern."""
        matching_columns = df.columns[df.columns.str.match(pattern)]
        df[matching_columns] += value

    # Add r_ohm to rk columns
    rk_pattern = r"^rk[0-2]?_ohm$|^rk_ohm$"
    adjust_columns(copy_selected_results, rk_pattern, r_ohm)

    # Add x_ohm to xk columns
    xk_pattern = r"^xk[0-2]?_ohm$|^xk_ohm$"
    adjust_columns(copy_selected_results, xk_pattern, x_ohm)

    return copy_selected_results


def load_pf_results(excel_file):
    """Load power flow results from Excel sheets."""
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
            # TODO also check skss_a_mw, skss_b_mw, skss_c_mw
            pf_results.drop(columns=['pf_skss_a_mw', 'pf_skss_b_mw', 'pf_skss_c_mw'], inplace=True)
            pf_results.columns = ["name", "ikss_a_ka", "ikss_b_ka", 'ikss_c_ka',
                                  "rk0_ohm", "xk0_ohm", "rk1_ohm", "xk1_ohm", "rk2_ohm", "xk2_ohm"]
        elif sheet.startswith("LG_"):
            pf_results.drop(columns=['pf_ikss_b_ka', 'pf_ikss_c_ka', 'pf_skss_b_mw', 'pf_skss_c_mw'], inplace=True)
            pf_results.columns = ["name", "ikss_ka", 'skss_mw', "rk0_ohm", "xk0_ohm", "rk1_ohm",
                                  "xk1_ohm", "rk2_ohm", "xk2_ohm"]

        dataframes[sheet] = pf_results

    return dataframes


def get_columns_to_check(fault):
    """Return the columns to check based on the fault type."""
    if fault in ["LLL", "LL"]:
        return ["ikss_ka", "skss_mw", "rk_ohm", "xk_ohm"]
    elif fault == "LG":
        return ["ikss_ka", "skss_mw", "rk0_ohm", "xk0_ohm", "rk1_ohm", "xk1_ohm", "rk2_ohm", "xk2_ohm"]
    elif fault == 'LLG':
        # TODO also check skss_a_mw, skss_b_mw, skss_c_mw
        return ["ikss_a_ka", "ikss_b_ka", 'ikss_c_ka', "rk0_ohm", "xk0_ohm", "rk1_ohm", "xk1_ohm", "rk2_ohm", "xk2_ohm"]
    return []


@pytest.mark.parametrize("fault, case, r_fault_ohm, x_fault_ohm", [
    ("LLL", "max", 0.0, 0.0),
    ("LLL", "min", 0.0, 0.0),
    ("LL", "max", 0.0, 0.0),
    ("LL", "min", 0.0, 0.0),
    ("LG", "max", 0.0, 0.0),
    ("LG", "min", 0.0, 0.0),
    ("LLG", "max", 0.0, 0.0),
    ("LLG", "min", 0.0, 0.0),
    ("LLL", "max", 5.0, 5.0),
    ("LLL", "min", 5.0, 5.0),
    ("LL", "max", 5.0, 5.0),
    ("LL", "min", 5.0, 5.0),
    ("LG", "max", 5.0, 5.0),
    ("LG", "min", 5.0, 5.0),
])
def test_all_faults_and_cases_with_fault_impedance(fault, case, r_fault_ohm, x_fault_ohm):
    net = from_json(os.path.join(pp_dir, "test", "shortcircuit", "SCE_Tests", "4_bus_radial_grid.json"))
    net.line.rename(columns={'temperature_degree_celsius': 'endtemp_degree'}, inplace=True)
    net.line["endtemp_degree"] = 250

    excel_file = os.path.join(pp_dir, "test", "shortcircuit", "SCE_Tests", "pf_bus_sc_results_all_cases.xlsx")
    dataframes = load_pf_results(excel_file)

    rtol = {"ikss_ka": 0, "skss_mw": 0, "rk_ohm": 0, "xk_ohm": 0}
    atol = {"ikss_ka": 1e-6, "skss_mw": 1e-5, "rk_ohm": 1e-6, "xk_ohm": 1e-6}

    columns_to_check = get_columns_to_check(fault)
    selected_sheet = f"{fault}_{case}"
    if r_fault_ohm != 0.0 and x_fault_ohm != 0.0:
        selected_sheet += "_fault"

    selected_pf_results = dataframes[selected_sheet]
    modified_pf_results = modify_impedance_values_with_fault_value(selected_pf_results, r_fault_ohm, x_fault_ohm)

    calc_sc(net, fault=fault, case=case, branch_results=True, ip=False, r_fault_ohm=r_fault_ohm, x_fault_ohm=x_fault_ohm)

    net.res_bus_sc["name"] = net.bus.name
    net.res_bus_sc = net.res_bus_sc[['name'] + [col for col in net.res_bus_sc.columns if col != 'name']]
    net.res_bus_sc.sort_values(by='name', inplace=True)

    for bus in net.bus.name:
        for column in columns_to_check:
            column_ar = check_pattern(column)
            assert np.isclose(
                net.res_bus_sc.loc[net.bus.name == bus, column].values[0],
                modified_pf_results.loc[modified_pf_results.name == bus, column].values[0],
                rtol=rtol[column_ar], atol=atol[column_ar]
            )


if __name__ == "__main__":
    pytest.main([__file__])

"""net = from_json('4_bus_radial_grid.json')
net.line.rename(columns={'temperature_degree_celsius': 'endtemp_degree'}, inplace=True)
net.line["endtemp_degree"] = 250
calc_sc(net, fault="LLG", case="max", branch_results=False, ip=False, r_fault_ohm=0, x_fault_ohm=0)"""
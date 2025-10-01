from itertools import product
from re import match
import copy
import os

import pandas as pd
import numpy as np
from pandapower.auxiliary import pandapowerNet

from pandapower import pp_dir
from pandapower.shortcircuit.calc_sc import calc_sc
from pandapower.file_io import from_json
from pandapower.create import create_ward

import warnings
import logging

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

testfiles_path = os.path.join(pp_dir, 'test', 'shortcircuit', 'sce_tests')


def create_parameter_list(net_names, faults, cases, values, lv_tol_percents, fault_location_buses, is_branch_test,
                          vector_groups, gen_idx, is_active_current, gen_mode, grounding_types):
    # parameter list for WP 2.1
    parametrize_values_wp21 = []

    for fault, case, value, fault_bus, branch_test in product(
            faults, cases, values, fault_location_buses, is_branch_test
    ):
        if case == "min":
            for lv_tol in lv_tol_percents:
                parametrize_values_wp21.append(
                    (fault, case, value, lv_tol, fault_bus, branch_test)
                )
        elif case == "max":
            parametrize_values_wp21.append(
                (fault, case, value, 10, fault_bus, branch_test)
            )

    # parameter list for WP 2.2 and WP 2.4
    parametrize_values_wp22 = list(
        product(faults, ['max'], values, [10], fault_location_buses, is_branch_test,
                gen_idx, is_active_current, gen_mode)
    )

    # parameter list with vector group for WP 2.1
    #  uses "Dyn" as vector group for LLL and LL. LLG and LG are combined with all vector groups.
    parametrize_values_vector_wp21 = []
    for net_name in net_names:
        if "twenty_bus" in net_name:
            fl_buses = [0, 8, 18]
        else:
            fl_buses = fault_location_buses

        parametrize_values_vector_wp21 += list(product(
            [net_name], faults[:2], cases, values, lv_tol_percents, vector_groups[:1],
            fl_buses, is_branch_test
        ))
        parametrize_values_vector_wp21 += list(product(
            [net_name], faults[2:], cases, values, lv_tol_percents, vector_groups,
            fl_buses, is_branch_test
        ))

    # parameter list with vector group for WP 2.2 and WP 2.4
    base_wp22 = []
    for net_name in net_names:
        if "twenty_bus" in net_name:
            fl_buses = [0, 8, 18]
        else:
            fl_buses = fault_location_buses
        base_wp22 += list(product(
            [net_name], faults[:2], ['max'], values, [10], vector_groups[:1],
            fl_buses, is_branch_test, [None], is_active_current, gen_mode
        ))
        base_wp22 += list(product(
            [net_name], faults[2:], ['max'], values, [10], vector_groups,
            fl_buses, is_branch_test, [None], is_active_current, gen_mode
        ))

    parametrize_values_vector_wp22 = []
    for params in base_wp22:
        (
            net_name, fault, case, value, tol, vgroup,
            fault_loc, branch_test, gen_id, active_curr, mode
        ) = params

        short_name = net_name.replace("test_case_", "")
        if "five_bus" in net_name:
            gen_id = [3, 4]

        if "twenty_bus" in net_name:
            if vgroup == "Dyn":
                gen_id = 4
            elif vgroup == "Yyn":
                gen_id = [4, 7, 14]
            elif vgroup == "YNyn":
                gen_id = [4, 7, 14, 19]

        parametrize_values_vector_wp22.append((
            short_name, fault, case, value, tol, vgroup,
            fault_loc, branch_test, gen_id, active_curr, mode
        ))

    # parameter list with vector group for WP 2.5
    parametrize_values_vector_wp25 = []
    for net_name in net_names:
        if "twenty_bus" in net_name:
            fl_buses = [0, 8, 18]
        else:
            fl_buses = fault_location_buses

        parametrize_values_vector_wp25 += list(product(
            [net_name[10:]], faults[2:], cases, values, lv_tol_percents, vector_groups,
            fl_buses, is_branch_test, grounding_types
        ))

    # parameter list for WP 2.5 with grounding bank
    base_wp25 = []
    for net_name in net_names:
        if "twenty_bus" in net_name:
            fl_buses = [0, 8, 18]
        else:
            fl_buses = fault_location_buses

        base_wp25 += list(product(
            [net_name[10:]],
            faults[2:],
            cases,
            values,
            lv_tol_percents,
            vector_groups,
            fl_buses,
            is_branch_test,
            grounding_types,
            [None]
        ))

    parametrize_values_vector_wp25_with_ward = []
    for params in base_wp25:
        (
            short_name, fault, case, value, tol, vgroup,
            fault_loc, branch_test, grounding, ward_id
        ) = params

        if "five_bus" in short_name:
            ward_id = [3, 4]

        if "twenty_bus" in short_name:
            if vgroup == "Dyn":
                ward_id = [4]
            elif vgroup == "Yyn":
                ward_id = [4, 7, 14]
            elif vgroup == "YNyn":
                ward_id = [4, 7, 14, 19]

        parametrize_values_vector_wp25_with_ward.append((
            short_name, fault, case, value, tol, vgroup,
            fault_loc, branch_test, grounding, ward_id
        ))

    return (parametrize_values_wp21, parametrize_values_vector_wp21,
            parametrize_values_wp22, parametrize_values_vector_wp22, parametrize_values_vector_wp25,
            parametrize_values_vector_wp25_with_ward)


def compare_results(columns_to_check, net_df, pf_results):
    # define tolerances
    rtol = {"ikss_ka": 0, "skss_mw": 0, "rk_ohm": 0, "xk_ohm": 0,
            "vm_pu": 0, "va_degree": 0, "p_mw": 0, "q_mvar": 0, "ikss_degree": 0}
    # TODO skss_mw and ikss_ka only 1e-4 sufficient?
    # TODO rk_ohm and xk_ohm only 1e-3 sufficient?
    atol = {"ikss_ka": 1e-4, "skss_mw": 1e-4, "rk_ohm": 1e-3, "xk_ohm": 1e-3,
            "vm_pu": 1e-4, "va_degree": 1e-2, "p_mw": 1e-4, "q_mvar": 1e-4, "ikss_degree": 1e-2}  # TODO: tolerances ok?

    for column in columns_to_check:
        if column == 'name':
            continue
        column_ar = check_pattern(column)

        # Part to handle mismatch due to possibility to write same angle as 180° or -180°
        if column_ar.endswith("degree"):
            if (net_df.loc[:, column] < 0).any():
                neg_values_mask = net_df.loc[:, column] < -0.2
                net_df.loc[neg_values_mask, column] += 360
            if (pf_results.loc[:, column] < 0).any():
                neg_values_mask = pf_results.loc[:, column] < -0.2
                pf_results.loc[neg_values_mask, column] += 360

        mismatch = np.isclose(
            net_df.loc[:, column],
            pf_results.loc[:, column],
            rtol=rtol[column_ar], atol=atol[column_ar]
        )
        assert mismatch.all(), (
            f"{column} mismatch for {net_df.loc[~mismatch, 'name']}: {net_df.loc[~mismatch, column]}"
            f"vs {pf_results.loc[~mismatch, column]}"
        )

def load_test_case(net_name: str) -> pandapowerNet:
    if net_name.endswith("_sgen") or net_name.endswith("_sgen_act") or net_name.endswith("_gen"):
        grid_folder = "wp_2.2_2.4"
    else:
        grid_folder = "wp_2.1"

    return from_json(os.path.join(testfiles_path, "test_grids", grid_folder, f"{net_name}.json"))


def load_test_case_data(net_name, fault_location_bus, vector_group=None, gen_idx=None, is_active_current=False,
                        gen_mode=None, grounding_type=None, grounding_bank_idx=None):
    is_gen = gen_idx is not None

    if is_gen:
        wp_folder = "wp_2.2_2.4"
    elif grounding_type is not None:
        wp_folder = "wp_2.5"
    else:
        wp_folder = "wp_2.1"

    if vector_group:
        if is_gen and is_active_current:
            net_name = f"{net_name}_{vector_group.lower()}_{gen_mode}_act"
        elif is_gen:
            net_name = f"{net_name}_{vector_group.lower()}_{gen_mode}"
        elif grounding_type is not None:
            net_name = f"{net_name}_{vector_group.lower()}_gen"
        else:
            net_name = f"{net_name}_{vector_group.lower()}"

    net = load_test_case(net_name)

    xn = None
    rn = None
    if grounding_type is not None:
        if grounding_type == "solid":
            xn = 0
            rn = 0
        elif grounding_type == "resistance":
            xn = 0
            rn = 5
        elif grounding_type == "inductance":
            xn = 5
            rn = 0
        elif grounding_type == "impedance":
            xn = 5
            rn = 5
        elif grounding_type == "isolated":
            xn = 1e99
            rn = 1e99
        elif grounding_type == "resonant":
            xn = 777
            rn = 0

        net.trafo['xn_ohm'] = xn
        net.trafo['rn_ohm'] = rn

    if grounding_bank_idx is not None:
        net.trafo['xn_ohm'] = 1e99
        net.trafo['rn_ohm'] = 1e99
        if grounding_type != "solid":
            # ToDo: Only temporary, ward should be created internally and not only for testing!
            for ward_idx in grounding_bank_idx:
                create_ward(net, ward_idx, 0, 0, 0, 0, "grounding_element",
                            True, rn_ohm=rn, xn_ohm=xn)

    if is_gen:
        if gen_mode == "sgen" or gen_mode == "all":
            net.sgen["k"] = 1.25 if is_active_current else 1.2
        if isinstance(gen_idx, list) | isinstance(gen_idx, tuple):
            if gen_mode == "sgen" or gen_mode == "all":
                net.sgen.loc[net.sgen.bus.isin(gen_idx), 'in_service'] = True
                net.sgen.loc[~net.sgen.bus.isin(gen_idx), 'in_service'] = False
            if gen_mode == "gen" or gen_mode == "all":
                net.gen.loc[net.gen.bus.isin(gen_idx), 'in_service'] = True
                net.gen.loc[~net.gen.bus.isin(gen_idx), 'in_service'] = False
            gen_str = f"_{gen_mode}{''.join(str(idx) for idx in gen_idx)}"
        else:
            if gen_mode == "sgen" or gen_mode == "all":
                net.sgen['in_service'] = False
                net.sgen.loc[net.sgen.bus == gen_idx, 'in_service'] = True
            if gen_mode == "gen" or gen_mode == "all":
                net.gen['in_service'] = False
                net.gen.loc[net.gen.bus == gen_idx, 'in_service'] = True
            gen_str = f"_{gen_mode}{gen_idx}"
    else:
        net.sgen['in_service'] = False
        net.gen['in_service'] = False
        gen_str = ''

    ward_str = ''
    if grounding_bank_idx is not None:
        if isinstance(grounding_bank_idx, list) | isinstance(grounding_bank_idx, tuple):
            ward_str = f"_ward{''.join(str(idx) for idx in grounding_bank_idx)}"
        else:
            ward_str = f"_ward{grounding_bank_idx}"

    if is_active_current:
        net.sgen["active_current"] = True
    else:
        net.sgen["active_current"] = False

    dataframes = {}
    for bb in ['bus', 'branch']:
        file_name = f"{net_name}_pf_sc_results_{fault_location_bus}_{bb}{gen_str}.xlsx"
        if grounding_type is not None:
            file_name = f"{net_name}_pf_sc_results_{fault_location_bus}_{bb}{gen_str}_{grounding_type}.xlsx"
            if grounding_bank_idx is not None:
                file_name = f"{net_name}_pf_sc_results_{fault_location_bus}_{bb}{ward_str}_{grounding_type}.xlsx"
        if gen_mode == 'sgen':
            file_name = file_name.replace('_gen_', '_sgen_')
        try:
            dataframes[bb] = load_pf_results(os.path.join(
                testfiles_path,
                "sc_result_comparison", wp_folder,
                file_name
            ))
        except FileNotFoundError:
            logger.warning(f"File {file_name} not found in {testfiles_path}/sc_result_comparison/")
    return net, dataframes


def run_test_cases(net, dataframes, fault, case, fault_values, lv_tol_percent, fault_location_bus,
                   branch_results=False):
    """
    Executes test cases for a given grid with specific fault parameters, fault type and case.

    This function takes a network, corresponding dataframes, and fault parameters, performs
    a fault analysis, and compares the results with the modified power flow results.

    :param pandapowerNet net: An object representing the electrical network, including bus and branch information.
    :param dataframes: A dictionary of DataFrames containing the power flow results,
                       where the key indicates the name of the sheet.
    :param str fault: A string indicating the type of fault. LLL, LLG, LL or LG
    :param str case: A string indicating the specific case. min or max
    :param tuple[float, float] fault_values: The resistive and reactive fault value in Ohms.
    :param int fault_location_bus: index of the bus the fault is located at.
    :param bool branch_results: A boolean indicating whether branch results are calculated or not.

    :raises AssertionError: If the calculated values are not within the specified tolerances.

    :note:
        - The function uses a relative tolerance (rtol) and an absolute tolerance (atol)
          for value comparisons.
        - The name of the selected sheet is generated from the fault and case.
        - If both the resistive and reactive faults are non-zero, "_fault" is appended
          to the sheet name.
        - The function sorts the results by bus name for comparison.
    """
    r_fault_ohm, x_fault_ohm = fault_values

    # columns_to_check = get_columns_to_check(fault)
    selected_sheet = f"{fault}_{case}_{lv_tol_percent}"
    if r_fault_ohm != 0.0 and x_fault_ohm != 0.0:
        selected_sheet = f"{fault}_{case}_fault_{lv_tol_percent}"

    selected_pf_results = dataframes[selected_sheet]
    modified_pf_results = modify_impedance_values_with_fault_value(selected_pf_results, r_fault_ohm, x_fault_ohm)

    calc_sc(net, bus=fault_location_bus, fault=fault, case=case, branch_results=branch_results,
            return_all_currents=False, ip=False,
            r_fault_ohm=r_fault_ohm, x_fault_ohm=x_fault_ohm, lv_tol_percent=lv_tol_percent)

    if branch_results:
        columns_to_check = net.res_line_sc.columns
        if fault == "LG" or fault == "LLG" or fault == "LL":
            if branch_results:
                patterns_to_drop = ["ikss_ka"]  # ToDo: Do we need the value ikss_ka ?
                columns_to_check = columns_to_check[~columns_to_check.isin(patterns_to_drop)]
        net.res_line_sc["name"] = net.line.name
        net.res_line_sc = net.res_line_sc[['name'] + [col for col in net.res_line_sc.columns if col != 'name']]
        net.res_line_sc.sort_values(by='name', inplace=True)

        modified_pf_results_selection = modified_pf_results[modified_pf_results['name'].isin(net.res_line_sc['name'])]
        modified_pf_results_selection = clean_small_angles(fault, modified_pf_results_selection)
        net_df = net.res_line_sc
        if fault == 'LLL':
            cols_to_drop = net_df.filter(regex=r'_(b|c)_').columns
            net_df = net_df.drop(columns=cols_to_drop)
            columns_to_check = net_df.columns
    else:
        columns_to_check = net.res_bus_sc.columns
        net.res_bus_sc.insert(0, "name", net.bus.name)
        net.res_bus_sc.sort_values(by='name', inplace=True)

        # shorten the results from the file to the ones calculated by pp.
        # This was done by only iterating over the net table before.
        modified_pf_results_selection = modified_pf_results[modified_pf_results['name'].isin(net.res_bus_sc['name'])]
        net_df = net.res_bus_sc
        if fault == 'LLL':
            cols_to_drop = net_df.filter(regex=r'_(b|c)_').columns
            net_df = net_df.drop(columns=cols_to_drop)
            columns_to_check = net_df.columns

    return columns_to_check, net_df, modified_pf_results_selection


def clean_small_angles(fault, results):
    if fault in ["LG", "LLG", "LL"]:
        for [mag, ang] in ("ikss_a_from_ka", "ikss_a_from_degree"), \
                ("ikss_b_from_ka", "ikss_b_from_degree"), \
                ("ikss_c_from_ka", "ikss_c_from_degree"), \
                ("ikss_a_to_ka", "ikss_a_to_degree"), \
                ("ikss_b_to_ka", "ikss_b_to_degree"), \
                ("ikss_c_to_ka", "ikss_c_to_degree"), \
                ("vm_a_from_pu", "va_a_from_degree"), \
                ("vm_b_from_pu", "va_b_from_degree"), \
                ("vm_c_from_pu", "va_c_from_degree"), \
                ("vm_a_to_pu", "va_a_to_degree"), \
                ("vm_b_to_pu", "va_b_to_degree"), \
                ("vm_c_to_pu", "va_c_to_degree"):
            results[ang][results[mag] < 1e-5] = 0
    else:
        for [mag, ang] in ("ikss_a_from_ka", "ikss_a_from_degree"), \
                ("ikss_a_to_ka", "ikss_a_to_degree"), \
                ("vm_a_from_pu", "va_a_from_degree"), \
                ("vm_a_to_pu", "va_a_to_degree"):
            results[ang][results[mag] < 1e-5] = 0

    return results


def check_pattern(pattern):
    """
    Maps detailed result column names to standardized identifiers.

    Recognizes patterns like 'ikss_a_ka', 'skss_from_mw', 'p_b_to_mw', etc.
    and maps them to generic identifiers used for tolerance comparison.

    :param str pattern: Column name to normalize.
    :returns str: Standardized identifier for result type or original pattern if no match found.
    """
    if match(r"^rk[0-2]?_ohm$", pattern):
        return "rk_ohm"
    elif match(r"^xk[0-2]?_ohm$", pattern):
        return "xk_ohm"
    elif match(r"^ikss_([abc]|from|to|[abc]_(from|to))_ka$", pattern):
        return "ikss_ka"
    elif match(r"^ikss_([abc]|from|to|[abc]_(from|to))_degree$", pattern):
        return "ikss_degree"
    elif match(r"^skss_([abc]|from|to|[abc]_(from|to))_mw$", pattern):
        return "skss_mw"
    elif match(r"^p_([abc]|from|to|[abc]_(from|to))_mw$", pattern):
        return "p_mw"
    elif match(r"^q_([abc]|from|to|[abc]_(from|to))_mvar$", pattern):
        return "q_mvar"
    elif match(r"^vm_([abc]|from|to|[abc]_(from|to))_pu$", pattern):
        return "vm_pu"
    elif match(r"^va_([abc]|from|to|[abc]_(from|to))_degree$", pattern):
        return "va_degree"
    else:
        return pattern


def modify_impedance_values_with_fault_value(selected_results, r_ohm, x_ohm):
    """
    Modifies the impedance values in a DataFrame by subtracting r_ohm from rk columns
    and x_ohm from xk columns.

    :param pd.DataFrame selected_results: The input DataFrame containing impedance values.
    :param float r_ohm: The value to be subtracted from the rk columns.
    :param float x_ohm: The value to be subtracted from the xk columns.

    :returns pd.DataFrame: The modified DataFrame with adjusted values.
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
    sheets = [sheet for sheet in pd.ExcelFile(excel_file).sheet_names]
    dataframes = {}

    # Dictionary with columns to keep for each fault type
    columns_mapping = {
        "LLL": ['name', 'pf_ikss_ka', 'pf_skss_mw', 'pf_rk_ohm', 'pf_xk_ohm'],
        "LL": ['name', 'pf_ikss_c_ka', 'pf_skss_c_mw',
               'pf_rk0_ohm', 'pf_xk0_ohm', 'pf_rk1_ohm', 'pf_xk1_ohm', 'pf_rk2_ohm', 'pf_xk2_ohm'],
        "LLG": ['name', 'pf_ikss_a_ka', 'pf_ikss_b_ka', 'pf_ikss_c_ka', 'pf_skss_a_mw', 'pf_skss_b_mw', 'pf_skss_c_mw',
                'pf_rk0_ohm', 'pf_xk0_ohm', 'pf_rk1_ohm', 'pf_xk1_ohm', 'pf_rk2_ohm', 'pf_xk2_ohm'],
        "LG": ['name', 'pf_ikss_a_ka', 'pf_skss_a_mw',
               'pf_rk0_ohm', 'pf_xk0_ohm', 'pf_rk1_ohm', 'pf_xk1_ohm', 'pf_rk2_ohm', 'pf_xk2_ohm'],
    }

    columns_mapping_branch = {
        "LLL": ['name', 'pf_ikss_from_ka', 'pf_ikss_from_ka', 'pf_ikss_from_degree', 'pf_ikss_to_ka',
                'pf_ikss_to_degree',
                'pf_p_from_mw', 'pf_q_from_mvar', 'pf_p_to_mw', 'pf_q_to_mvar',
                'pf_vm_from_pu', 'pf_va_from_degree', 'pf_vm_to_pu', 'pf_va_to_degree'],
        "LL": ['name', 'pf_ikss_a_from_ka', 'pf_ikss_a_from_ka', 'pf_ikss_a_from_degree', 'pf_ikss_a_to_ka',
               'pf_ikss_a_to_degree',
               'pf_p_a_from_mw', 'pf_q_a_from_mvar', 'pf_p_a_to_mw', 'pf_q_a_to_mvar',
               'pf_vm_a_from_pu', 'pf_va_a_from_degree', 'pf_vm_a_to_pu', 'pf_va_a_to_degree',
               'pf_ikss_b_from_ka', 'pf_ikss_b_from_degree', 'pf_ikss_b_to_ka', 'pf_ikss_b_to_degree',
               'pf_p_b_from_mw', 'pf_q_b_from_mvar', 'pf_p_b_to_mw', 'pf_q_b_to_mvar',
               'pf_vm_b_from_pu', 'pf_va_b_from_degree', 'pf_vm_b_to_pu', 'pf_va_b_to_degree',
               'pf_ikss_c_from_ka', 'pf_ikss_c_from_degree', 'pf_ikss_c_to_ka', 'pf_ikss_c_to_degree',
               'pf_p_c_from_mw', 'pf_q_c_from_mvar', 'pf_p_c_to_mw', 'pf_q_c_to_mvar',
               'pf_vm_c_from_pu', 'pf_va_c_from_degree', 'pf_vm_c_to_pu', 'pf_va_c_to_degree'],
        "LLG": ['name', 'pf_ikss_a_from_ka', 'pf_ikss_a_from_ka', 'pf_ikss_a_from_degree', 'pf_ikss_a_to_ka',
                'pf_ikss_a_to_degree',
                'pf_p_a_from_mw', 'pf_q_a_from_mvar', 'pf_p_a_to_mw', 'pf_q_a_to_mvar',
                'pf_vm_a_from_pu', 'pf_va_a_from_degree', 'pf_vm_a_to_pu', 'pf_va_a_to_degree',
                'pf_ikss_b_from_ka', 'pf_ikss_b_from_degree', 'pf_ikss_b_to_ka', 'pf_ikss_b_to_degree',
                'pf_p_b_from_mw', 'pf_q_b_from_mvar', 'pf_p_b_to_mw', 'pf_q_b_to_mvar',
                'pf_vm_b_from_pu', 'pf_va_b_from_degree', 'pf_vm_b_to_pu', 'pf_va_b_to_degree',
                'pf_ikss_c_from_ka', 'pf_ikss_c_from_degree', 'pf_ikss_c_to_ka', 'pf_ikss_c_to_degree',
                'pf_p_c_from_mw', 'pf_q_c_from_mvar', 'pf_p_c_to_mw', 'pf_q_c_to_mvar',
                'pf_vm_c_from_pu', 'pf_va_c_from_degree', 'pf_vm_c_to_pu', 'pf_va_c_to_degree'],
        "LG": ['name', 'pf_ikss_a_from_ka', 'pf_ikss_a_from_ka', 'pf_ikss_a_from_degree', 'pf_ikss_a_to_ka',
               'pf_ikss_a_to_degree',
               'pf_p_a_from_mw', 'pf_q_a_from_mvar', 'pf_p_a_to_mw', 'pf_q_a_to_mvar',
               'pf_vm_a_from_pu', 'pf_va_a_from_degree', 'pf_vm_a_to_pu', 'pf_va_a_to_degree',
               'pf_ikss_b_from_ka', 'pf_ikss_b_from_degree', 'pf_ikss_b_to_ka', 'pf_ikss_b_to_degree',
               'pf_p_b_from_mw', 'pf_q_b_from_mvar', 'pf_p_b_to_mw', 'pf_q_b_to_mvar',
               'pf_vm_b_from_pu', 'pf_va_b_from_degree', 'pf_vm_b_to_pu', 'pf_va_b_to_degree',
               'pf_ikss_c_from_ka', 'pf_ikss_c_from_degree', 'pf_ikss_c_to_ka', 'pf_ikss_c_to_degree',
               'pf_p_c_from_mw', 'pf_q_c_from_mvar', 'pf_p_c_to_mw', 'pf_q_c_to_mvar',
               'pf_vm_c_from_pu', 'pf_va_c_from_degree', 'pf_vm_c_to_pu', 'pf_va_c_to_degree']
    }

    if '_resonant' in excel_file:
        columns_mapping['LG'].append('3xI0')
        columns_mapping['LLG'].append('3xI0')
        columns_mapping_branch['LG'].append('3xI0')
        columns_mapping_branch['LLG'].append('3xI0')

    for sheet in sheets:
        pf_results = pd.read_excel(excel_file, sheet_name=sheet)
        fault_type = None
        if sheet.startswith("LLL_"):
            fault_type = "LLL"
        elif sheet.startswith("LL_"):
            fault_type = "LL"
        elif sheet.startswith("LLG_"):
            fault_type = "LLG"
        elif sheet.startswith("LG_"):
            fault_type = "LG"
        parts_excel_file = excel_file.split('_')
        if 'bus' in parts_excel_file and 'branch' not in parts_excel_file and 'branch.xlsx' not in parts_excel_file:
            relevant_columns = columns_mapping[fault_type]
            pf_results = pf_results[relevant_columns]
            pf_results['name'] = pf_results['name'].astype(str)
            if fault_type == 'LLL':
                pf_results.columns = ['name', 'ikss_ka', 'skss_mw', 'rk_ohm', 'xk_ohm']
            if fault_type == 'LL':
                pf_results.columns = ['name', 'ikss_ka', 'skss_mw',
                                      "rk0_ohm", "xk0_ohm", "rk1_ohm", "xk1_ohm", "rk2_ohm", "xk2_ohm"]
            elif fault_type == 'LLG':
                if '_resonant' in excel_file:
                    pf_results.columns = ["name", "ikss_a_ka", "ikss_b_ka", 'ikss_c_ka', 'skss_a_mw', 'skss_b_mw',
                                          'skss_c_mw',
                                          "rk0_ohm", "xk0_ohm", "rk1_ohm", "xk1_ohm", "rk2_ohm", "xk2_ohm", "3xI0"]
                else:
                    pf_results.columns = ["name", "ikss_a_ka", "ikss_b_ka", 'ikss_c_ka', 'skss_a_mw', 'skss_b_mw',
                                          'skss_c_mw',
                                          "rk0_ohm", "xk0_ohm", "rk1_ohm", "xk1_ohm", "rk2_ohm", "xk2_ohm"]
            elif fault_type == 'LG':
                if '_resonant' in excel_file:
                    pf_results.columns = ["name", "ikss_ka", 'skss_mw',
                                          "rk0_ohm", "xk0_ohm", "rk1_ohm", "xk1_ohm", "rk2_ohm", "xk2_ohm", "3xI0"]
                else:
                    pf_results.columns = ["name", "ikss_ka", 'skss_mw',
                                        "rk0_ohm", "xk0_ohm", "rk1_ohm", "xk1_ohm", "rk2_ohm", "xk2_ohm"]


            dataframes[sheet] = pf_results

        elif 'branch' in parts_excel_file or 'branch.xlsx' in parts_excel_file:
            relevant_columns = columns_mapping_branch[fault_type]
            pf_results = pf_results[relevant_columns]
            if fault_type == 'LLL':
                pf_results.columns = ['name', 'ikss_ka', 'ikss_a_from_ka', 'ikss_a_from_degree', 'ikss_a_to_ka',
                                      'ikss_a_to_degree',
                                      'p_a_from_mw', 'q_a_from_mvar', 'p_a_to_mw', 'q_a_to_mvar',
                                      'vm_a_from_pu', 'va_a_from_degree', 'vm_a_to_pu', 'va_a_to_degree']

            elif fault_type == 'LLG' or fault_type == 'LG' or fault_type == 'LL':
                if '_resonant' in excel_file and fault_type in ['LLG', 'LG']:
                    pf_results.columns = ['name', 'ikss_ka', 'ikss_a_from_ka', 'ikss_a_from_degree', 'ikss_a_to_ka',
                                          'ikss_a_to_degree',
                                          'p_a_from_mw', 'q_a_from_mvar', 'p_a_to_mw', 'q_a_to_mvar',
                                          'vm_a_from_pu', 'va_a_from_degree', 'vm_a_to_pu', 'va_a_to_degree',
                                          'ikss_b_from_ka', 'ikss_b_from_degree', 'ikss_b_to_ka', 'ikss_b_to_degree',
                                          'p_b_from_mw', 'q_b_from_mvar', 'p_b_to_mw', 'q_b_to_mvar',
                                          'vm_b_from_pu', 'va_b_from_degree', 'vm_b_to_pu', 'va_b_to_degree',
                                          'ikss_c_from_ka', 'ikss_c_from_degree', 'ikss_c_to_ka', 'ikss_c_to_degree',
                                          'p_c_from_mw', 'q_c_from_mvar', 'p_c_to_mw', 'q_c_to_mvar',
                                          'vm_c_from_pu', 'va_c_from_degree', 'vm_c_to_pu', 'va_c_to_degree',
                                          '3xI0']
                else:
                    pf_results.columns = ['name', 'ikss_ka', 'ikss_a_from_ka', 'ikss_a_from_degree', 'ikss_a_to_ka',
                                          'ikss_a_to_degree',
                                          'p_a_from_mw', 'q_a_from_mvar', 'p_a_to_mw', 'q_a_to_mvar',
                                          'vm_a_from_pu', 'va_a_from_degree', 'vm_a_to_pu', 'va_a_to_degree',
                                          'ikss_b_from_ka', 'ikss_b_from_degree', 'ikss_b_to_ka', 'ikss_b_to_degree',
                                          'p_b_from_mw', 'q_b_from_mvar', 'p_b_to_mw', 'q_b_to_mvar',
                                          'vm_b_from_pu', 'va_b_from_degree', 'vm_b_to_pu', 'va_b_to_degree',
                                          'ikss_c_from_ka', 'ikss_c_from_degree', 'ikss_c_to_ka', 'ikss_c_to_degree',
                                          'p_c_from_mw', 'q_c_from_mvar', 'p_c_to_mw', 'q_c_to_mvar',
                                          'vm_c_from_pu', 'va_c_from_degree', 'vm_c_to_pu', 'va_c_to_degree',
                                          ]
                # ToDo: Do we need the value ikss_ka ?
                pf_results['ikss_ka'] = np.max(
                    [pf_results['ikss_a_from_ka'], pf_results['ikss_a_to_ka'], pf_results['ikss_b_from_ka'],
                     pf_results['ikss_b_to_ka'], pf_results['ikss_c_from_ka'], pf_results['ikss_c_to_ka']], axis=0)

            dataframes[sheet] = pf_results

    return dataframes

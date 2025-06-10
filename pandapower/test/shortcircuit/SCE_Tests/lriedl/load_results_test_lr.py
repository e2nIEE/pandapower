import pandas as pd



excel_file = r"/pandapower/test/shortcircuit/SCE_Tests/sc_result_comparison/test_case_1_four_bus_radial_grid_pf_sc_results_branch.xlsx"


def load_pf_results(excel_file):
    """Load power flow results from Excel sheets."""
    # TODO also include branch results and check all dropped columns --> Done :)
    sheets = [sheet for sheet in pd.ExcelFile(excel_file).sheet_names]
    dataframes = {}

    # Dictionary with columns to keep for each fault type
    columns_mapping = {
        "LLL": ['name', 'pf_ikss_ka', 'pf_skss_mw', 'pf_rk_ohm', 'pf_xk_ohm'],
        "LL": ['name', 'pf_ikss_c_ka', 'pf_skss_c_mw', 'pf_rk2_ohm', 'pf_xk2_ohm'],
        "LLG": ['name', 'pf_ikss_a_ka', 'pf_ikss_b_ka', 'pf_ikss_c_ka', 'pf_skss_a_mw', 'pf_skss_b_mw', 'pf_skss_c_mw',
                'pf_rk0_ohm', 'pf_xk0_ohm', 'pf_rk1_ohm', 'pf_xk1_ohm', 'pf_rk2_ohm', 'pf_xk2_ohm'],
        "LG": ['name', 'pf_ikss_a_ka', 'pf_skss_a_mw', 'pf_rk0_ohm', 'pf_xk0_ohm', 'pf_rk1_ohm', 'pf_xk1_ohm',
               'pf_rk2_ohm', 'pf_xk2_ohm']
    }

    columns_mapping_branch = {
        "LLL": ['pf_ikss_from_ka', 'pf_ikss_from_ka', 'pf_ikss_from_degree', 'pf_ikss_to_ka', 'pf_ikss_to_degree',
                'pf_p_from_mw', 'pf_q_from_mvar', 'pf_p_to_mw', 'pf_q_to_mvar',
                'pf_vm_from_pu', 'pf_va_from_degree', 'pf_vm_to_pu', 'pf_va_to_degree'],
        "LL": ['pf_ikss_c_from_ka', 'pf_ikss_c_from_ka', 'pf_ikss_c_from_degree', 'pf_ikss_c_to_ka', 'pf_ikss_c_to_degree',
                'pf_p_c_from_mw', 'pf_q_c_from_mvar', 'pf_p_c_to_mw', 'pf_q_c_to_mvar',
                'pf_vm_c_from_pu', 'pf_va_c_from_degree', 'pf_vm_c_to_pu', 'pf_va_c_to_degree'],
        "LLG": ['pf_ikss_a_from_ka', 'pf_ikss_a_from_ka', 'pf_ikss_a_from_degree', 'pf_ikss_a_to_ka', 'pf_ikss_a_to_degree',
                'pf_p_a_from_mw', 'pf_q_a_from_mvar', 'pf_p_a_to_mw', 'pf_q_a_to_mvar',
                'pf_vm_a_from_pu', 'pf_va_a_from_degree', 'pf_vm_a_to_pu', 'pf_va_a_to_degree',
                'pf_ikss_b_from_ka', 'pf_ikss_b_from_degree', 'pf_ikss_b_to_ka', 'pf_ikss_b_to_degree',
                'pf_p_b_from_mw', 'pf_q_b_from_mvar', 'pf_p_b_to_mw', 'pf_q_b_to_mvar',
                'pf_vm_b_from_pu', 'pf_va_b_from_degree', 'pf_vm_b_to_pu', 'pf_va_b_to_degree',
                'pf_ikss_c_from_ka', 'pf_ikss_c_from_degree', 'pf_ikss_c_to_ka', 'pf_ikss_c_to_degree',
                'pf_p_c_from_mw', 'pf_q_c_from_mvar', 'pf_p_c_to_mw', 'pf_q_c_to_mvar',
                'pf_vm_c_from_pu', 'pf_va_c_from_degree', 'pf_vm_c_to_pu', 'pf_va_c_to_degree'],
        "LG": ['pf_ikss_a_from_ka', 'pf_ikss_a_from_ka', 'pf_ikss_a_from_degree', 'pf_ikss_a_to_ka', 'pf_ikss_a_to_degree',
                'pf_p_a_from_mw', 'pf_q_a_from_mvar', 'pf_p_a_to_mw', 'pf_q_a_to_mvar',
                'pf_vm_a_from_pu', 'pf_va_a_from_degree', 'pf_vm_a_to_pu', 'pf_va_a_to_degree',
                'pf_ikss_b_from_ka', 'pf_ikss_b_from_degree', 'pf_ikss_b_to_ka', 'pf_ikss_b_to_degree',
                'pf_p_b_from_mw', 'pf_q_b_from_mvar', 'pf_p_b_to_mw', 'pf_q_b_to_mvar',
                'pf_vm_b_from_pu', 'pf_va_b_from_degree', 'pf_vm_b_to_pu', 'pf_va_b_to_degree',
                'pf_ikss_c_from_ka', 'pf_ikss_c_from_degree', 'pf_ikss_c_to_ka', 'pf_ikss_c_to_degree',
                'pf_p_c_from_mw', 'pf_q_c_from_mvar', 'pf_p_c_to_mw', 'pf_q_c_to_mvar',
                'pf_vm_c_from_pu', 'pf_va_c_from_degree', 'pf_vm_c_to_pu', 'pf_va_c_to_degree']
    }

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

        if excel_file.endswith('_bus.xlsx'):
            relevant_columns = columns_mapping[fault_type]
            pf_results = pf_results[relevant_columns]
            if fault_type == 'LLL' or fault_type == 'LL':
                pf_results.columns = ['name', 'ikss_ka', 'skss_mw', 'rk_ohm', 'xk_ohm']
            elif fault_type == 'LLG':
                pf_results.columns = ["name", "ikss_a_ka", "ikss_b_ka", 'ikss_c_ka', 'skss_a_mw', 'skss_b_mw', 'skss_c_mw',
                                      "rk0_ohm", "xk0_ohm", "rk1_ohm", "xk1_ohm", "rk2_ohm", "xk2_ohm"]
            elif fault_type == 'LG':
                pf_results.columns = ["name", "ikss_ka", 'skss_mw', "rk0_ohm", "xk0_ohm", "rk1_ohm",
                                      "xk1_ohm", "rk2_ohm", "xk2_ohm"]

            dataframes[sheet] = pf_results

        elif excel_file.endswith('_branch.xlsx'):
            relevant_columns = columns_mapping_branch[fault_type]
            pf_results = pf_results[relevant_columns]
            if fault_type == 'LLL' or fault_type == 'LL':
                pf_results.columns = ['ikss_ka', 'ikss_from_ka', 'ikss_from_degree', 'ikss_to_ka', 'ikss_to_degree',
                                      'p_from_mw', 'q_from_mvar', 'p_to_mw', 'q_to_mvar',
                                      'vm_from_pu', 'va_from_degree', 'vm_to_pu', 'va_to_degree']
            elif fault_type == 'LLG' or fault_type == 'LG':
                pf_results.columns = ['ikss_ka', 'ikss_a_from_ka', 'ikss_a_from_degree', 'ikss_a_to_ka', 'ikss_a_to_degree',
                                      'p_a_from_mw', 'q_a_from_mvar', 'p_a_to_mw', 'q_a_to_mvar',
                                      'vm_a_from_pu', 'va_a_from_degree', 'vm_a_to_pu', 'va_a_to_degree',
                                      'ikss_b_from_ka', 'ikss_b_from_degree', 'ikss_b_to_ka', 'ikss_b_to_degree',
                                      'p_b_from_mw', 'q_b_from_mvar', 'p_b_to_mw', 'q_b_to_mvar',
                                      'vm_b_from_pu', 'va_b_from_degree', 'vm_b_to_pu', 'va_b_to_degree',
                                      'ikss_c_from_ka', 'ikss_c_from_degree', 'ikss_c_to_ka', 'ikss_c_to_degree',
                                      'p_c_from_mw', 'q_c_from_mvar', 'p_c_to_mw', 'q_c_to_mvar',
                                      'vm_c_from_pu', 'va_c_from_degree', 'vm_c_to_pu', 'va_c_to_degree',
                                      ]

            dataframes[sheet] = pf_results

    return dataframes

##
df = load_pf_results(excel_file)
##


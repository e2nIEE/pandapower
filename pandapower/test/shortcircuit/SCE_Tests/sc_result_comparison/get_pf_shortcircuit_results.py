import sys
import pandas as pd
from pandapower.test.shortcircuit.SCE_Tests.sc_result_comparison.pf_shortcircuit_analysis import PFShortCircuitAnalysis
from pandapower import pp_dir
import os

sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP1\Python\3.11")
try:
    import powerfactory as pf
except:
    pass

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging
app = pf.GetApplication()
testfiles_path = os.path.join(pp_dir, 'test', 'shortcircuit', 'sce_tests')


## Function to export and save all pf short circuit results for a given project
def get_all_pf_sc_results(proj_name, fault_location=None, save_to_excel=True):
    fault_types = ["LLL", "LL", "LG", "LLG"]
    cases = ["max", "min"]
    fault_impedances = [(0.0, 0.0), (5.0, 5.0)]
    lv_tol_percents = [6, 10]
    fault_locations = [fault_location] if fault_location is None else fault_location
    elements = ['bus', 'branch']

    dict_results = {}
    for element in elements:
        dict_results[element] = {}

        if save_to_excel:
            out_path = os.path.join(pp_dir, "test", "shortcircuit", "sce_tests", "sc_result_comparison",
                                    proj_name + f'_pf_sc_results_{element}.xlsx')
            if fault_location is not None:
                out_path = os.path.join(pp_dir, "test", "shortcircuit", "sce_tests", "sc_result_comparison",
                                    proj_name + f'_pf_sc_results_{fault_location}_{element}.xlsx')
            writer = pd.ExcelWriter(out_path)

        for fault_type in fault_types:
            for case in cases:
                for fault_impedance in fault_impedances:
                    for lv_tol_percent in lv_tol_percents:
                        for fault_location in fault_locations:
                            pf_analysis = PFShortCircuitAnalysis(
                                app=app, proj_name=proj_name, fault_type=fault_type, calc_mode=case,
                                fault_impedance_rf=fault_impedance[0], fault_impedance_xf=fault_impedance[1],
                                lv_tol_percent=lv_tol_percent, fault_location_index=fault_location
                            )
                            if element == 'bus':
                                df = pf_analysis.get_pf_sc_bus_results()
                            elif element == 'branch':
                                df = pf_analysis.get_pf_sc_branch_results()
                            sheet_name_base = pf_analysis.get_case_name()
                            dict_results[element][sheet_name_base] = df

                            if save_to_excel:
                                df.to_excel(writer, sheet_name=sheet_name_base, index=False)

        if save_to_excel:
            writer.close()

    return dict_results


## get results for single project
proj_name = 'test_case_1_four_bus_radial_grid'
fault_location = [3]
pf_dict = get_all_pf_sc_results(proj_name, fault_location, save_to_excel=False)


## get results for all projects
base_dir = os.getcwd()
folder = os.path.join(base_dir, "pandapower", "test", "shortcircuit", "SCE_Tests", "test_grids")
#folder = r"C:\Users\lriedl\PycharmProjects\pandapower\pandapower\test\shortcircuit\SCE_Tests\test_grids"
pfd_files = [f for f in os.listdir(folder) if f.endswith(".pfd")]
fault_location = [3]

pf_dict_all = {}
for file in pfd_files:
    proj_name = os.path.splitext(file)[0]
    pf_dict = get_all_pf_sc_results(proj_name, fault_location, save_to_excel=True)
    pf_dict_all[proj_name] = pf_dict

##


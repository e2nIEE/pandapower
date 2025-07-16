import sys
import pandas as pd
from pandapower.test.shortcircuit.sce_tests.sc_result_comparison.pf_shortcircuit_analysis import PFShortCircuitAnalysis
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
def get_all_pf_sc_results(proj_name, fault_location=None, activate_sgen=None, save_to_excel=True):
    fault_types = ["LLL", "LL", "LG", "LLG"]
    cases = ["max", "min"]
    fault_impedances = [(0.0, 0.0), (5.0, 5.0)]
    lv_tol_percents = [6, 10]
    if fault_location is None:
        fault_locations = [None]
    elif isinstance(fault_location, (list, tuple)):
        fault_locations = fault_location
    else:
        fault_locations = [fault_location]
    if not isinstance(activate_sgen, (list, tuple)):
        activate_sgen = [activate_sgen]
    elements = ['bus', 'branch']

    dict_results = {}
    for element in elements:
        dict_results[element] = {}

        if save_to_excel:
            out_path = os.path.join(pp_dir, "test", "shortcircuit", "sce_tests", "sc_result_comparison", "wp_2.1",
                                    proj_name + f'_pf_sc_results_{element}.xlsx')
            if fault_location is not None:
                out_path = os.path.join(pp_dir, "test", "shortcircuit", "sce_tests", "sc_result_comparison", "wp_2.1",
                                    proj_name + f'_pf_sc_results_{fault_location}_{element}.xlsx')
            if activate_sgen is not None:
                sgen_names = ''
                for active_sgen in activate_sgen:
                    sgen_name = str(active_sgen)
                    sgen_names += sgen_name
                out_path = os.path.join(pp_dir, "test", "shortcircuit", "sce_tests", "sc_result_comparison", "wp_2.2",
                                    proj_name + f'_pf_sc_results_{fault_location}_{element}_sgen{sgen_names}.xlsx')
            writer = pd.ExcelWriter(out_path)

        for fault_type in fault_types:
            for case in cases:
                for fault_impedance in fault_impedances:
                    for lv_tol_percent in lv_tol_percents:
                        for fault_location in fault_locations:
                            pf_analysis = PFShortCircuitAnalysis(
                                app=app, proj_name=proj_name, fault_type=fault_type, calc_mode=case,
                                fault_impedance_rf=fault_impedance[0], fault_impedance_xf=fault_impedance[1],
                                lv_tol_percent=lv_tol_percent, fault_location_index=fault_location,
                                activate_sgens_at_bus=activate_sgen
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
proj_name = '1_four_bus_radial_grid_sgen_act'
fault_location = [0,1,2,3]
activate_sgen = [1, 3]
for fl in fault_location:
    pf_dict = get_all_pf_sc_results(proj_name, fl, activate_sgen=activate_sgen, save_to_excel=True)


## get results for all projects
# base_dir = os.getcwd()
# folder = os.path.join(base_dir, "pandapower", "test", "shortcircuit", "SCE_Tests", "test_grids")
#folder = r"C:\Users\lriedl\PycharmProjects\pandapower\pandapower\test\shortcircuit\SCE_Tests\test_grids"
folder = os.path.join(pp_dir, "test", "shortcircuit", "sce_tests", "test_grids", "wp_2.2")
pfd_files = [f for f in os.listdir(folder) if f.endswith(".pfd")]

activate_sgen = [3, 4]
for fault_location in [0, 1, 2, 3]:
    pf_dict_all = {}
    for file in pfd_files:
        proj_name = os.path.splitext(file)[0]
        pf_dict = get_all_pf_sc_results(proj_name, fault_location, activate_sgen=activate_sgen, save_to_excel=True)
        pf_dict_all[proj_name] = pf_dict

##


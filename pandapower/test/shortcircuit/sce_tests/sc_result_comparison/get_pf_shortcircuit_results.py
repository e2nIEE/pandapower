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
def get_all_pf_sc_results(proj_name, fault_location=None, activate_sgen=None, activate_gen=None,
                          grounding_type=None, save_to_excel=True):
    fault_types = ["LLL", "LL", "LG", "LLG"]
    cases = ["max", "min"]
    fault_impedances = [(0.0, 0.0), (5.0, 5.0)]
    lv_tol_percents = [6, 10]
    activate_all = True if activate_sgen is not None and activate_gen is not None else False
    if fault_location is None:
        fault_locations = [None]
    elif isinstance(fault_location, (list, tuple)):
        fault_locations = fault_location
    else:
        fault_locations = [fault_location]
    if not isinstance(activate_sgen, (list, tuple)) and activate_sgen is not None:
        activate_sgen = [activate_sgen]
    if not isinstance(activate_gen, (list, tuple)) and activate_gen is not None:
        activate_gen = [activate_gen]
    elements = ['bus', 'branch']

    dict_results = {}
    for element in elements:
        dict_results[element] = {}

        if save_to_excel:
            out_path = os.path.join(testfiles_path, "sc_result_comparison", "wp_2.1",
                                    proj_name + f'_pf_sc_results_{element}.xlsx')
            if fault_location is not None:
                out_path = os.path.join(testfiles_path, "sc_result_comparison", "wp_2.1",
                                        proj_name + f'_pf_sc_results_{fault_location}_{element}.xlsx')
            if activate_sgen is not None or activate_gen is not None:
                gen_names = ''
                if activate_all:
                    for active_sgen in activate_sgen:
                        gen_name = str(active_sgen)
                        gen_names += gen_name
                        elm_name = '_all'
                elif activate_sgen is not None:
                    for active_sgen in activate_sgen:
                        gen_name = str(active_sgen)
                        gen_names += gen_name
                        elm_name = '_sgen'
                elif activate_gen is not None:
                    for active_gen in activate_gen:
                        gen_name = str(active_gen)
                        gen_names += gen_name
                        elm_name = '_gen'
                out_path = os.path.join(testfiles_path, "sc_result_comparison", "wp_2.2_2.4",
                                        proj_name + f'_pf_sc_results_{fault_location}_{element}{elm_name}{gen_names}.xlsx')
            if grounding_type is not None:
                out_path = os.path.join(testfiles_path, "sc_result_comparison", "wp_2.5",
                                        proj_name + f'_pf_sc_results_{fault_location}_{element}_{grounding_type}.xlsx')
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
                                activate_sgens_at_bus=activate_sgen, activate_gens_at_bus=activate_gen,
                                grounding_type=grounding_type
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
proj_name = '2_five_bus_radial_grid_dyn_gen'
fault_location = [0, 8, 18]
activate_sgen = None
activate_gen = None
grounding_types = ["isolated"]  # "solid", "resistance", "inductance", "impedance", "resonant", "isolated"
proj_names = ['4_twenty_bus_radial_grid_dyn_gen', '4_twenty_bus_radial_grid_yyn_gen', '4_twenty_bus_radial_grid_ynyn_gen']

for proj_name in proj_names:
    for fl in fault_location:
        for grounding_type in grounding_types:
            pf_dict = get_all_pf_sc_results(proj_name, fl, activate_sgen=activate_sgen, activate_gen=activate_gen,
                                            grounding_type=grounding_type, save_to_excel=True)

## get results for all projects
folder = os.path.join(testfiles_path, "test_grids", "wp_2.2_2.4")
pfd_files = [f for f in os.listdir(folder) if f.endswith("_gen.pfd")]

pf_dict_all = {}
for file in pfd_files:
    proj_name = os.path.splitext(file)[0]

    if file.startswith('1_'):
        fault_location = [0, 1, 2, 3]
        activate_gen = [1, [1, 3]]
    elif file.startswith('2_') or file.startswith('3_'):
        fault_location = [0, 1, 2, 3]
        activate_gen = [[3, 4]]
    elif file.startswith('4_'):
        fault_location = [0, 8, 18]
        if '_dyn_' in file:
            activate_gen = [4]
        elif '_yyn_' in file:
            activate_gen = [[4, 7, 14]]
        elif '_ynyn_' in file:
            activate_gen = [[4, 7, 14, 19]]

    for fl in fault_location:
        for act_gen in activate_gen:
            pf_dict = get_all_pf_sc_results(proj_name, fault_location=fl, activate_sgen=None, activate_gen=act_gen,
                                            save_to_excel=False)
    pf_dict_all[proj_name] = pf_dict

##

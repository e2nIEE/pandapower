from pandapower.converter.powerfactory.pf_export_functions import run_short_circuit
from pandapower.test.shortcircuit.SCE_Tests.sc_result_comparison.get_pf_shortcircuit_results import get_pf_sc_bus_results, get_pf_sc_branch_results
import sys
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP1\Python\3.11")
try:
    import powerfactory as pf
except:
    pass

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging


proj_name = 'test_case_1_four_bus_radial_grid'
app = pf.GetApplication()
app.ActivateProject(proj_name)
active_project = app.GetActiveProject()

# activate study case
study_case_folder = active_project.GetContents("Study Cases")[0]
study_cases = study_case_folder.GetContents()
study_case = study_cases[0]
study_case.Activate()

# ##
# fault_types = ['lll', 'll', 'llg', 'lg']
# cases = ['max', 'min']
# fault_impedances = [(0, 0), (5, 5)]
# lv_tol_percents = [6, 10]


##
df_bus = get_pf_sc_bus_results(app, fault_type='LLL', calc_mode='max', fault_impedance_xf=0, fault_impedance_rf=0,
                  lv_tol_percent=10, fault_location_index=4)

df_branch = get_pf_sc_branch_results(app, fault_type='LLL', calc_mode='max', fault_impedance_xf=0, fault_impedance_rf=0,
                  lv_tol_percent=10, fault_location_index=3)

##


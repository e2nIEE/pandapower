import sys
from pandapower.converter.powerfactory.export_pfd_to_pp import from_pfd
import pandas as pd
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP1\Python\3.11")
import powerfactory as pf
try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#logger = logging.getLogger(__name__)


app = pf.GetApplication()
app.ActivateProject("Short_Circuit_Test_Case_SCE")
active_project = app.GetActiveProject()

# activate study case
study_case_folder = active_project.GetContents("Study Cases")[0]
study_cases = study_case_folder.GetContents()
study_case = study_cases[0]
study_case.Activate()

## convert net from pf to pp and get short circuit results
net = from_pfd(app, prj_name="Short_Circuit_Test_Case_SCE", sc_type='ll', sc_mode='max',
               sc_impedance_r=0, sc_impedance_x=0)
pf_sc_results_bus = net.res_bus_sc
pf_sc_results_line = net.res_line_sc
pf_sc_results_bus.head()

## comparison
from pandapower.shortcircuit.calc_sc import calc_sc
calc_sc(net, fault="3ph", case='max', branch_results=True, ip=False)
net.res_bus_sc.head()

compare_results_bus = pd.concat([net.res_bus_sc, pf_sc_results_bus], axis=1)


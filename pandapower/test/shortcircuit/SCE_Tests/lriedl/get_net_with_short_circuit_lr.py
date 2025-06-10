import sys
from pandapower.converter.powerfactory.export_pfd_to_pp import from_pfd
import pandas as pd
from pandapower.converter.powerfactory.pf_export_functions import run_load_flow
from pandapower.converter.powerfactory.validate import validate_pf_conversion
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP1\Python\3.11")
import powerfactory as pf
try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#logger = logging.getLogger(__name__)
proj_name = "test_case_2_five_bus_radial_grid_dyn_gen"

app = pf.GetApplication()
app.ActivateProject(proj_name)
active_project = app.GetActiveProject()

# activate study case
study_case_folder = active_project.GetContents("Study Cases")[0]
study_cases = study_case_folder.GetContents()
study_case = study_cases[0]
study_case.Activate()

## run load flow in pf and convert net from pf to pp
net = from_pfd(app, prj_name=proj_name, sc_type='ll', sc_mode='max')
pf_sc_results = net.res_bus_sc

##
from pandapower.shortcircuit.calc_sc import calc_sc
calc_sc(net, fault="LL", case='max', branch_results=False, ip=True)

net.res_bus_sc = pd.concat([net.res_bus_sc, pf_sc_results], axis=1)

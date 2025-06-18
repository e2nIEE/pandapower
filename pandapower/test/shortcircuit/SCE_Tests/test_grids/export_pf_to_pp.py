import sys
from pandapower.converter.powerfactory.export_pfd_to_pp import from_pfd
from pandapower.converter.powerfactory.validate import validate_pf_conversion
from pandapower.file_io import to_json
import os

#sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP5A\Python\3.9")
sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP1\Python\3.11")

try:
    import powerfactory as pf
except:
    pass

app = pf.GetApplication()

##
base_dir = os.getcwd()
folder = os.path.join(base_dir)
pfd_files = [f for f in os.listdir(folder) if f.endswith(".pfd")]
net_dict = {}
all_diff_dict = {}

# for file in pfd_files:
prj_name = "test_case_2_five_bus_radial_grid_dyn"
net = from_pfd(app, prj_name=prj_name)
all_diffs = validate_pf_conversion(net, tolerance_mva=1e-9)
to_json(net, f"{prj_name}.json")
net_dict[prj_name] = net
all_diff_dict[prj_name] = all_diffs

##
##


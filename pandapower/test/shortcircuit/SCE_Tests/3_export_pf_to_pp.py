import sys
from pandapower.converter.powerfactory.export_pfd_to_pp import from_pfd
from pandapower.converter.powerfactory.validate import validate_pf_conversion
import powerfactory as pf
from pandapower.file_io import to_json

sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP5A\Python\3.11")

app = pf.GetApplication()
net = from_pfd(app, prj_name="Short_Circuit_Test_Case_SCE")
all_diffs = validate_pf_conversion(net, tolerance_mva=1e-9)
to_json(net, "4_bus_radial_grid.json")
import sys
from pandapower.converter.powerfactory.export_pfd_to_pp import from_pfd
from pandapower.converter.powerfactory.validate import validate_pf_conversion
import powerfactory as pf
from pandapower.file_io import to_json

sys.path.append(r"C:\Program Files\DIgSILENT\PowerFactory 2024 SP5A\Python\3.9")

app = pf.GetApplication()

"""net = from_pfd(app, prj_name="test_case_1_four_bus_radial_grid")
all_diffs_1 = validate_pf_conversion(net, tolerance_mva=1e-9)
to_json(net, "test_case_1_four_bus_radial_grid.json")"""

"""net = from_pfd(app, prj_name="test_case_2_five_bus_radial_grid")
all_diffs_2 = validate_pf_conversion(net, tolerance_mva=1e-9)
to_json(net, "test_case_2_five_bus_radial_grid.json")"""

net = from_pfd(app, prj_name="test_case_3_five_bus_meshed_grid")
all_diffs_2 = validate_pf_conversion(net, tolerance_mva=1e-9)
to_json(net, "test_case_3_five_bus_meshed_grid.json")
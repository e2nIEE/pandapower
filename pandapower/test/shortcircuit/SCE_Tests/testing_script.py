import pandapower as pp
import pandapower.shortcircuit as sc
from pandapower.shortcircuit.calc_sc import calc_sc
import numpy as np
from pandapower.create import create_empty_network, create_bus, create_ext_grid, create_line, create_sgen, \
    create_transformer_from_parameters, create_transformers_from_parameters, create_line_from_parameters, create_buses, \
    create_lines_from_parameters, create_switch, create_load, create_shunt, create_ward, create_xward
import pandas as pd
import numpy as np
from pandapower.shortcircuit.calc_sc import calc_sc
from pandapower.file_io import from_json
import pytest
import re
import copy
import os
from pandapower import pp_dir

# net_name = "test_case_1_four_bus_radial_grid.json"
# net_name = "test_case_2_five_bus_radial_grid_Yyn.json"
# net_name = "test_case_3_five_bus_meshed_grid_Dyn.json"
# net_name = "test_trafo_simple.json"
# net_name = "test_case_4_twenty_bus_radial_grid_YNyn.json"
net_name = r"wp_2.2\1_four_bus_radial_grid_sgen.json"

net = from_json(os.path.join(pp_dir, "test", "shortcircuit", "sce_tests", "test_grids", net_name))
net.sgen['k'] = 1.2
net.sgen['active_current'] = False
net.sgen.loc[net.sgen.bus == 1, 'in_service'] = True
net.sgen.loc[net.sgen.bus == 2, 'in_service'] = False
net.sgen.loc[net.sgen.bus == 3, 'in_service'] = False
net.line["c0_nf_per_km"] = 0
net.line["c_nf_per_km"] = 0
calc_sc(net, fault="LG", case="max", bus=1, return_all_currents=False, branch_results=True, ip=False, r_fault_ohm=0, x_fault_ohm=0, lv_tol_percent=6)
print(net.res_bus_sc)
print(net.res_line_sc)
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
# net_name = r"wp_2.2_2.4\1_four_bus_radial_grid_gen.json"
net_name = r"wp_2.2_2.4\4_twenty_bus_radial_grid_dyn_gen.json"

net = from_json(os.path.join(pp_dir, "test", "shortcircuit", "sce_tests", "test_grids", net_name))
net.sgen.in_service = False
net.gen.in_service = False
net.gen.loc[net.gen.bus == 1, 'in_service'] = False
net.gen.loc[net.gen.bus == 2, 'in_service'] = False
net.gen.loc[net.gen.bus == 3, 'in_service'] = False
net.gen.loc[net.gen.bus == 4, 'in_service'] = True

calc_sc(net, fault="LG", case="min", bus=0, return_all_currents=False, branch_results=True, ip=False, r_fault_ohm=5, x_fault_ohm=5, lv_tol_percent=10)
print(net.res_bus_sc)
print(net.res_line_sc)
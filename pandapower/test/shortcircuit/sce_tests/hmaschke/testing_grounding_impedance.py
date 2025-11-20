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

# net_name = r"wp_2.1\test_case_2_five_bus_radial_grid_YNyn.json"
# net_name = "test_case_3_five_bus_meshed_grid_Dyn.json"
# net_name = "test_trafo_simple.json"
# net_name = "test_case_4_twenty_bus_radial_grid_YNyn.json"
# net_name = r"wp_2.2_2.4\1_four_bus_radial_grid_gen.json"
# net_name = r"wp_2.2_2.4\4_twenty_bus_radial_grid_dyn_gen.json"
net_name = r"wp_2.2_2.4\2_five_bus_radial_grid_dyn_gen.json"

net = from_json(os.path.join(pp_dir, "test", "shortcircuit", "sce_tests", "test_grids", net_name))
net.sgen.in_service = False
net.gen.in_service = False
net.load.in_service = False

grounding_types = ["solid", "resistance", "inductance", "impedance", "isolated", "resonant"]
grounding_type = "isolated"
net.trafo['grounding_type'] = grounding_type

if grounding_type == "solid":
    net.trafo['xn_ohm'] = 0
    net.trafo['rn_ohm'] = 0
elif grounding_type == "resistance":
    net.trafo['xn_ohm'] = 0
    net.trafo['rn_ohm'] = 5
elif grounding_type == "inductance":
    net.trafo['xn_ohm'] = 5
    net.trafo['rn_ohm'] = 0
elif grounding_type == "impedance":
    net.trafo['xn_ohm'] = 5
    net.trafo['rn_ohm'] = 5
elif grounding_type == "isolated":
    net.trafo['xn_ohm'] = 1e8
    net.trafo['rn_ohm'] = 1e8
elif grounding_type == "resonant":
    # 20000 / np.sqrt(3) / 14.84958 = 777.598
    net.trafo['xn_ohm'] = 777
    net.trafo['rn_ohm'] = 0

pp.create.create_ward(net, 1, 0, 0, 0, 0, "grounding_element", True, rn_ohm=0, xn_ohm=0)

ward = net.ward
ward_buses = ward.bus.values
# how to calculate r and x in Ohm:
y_ward_pu = (ward["pz_mw"].values + ward["qz_mvar"].values * 1j)
z_ward_pu = 1/y_ward_pu
vn_net = net.bus.loc[ward_buses, "vn_kv"].values
z_base_ohm = (vn_net ** 2)# / base_sn_mva)
z_ward_ohm = z_ward_pu * z_base_ohm

# only LG and LLG
calc_sc(net, fault="LG", case="max", bus=1, return_all_currents=False, branch_results=True, ip=False, r_fault_ohm=0, x_fault_ohm=0, lv_tol_percent=10)
print(net.res_bus_sc)
print(net.res_line_sc)
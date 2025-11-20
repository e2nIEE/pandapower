import pandapower as pp
import pandapower.shortcircuit as sc
from pandapower.shortcircuit.calc_sc import calc_sc
import numpy as np
from pandapower.create import create_empty_network, create_bus, create_ext_grid, create_line, create_sgen, \
    create_transformer_from_parameters, create_transformers_from_parameters, create_line_from_parameters, create_buses, \
    create_lines_from_parameters, create_switch, create_load, create_shunt, create_ward, create_xward
import pandas as pd
import numpy as np
import pandapower.plotting as plot
import matplotlib.pyplot as plt
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
net_name = r"wp_2.2_2.4\1_four_bus_radial_grid_gen.json"

net = from_json(os.path.join(pp_dir, "test", "shortcircuit", "sce_tests", "test_grids", net_name))
net.sgen['k'] = 1.2
net.sgen['active_current'] = False
net.gen['current_source'] = False
net.sgen.in_service = False
net.gen.in_service = False
net.line["c0_nf_per_km"] = 0
net.line["c_nf_per_km"] = 0
net.gen.loc[net.gen.bus == 1, 'in_service'] = True
calc_sc(net, fault="LLL", case="max", bus=1, return_all_currents=False, branch_results=True, ip=False, r_fault_ohm=0, x_fault_ohm=0, lv_tol_percent=6)
#print(net.res_bus_sc)
#print(net.res_line_sc)
print('Hello,this is table of results for %s fault'%net["_options"]["fault"])
print(net.res_gen_sc)



#plotting-it does not show lines for some reason-COULD NOT FIND REASON WHY-ASK HENDRIK
#WITHOUT GENERATING GEODATA, PLOT ONLY IS BLANK
"""plot.create_generic_coordinates(net, respect_switches=False, overwrite=True)
#plot.create_line_geodata(net, lines=net.line.index, overwrite=True)
sizes = plot.get_collection_sizes(net)
lc = plot.create_line_collection(net, net.line.index, linewidths=2, zorder=1, color="grey")
bc = plot.create_bus_collection(net, net.bus.index, size=sizes['bus'], zorder=2, color="green")
#plot.simple_plot(net)
plot.draw_collections([lc, bc])
plt.show()
print(net)
print(net.line)
print("Line collection:", lc)
print("Bus collection:", bc)
print(net.bus)
print(net.line[['name', 'from_bus', 'to_bus']])"""



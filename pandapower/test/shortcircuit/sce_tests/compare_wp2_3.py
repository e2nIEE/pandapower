import pandapower as pp
from pandapower.shortcircuit.calc_sc import calc_sc
import pandas as pd
import os
import numpy as np
from pandapower.test.shortcircuit.sce_tests.functions_tests import load_pf_results

##
base_dir = os.getcwd()
folder = os.path.join(base_dir, 'pandapower', 'test', 'shortcircuit', 'sce_tests', 'test_grids', 'wp_2.3')
json_files = [f for f in os.listdir(folder) if f.endswith(".json")]
print(json_files)
##
file = json_files[1]
print(file)
net = pp.from_json(os.path.join(folder, file))

#net = pp.from_json(r"C:\Users\lriedl\PycharmProjects\sce\pandapower\test\shortcircuit\sce_tests\test_grids\wp_2.3\4_five_bus_radial_grid_1ph_yyn_MP.json")

# check if parameters are set correctly to zero
print(net.line.c_nf_per_km)
print(net.trafo.i0_percent)

## change zero parameters to very small values
x = 1e-20

net.line.loc[net.line.name.str.contains('1ph'), "c0_nf_per_km"] = x
net.line.loc[net.line.name.str.contains('1ph'), "r0_ohm_per_km"] = x
net.line.loc[net.line.name.str.contains('1ph'), "x0_ohm_per_km"] = x

net.line.loc[net.line.name.str.contains('2ph'), "c0_nf_per_km"] = x
net.line.loc[net.line.name.str.contains('2ph'), "r0_ohm_per_km"] = x
net.line.loc[net.line.name.str.contains('2ph'), "x0_ohm_per_km"] = x


## short circuit calculation
fault = 'LL'
fault_bus = 3
case = 'max'
lv_tol_percent = 10
branch = True
r_fault_ohm = 0
x_fault_ohm = 0

calc_sc(net, fault=fault, case=case, branch_results=branch, ip=False,
        r_fault_ohm=r_fault_ohm, x_fault_ohm=x_fault_ohm, bus=fault_bus, return_all_currents=False)

# show pp results
print(net.res_bus_sc.loc[fault_bus])

# show pf results
folder_results = os.path.join(base_dir, 'pandapower', 'test', 'shortcircuit',
                              'sce_tests', 'sc_result_comparison', 'wp_2.3')
result_table = load_pf_results(os.path.join(folder_results, f'{file[:-5]}_pf_sc_results_{fault_bus}_bus.xlsx'))
result_pf = result_table[f'{fault}_{case}_{lv_tol_percent}']
print(result_pf.loc[fault_bus])

##

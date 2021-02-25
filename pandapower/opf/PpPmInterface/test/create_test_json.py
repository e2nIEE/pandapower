# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 13:35:35 2021

@author: x230
"""
import os
import pathlib
import simbench as sb
import pandapower as pp
# from pandapower import pp_dir
from pandapower.converter.powermodels.to_pm import convert_pp_to_pm

grid_code = "1-HV-urban--0-sw"
net = sb.get_simbench_net(grid_code)
pp.runpp(net)

# pkg_dir = pathlib.Path(pp_dir, "pandapower", "opf", "PpPmInterface") #general direction
pkg_dir = pathlib.Path(pathlib.Path.home(), "GitHub", "pandapower", "pandapower", "opf", "PpPmInterface")
json_path = os.path.join(pkg_dir, "test" , "test_ipopt.json")

test_net = convert_pp_to_pm(net, pm_file_path=json_path, correct_pm_network_data=True, calculate_voltage_angles=True, ac=True,
                     trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
                     pp_to_pm_callback=None, pm_model="DCPowerModel", pm_solver="ipopt",
                     pm_mip_solver="cbc", pm_nl_solver="ipopt")

# -*- coding: utf-8 -*-
"""
voltage deviation minimization

Created on Fri Oct 15 2021

@author: Zheng, Maryam
"""
import pandapower as pp
import pandapower.networks as nw
# get net
net = nw.create_cigre_network_mv(with_der="pv_wind")
# pp.runpp(net)
# add scale 
net.sgen.p_mw = net.sgen.p_mw * 8 #TODO: why dont you scale up q?
net.sgen.sn_mva = net.sgen.sn_mva * 8 #TODO: why scale sn_mva?
# set controllable sgen
net.load['controllable'] = False
net.sgen['controllable'] = True
#set limits:
net.sgen["max_p_mw"] = net.sgen.p_mw.values
net.sgen["min_p_mw"] = net.sgen.p_mw.values #TODO: why min and max are same?
net.sgen["max_q_mvar"] = net.sgen.p_mw.values * 0.328 #TODO: why * 0.328?
net.sgen["min_q_mvar"] = -net.sgen.p_mw.values * 0.328  
 
net.bus["max_vm_pu"] = 1.1
net.bus["min_vm_pu"] = 0.9

net.ext_grid["max_q_mvar"] = 10000.0
net.ext_grid["min_q_mvar"]= -10000.0
net.ext_grid["max_p_mw"] = 10000.0
net.ext_grid["min_p_mw"] = -10000.0

net.gen["max_p_mw"]= net.gen.p_mw.values       
net.gen["min_p_mw"] = net.gen.p_mw.values        
net.gen["max_q_mvar"]= 10000.0        
net.gen["min_q_mvar"] = -10000.0

net.trafo["max_loading_percent"] = 500.0
net.line["max_loading_percent"] = 500.0

# add new column to bus for voltage threshold:
net.bus["pm_param/threshold_v"] = None
# for buses with controllable sgen set threshold:
net.bus["pm_param/threshold_v"].loc[net.sgen.bus] = 0.99

# net = get_network("cigre_mv_ts")
# net.bus["pm_param/threshold_v"].loc[net.sgen.bus] = 0.99
# net.bus["pm_param/threshold_v"] = None

for idx in net.sgen.index:
    pp.create_poly_cost(net, idx, "sgen", 1.0)

for idx in net.gen.index:
    pp.create_poly_cost(net, idx, "gen", 1.0)

for idx in net.ext_grid.index:
    pp.create_poly_cost(net, idx, "ext_grid", 1.0)

# TODO: pm_logger
# TODO: specific results?

pp.runpm_vd(net, calculate_voltage_angles=True,
                      trafo_model="t", delta=1e-8, trafo3w_losses="hv", check_connectivity=True,
                      n_timesteps=96, time_elapsed=0.25, correct_pm_network_data=True,
                      pm_model="ACPPowerModel", pm_time_limits=None, pm_log_level=0,
                      delete_buffer_file=False, pm_file_path=None,
                      pm_tol=1e-8, pdm_dev_mode=True)


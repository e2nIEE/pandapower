# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:39:53 2019

@author: uk067483
"""

import os
import numpy as np
import pandas as pd
import tempfile
import pandapower as pp
from pandapower.control import ConstControl
from pandapower.timeseries import DFData
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.networks import ieee_european_lv_asymmetric

parent = os.path.dirname(os.path.realpath(__file__))
path = parent + "\\European_LV_CSV\\"
load_path = path + "\\Load Profiles\\"
def remove_comments(f):
    '''Pass comments'''
    start=f.seek(0)
    for index in range(5):
        start=f.tell()
        if f.readline().startswith('#'):
            continue
        else:
            break      
    f.seek(start)
    return f

with open (path+"Loads.csv",'r') as f:
    f = remove_comments(f)
    loads = pd.read_csv(f)
    f.close()
    
with open (path+"LoadShapes.csv",'r') as f:
    f = remove_comments(f)
    loadshapes = pd.read_csv(f)
    f.close()
        
def timeseries_example(output_dir):
    # 1. create test net
    net = ieee_european_lv_asymmetric()
    # 2. create data source for loads
    profiles, ds = create_data_source()
    # 3. create controllers (to control P values of the load and the sgen)
    net = create_controllers(net, ds)

    # time steps to be calculated. Could also be a list with non-consecutive time steps
    time_steps = range(0, loadshapes.npts[0])

    # 4. the output writer with the desired results to be stored to files.
    ow = create_output_writer(net, time_steps, output_dir)

    # 5. the main time series function
    run_timeseries(net, time_steps, output_writer=ow, run=pp.runpp_3ph,continue_on_divergence=True)

def create_data_source():
    profiles = pd.DataFrame()
    for loadprofile,file in (loadshapes[['Name','File']].values):
        profiles[loadprofile] = pd.read_csv(load_path+file).mult.values*1e-3 
        profiles[loadprofile+'cos_phi'] = float(loads[loads.Yearly==loadprofile].PF.values)
    ds = DFData(profiles)
    return profiles, ds

def create_controllers(net, ds):
    ConstControl(net, element='asymmetric_load', variable='p_a_mw', element_index=loads[loads['phases']=='A'].index,
                 data_source=ds, profile_name=loads[loads['phases']=='A'].Yearly)
    ConstControl(net, element='asymmetric_load', variable='p_b_mw', element_index=loads[loads['phases']=='B'].index,
                 data_source=ds, profile_name=loads[loads['phases']=='B'].Yearly)
    ConstControl(net, element='asymmetric_load', variable='p_c_mw', element_index=loads[loads['phases']=='C'].index,
                 data_source=ds, profile_name=loads[loads['phases']=='C'].Yearly)
    ConstControl(net, element='asymmetric_load', variable='cos_phi', element_index=loads.index,
                 data_source=ds, profile_name=loads.Yearly+'cos_phi', set_q_from_cosphi_3ph=True)    
    return net

def create_output_writer(net, time_steps, output_dir):
    ow = OutputWriter(net, time_steps, output_path=output_dir, output_file_type=".xls")
    ow.log_variable('res_bus_3ph', 'p_a_mw', index=net.bus.index, eval_function=np.sum, eval_name="bus_sum_pa")
    ow.log_variable('res_bus_3ph', 'p_b_mw', index=net.bus.index, eval_function=np.sum, eval_name="bus_sum_pb")
    ow.log_variable('res_bus_3ph', 'p_c_mw', index=net.bus.index, eval_function=np.sum, eval_name="bus_sum_pc")
    ow.log_variable('res_bus_3ph', 'vm_a_pu', index=net.bus.index, eval_function=np.max, eval_name="bus_max_va")
    ow.log_variable('res_bus_3ph', 'vm_b_pu', index=net.bus.index, eval_function=np.max, eval_name="bus_max_vb")
    ow.log_variable('res_bus_3ph', 'vm_c_pu', index=net.bus.index, eval_function=np.max, eval_name="bus_max_vc")
    return ow

output_dir = os.path.join(tempfile.gettempdir(), "time_series_example")
print("Results can be found in your local temp folder: {}".format(output_dir))
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
net=timeseries_example(output_dir)

import matplotlib.pyplot as plt
 

# voltage results
vm_pu_file = os.path.join(output_dir, "res_bus_3ph", "vm_a_pu.xls")
vm_pu = pd.read_excel(vm_pu_file)
vm_pu.plot(label="vm_a_pu")
plt.xlabel("time step")
plt.ylabel("Phase A voltage mag. [p.u.]")
plt.title("Voltage Magnitude Phase A")
plt.grid()
plt.show()


# p_mw results
p_a_mw_file = os.path.join(output_dir, "res_bus_3ph", "p_a_mw.xls")
p_b_mw_file = os.path.join(output_dir, "res_bus_3ph", "p_b_mw.xls")
p_c_mw_file = os.path.join(output_dir, "res_bus_3ph", "p_c_mw.xls")

p_a_mw = pd.read_excel(p_a_mw_file)
p_a_mw.plot(label="p_a_mw")
plt.xlabel("time step")
plt.ylabel("P [MW] Phase A")
plt.title("Real Power at Buses")
plt.grid()
plt.show()

p_b_mw = pd.read_excel(p_b_mw_file)
p_b_mw.plot(label="p_b_mw")
plt.xlabel("time step")
plt.ylabel("P [MW] Phase A")
plt.title("Real Power at Buses")
plt.grid()
plt.show()

p_c_mw = pd.read_excel(p_c_mw_file)
p_c_mw.plot(label="p_c_mw")
plt.xlabel("time step")
plt.ylabel("P [MW] Phase A")
plt.title("Real Power at Buses")
plt.grid()
plt.show()
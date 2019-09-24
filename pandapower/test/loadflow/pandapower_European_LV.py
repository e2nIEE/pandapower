# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:05:24 2019

@author: uk067483
"""
import pandas as pd
import pandapower as pp
import numpy as np
from pandapower.plotting import simple_plot
from pandapower import pp_dir
import os
#  Path of this script
parent = os.path.join(pp_dir,'test','test_files')
#  Path of the European LV network csv data files
path = parent + "\\European_LV_CSV\\"
#  Path of Load Profile data
load_path = path + "\\Load Profiles\\"
#  Path of the OpenDSS Solution data files
dss_snapPath = parent + '\\Solutions\\OpenDSS\\Snapshots\\'
#  Path of the GridLab-D Solution data files
gld_snapPath = parent + '\\Solutions\\GridLAB-D\\Snapshots'

# =============================================================================
# Remove comments and unnecessary text from csv data
# =============================================================================
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
# Buses
with open(path+"Buscoords.csv", "r") as f:
    f = remove_comments(f)
    buses = pd.read_csv(f)
    f.close()    
# Line Types
with open (path+"LineCodes.csv",'r') as f:
    '''Pass comments'''
    f = remove_comments(f)
    line_typ = pd.read_csv(f)
    f.close()
# Lines
with open (path+"Lines.csv",'r') as f:
    '''Pass comments'''
    f = remove_comments(f)
    lines = pd.read_csv(f)
    f.close()
# Loads
with open (path+"Loads.csv",'r') as f:
    '''Pass comments'''
    f = remove_comments(f)
    loads = pd.read_csv(f)
    f.close()
# Load Shape
with open (path+"LoadShapes.csv",'r') as f:
    '''Pass comments'''
    f = remove_comments(f)
    loadshapes = pd.read_csv(f)
    f.close()
# Time series data of Load Shape
loadprofiles = dict()
for loadprofile,file in (loadshapes[['Name','File']].values):
        loadprofiles[loadprofile] = pd.read_csv(load_path+file)

# =============================================================================
#  Creating three snapshot networks from load data and comparing with solutions
# =============================================================================
for snap in [0,565,1439]:
    net = pp.create_empty_network()
    source_bus= pp.create_bus(net,vn_kv=11,name='SOURCEBUS')
    s_sc_max_mva = np.sqrt(3)*11*3
    pp.create_ext_grid(net,source_bus,vm_pu=1.05,name='Source',
                       s_sc_max_mva=s_sc_max_mva,rx_max =0.1,
                       r0x0_max=0.1,x0x_max=1.0)
    #pandpower uses lv side voltage as base voltage of the network
    x_percent  =  (4./100.)*((11.*11.)/0.8) / ((0.416 ** 2)/0.8) #base changed from hv to lv side
    r_percent =  (0.4/100)* ( (11.*11.)/0.8) / ((0.416 ** 2)/0.8)# base changed from hv to lv side
    z_percent = np.sqrt(r_percent**2+x_percent**2)
    pp.create_std_type(net, {"sn_mva": 0.8,
                "vn_hv_kv": 11,
                "vn_lv_kv": 0.416,
                "vk_percent": z_percent ,
                "vkr_percent": r_percent,
                "pfe_kw":1.7,
                "i0_percent": 0.21251,
#                "pfe_kw":0.,
#                "i0_percent": 0.,                
                "shift_degree": 30,
                "vector_group": 'Dyn',
                "tap_side": "lv",
                "tap_neutral": 0,
                "tap_min": -2,
                "tap_max": 2,
                "tap_step_degree": 0,
                "tap_step_percent": 2.5,
                "tap_phase_shifter": False,
                "vk0_percent": z_percent, 
                "vkr0_percent": r_percent, 
                "mag0_percent": 100,
                "mag0_rx": 0.,
                "si0_hv_partial": 0.9,}, name='0.8 MVA 10/0.4 kV Dyn1 ASEA', element="trafo")
    pp.create_buses(net,len(buses),vn_kv=0.416,name=buses['Busname'].values.astype(str),
                    geodata=list(zip(buses[' x'], buses[' y'])))
    
    pp.create_transformer(net,hv_bus= source_bus, 
                          lv_bus= np.where(net.bus.name=='1')[0][0] , 
                          std_type= '0.8 MVA 10/0.4 kV Dyn1 ASEA')
    
    for index,typ in line_typ.iterrows():        
        pp.create_std_type(net,{"r_ohm_per_km": float(typ['R1']), 
                                    "x_ohm_per_km": float(typ['X1']),
                                    "c_nf_per_km": float(typ['C1']), 
                                    "max_i_ka": 0.421,"endtemp_degree": 70.0, 
                                    "r0_ohm_per_km": float(typ['R0']),
                                    "x0_ohm_per_km": float(typ['X0']),
                                    "c0_nf_per_km":  float(typ['C0']),
                                    "type": 'ol'
                                    }, name=typ['Name'],element = "line")
    for index,line in lines.iterrows():
        pp.create_line(net,from_bus=np.where(net.bus.name == str(line['Bus1']))[0], 
                       to_bus= np.where(net.bus.name == str(line['Bus2']))[0], 
                       length_km= float(line['Length'])*1e-3,
                       std_type= line['LineCode'])
    for index,load in loads.iterrows():
        if load['phases']=='A':
            load_shape=load['Yearly']
            P=float(loadprofiles[load_shape]['mult'][snap])*1e-3 # kW chages to MW
            pf=float(load['PF'])
            Q= np.sqrt((P/pf)**2-P**2)*1e-3
            pp.create_asymmetric_load(net,np.where(net.bus.name == str(load['Bus']))[0][0],p_a_mw = P, q_a_mvar=Q )
        elif load['phases']=='B':
            load_shape=load['Yearly']
            P=float(loadprofiles[load_shape]['mult'][snap])*1e-3
            pf=float(load['PF'])
            Q= np.sqrt((P/pf)**2-P**2)*1e-3
            pp.create_asymmetric_load(net,np.where(net.bus.name == str(load['Bus']))[0][0],p_b_mw = P, q_b_mvar=Q )
        elif load['phases']=='C':
            load_shape=load['Yearly']
            P=float(loadprofiles[load_shape]['mult'][snap])*1e-3
            pf=float(load['PF'])
            Q= np.sqrt((P/pf)**2-P**2)*1e-3
            pp.create_asymmetric_load(net,np.where(net.bus.name == str(load['Bus']))[0][0],p_c_mw = P, q_c_mvar=Q )        
    pp.add_zero_impedance_parameters(net)
    pp.runpp_3ph(net,calculate_voltage_angles = True)
    nw_dir = os.path.join(pp_dir,'networks')
    if(snap == 0):
        pp.to_json(net, filename=nw_dir + '\\from_csv_IEEE_European_LV_Off_Peak_start.json')
    elif(snap == 565):
        pp.to_json(net, filename=nw_dir + '\\from_csv_IEEE_European_LV_On_Peak_mid.json')
    elif(snap == 1439):     
        pp.to_json(net, filename=nw_dir + '\\from_csv_IEEE_European_LV_Off_Peak_end.json')
#app = powerf.GetApplication()
#pf_project_name = "IEEE_LV_Grid"
#pf_net_name = "IEEE_European_LV"
#conv = to_powerfactory(app, net, pf_project_name, pf_net_name)
## print(conv)
#ldf = app.GetFromStudyCase("ComLdf")
#
## ldf.iopt_noinit = 1 no initialization (->flat start)
## app.PrintPlain(ldf.iopt_noinit)
#ldf.Execute()  # loadflow @ pf
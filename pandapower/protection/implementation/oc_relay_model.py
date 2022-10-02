# -*- coding: utf-8 -*-

# This function implemenets the protecton module using over current relay
# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pandapower.shortcircuit as sc
import copy
import pandapower as pp
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore') 
import sys
from pandapower.protection.implementation.utility_functions import *
from pandapower.plotting import simple_plot

# set the parameters
def oc_parameters(net2, switch_id,t_ig,t_igg, sc_fraction=None, overload_factor=None,
                  ct_current_factor=None,safety_factor=None, oc_pickup_current_manual=None): 
    
    net = copy.deepcopy(net2)

    # get the line index and bus index from the line
    line_idx = get_line_idx(net, switch_id)
    
    bus_idx = get_bus_idx(net, switch_id)

    
    # creat short circuit on the given line and location
    if oc_pickup_current_manual is None:
        
        net_sc = create_sc_bus(net, line_idx, sc_fraction)
        sc.calc_sc(net_sc, bus=max(net_sc.bus.index), branch_results=True)

    
    #net.ext_grid['s_sc_max_mva'] = 1000
    # I_g and I_gg current treshold calculation
        I_g = net_sc.line.max_i_ka.at[line_idx] * overload_factor * ct_current_factor
        I_gg = net_sc.res_line_sc.ikss_ka.at[line_idx] * safety_factor
    
    else:
        I_g= oc_pickup_current_manual[2] # take manual inputs
        I_gg=oc_pickup_current_manual[1]
    # dictionary to store the relay settings for each lines

    settings = {"net": net, "switch_id": switch_id, "line_idx": line_idx, "bus_idx": bus_idx,
                "Ig_ka": I_g, "Igg_ka": I_gg, "tg": t_ig, "tgg": t_igg}
    
    return settings

    # get the short circuit current values in the lines
def oc_get_measurement_at_relay_location(net, settings):
    switch_id = settings.get("switch_id")
    line_idx = get_line_idx(net, switch_id)
    i_ka = net.res_line_sc.ikss_ka.at[line_idx]

    return i_ka

    # based on the short circuit currents, the trip decision is taken
def oc_get_trip_decision(net, settings, i_ka):
    switch_id = settings.get("switch_id")
    max_ig = settings.get("Ig_ka")
    max_igg = settings.get("Igg_ka")
    t_ig = settings.get("tg")
    t_igg = settings.get("tgg")


    if i_ka > max_igg:
        trip = True
        trip_type = "instantaneous"
        trip_time = t_igg
        
    elif i_ka > max_ig:
        trip = True
        trip_type = "backup"
        trip_time = t_ig 

    else:
        trip = False
        trip_type = "no trip"
        trip_time = np.inf
     
    trip_decision = {"Switch ID": switch_id,'Switch type':'OC', "Trip": trip, "Fault Current [kA]": i_ka, "Trip Type": trip_type,
                     "Trip time [s]": trip_time, "Ig": max_ig, "Igg": max_igg, 'tg':t_ig, 't_gg':t_igg}
    
    return trip_decision


    # Time graded protection (backup) for definte over current relay
def time_graded_overcurrent(net,tripping_time_auto, tripping_time_manual):
      
    
    if tripping_time_auto:
    
        # get end buses from meshed network and radial network if any
        pf_loop_end_buses, pf_radial_end_buses = power_flow_end_points(net)
        
        switches = net.switch.loc[net.switch.et == "l"].index
        protection_settings = []
        pp.runpp(net)
        
        end_bus = net.ext_grid.bus
        if len(end_bus) > 1:
            assert ("Max number of ext grid should be 1")
        
        # Get the bus path from the given start and end buses
        bus_path = bus_path_from_to_bus(net,radial_start_bus=pf_radial_end_buses,loop_start_bus=pf_loop_end_buses, end_bus=end_bus[0])
       
        pathes = []
        for path in bus_path:
            listintersection = set(path) & set(pf_loop_end_buses)
            if len(listintersection) <= 1:
                pathes.append(path)
        pathes = [x for x in pathes if len(x) !=1]
        
            
        time_lines={}
        time_switches={}
        tg_sw_setting=[]
        
        for bus_path in pathes:
            line_path=get_line_path(net, bus_path)
        
            count=0
            # switching time based on the radial connections 
            for line in line_path:
                time_lines[line]=count*tripping_time_auto[2]+tripping_time_auto[1]
                count+=1
                
        for switch in net.switch[net.switch.closed == True].index:

            line_id=net.switch[net.switch.closed == True].element.at[switch]
            tg = time_lines[line_id] 
            time_switches[switch]=tg
            tg_sw_setting.append([switch,tg])
    
        # if there is multiple time setting for each switch take only the highest one
            df_protection_settings = pd.DataFrame(tg_sw_setting)
            df_protection_settings = df_protection_settings.sort_values(by=1, ascending = False)
            df_protection_settings = df_protection_settings.groupby(0).head(1)
            df_protection_settings["t_gg"] = [tripping_time_auto[0]] * len(df_protection_settings) 
            df_protection_settings.columns=["switch_id","t_g","t_gg"]
            df_protection_settings= df_protection_settings.sort_values(by=['switch_id'])
            df_protection_settings=df_protection_settings.reset_index(drop=True)

    if   tripping_time_manual is not None:
         df_protection_settings=pd.DataFrame()
         df_protection_settings['switch_id']=tripping_time_manual['switch_id']
         df_protection_settings['t_g']=tripping_time_manual['t_g']
         df_protection_settings['t_gg']=tripping_time_manual['t_gg']
    
    return df_protection_settings


def run_fault_scenario_oc(net, sc_line_id, sc_location,tripping_time_auto=None,tripping_time_manual=None,
                          sc_fraction=0.95, overload_factor=1.2, ct_current_factor=1.25,
                          safety_factor=1, pickup_current_manual=None):

    """
    The main function to create fault scenario in network at defined location to get the tripping decisons
    
       INPUT:
           **net** (pandapowerNet) - Pandapower network with switch type as "CB_non_dir" in net.switch.type
           
           **sc_line_id** (int, index)- Index of the line to create the short circuit
           
           **sc_location** (float)- Location of short circuit on the on line (between 0 and 1). 
           
           **tripping_time_auto** (list, float) - Relay tripping time calculated based on topological grid search.
          
            - tripping_time_auto =[t_>>, t_>, t_diff]
            - t_>> (t_gg): instantaneous tripping time in seconds,
            - t_> (t_g):  primary backup tripping time in seconds, 
            - t_diff: time grading delay difference in seconds
            
           "or" 
                
           **tripping_time_manual**- (dataframe, float) - User defined relay trip currents given as a dataframe with columns as 'switch_id', 't_gg', 't_g'
           
           (Note: either tripping_time_auto or the tripping_time_manual needed to be provided and not both)
           
           **sc_fraction** (float, 0.95) - Maximum possible extent to which the short circuit can be created on the line
                                 
           **overload_factor** - (float, 1.25)- Allowable over loading on the line used to calculate the backup pick up current
                                 
           **ct_current_factor** -(float, 1.2) - Current mutiplication factor to define the backup pick up current
           
           **safety_factor** -(float, 1) - Safety limit for the instantaneous pick up current
                    
            
        OPTIONAL:
            **pickup_current_manual** - (DataFrame, None) - User defined relay trip currents given as a dataframe with columns as 'switch_id', 'I_gg', 'I_g'

        
        OUTPUT:
            **return** (list(Dict),net_sc) - Return trip decision of each relays and short circuit net
        """

    
    df_protection_settings =time_graded_overcurrent(net,tripping_time_auto,tripping_time_manual)
    oc_relay_settings = []
    for switch_id in net.switch.index:
        
        if (net.switch.closed.at[switch_id]) & (net.switch.type.at[switch_id] == "CB_non_dir") & (
                net.switch.et.at[switch_id] == "l"):

            t_g=df_protection_settings.loc[df_protection_settings['switch_id']==switch_id]['t_g'].item()
            
            t_gg=df_protection_settings.loc[df_protection_settings['switch_id']==switch_id]['t_gg'].item()
            
            if pickup_current_manual is not None:
                
                oc_pickup_current_manual=list(pickup_current_manual.loc[switch_id])
                settings = oc_parameters(net, switch_id,t_g,t_gg, sc_fraction, overload_factor,ct_current_factor, safety_factor, oc_pickup_current_manual)

            else:
                settings = oc_parameters(net, switch_id,t_g,t_gg, sc_fraction, overload_factor,ct_current_factor, safety_factor)

            oc_relay_settings.append(settings)
            
        
    net_sc = create_sc_bus(net, sc_line_id, sc_location)
    sc.calc_sc(net_sc, bus=max(net_sc.bus.index), branch_results=True)

    trip_decisions = []
    
    for setting_idx in range(len(oc_relay_settings)):
        
        i_ka = oc_get_measurement_at_relay_location(net_sc, oc_relay_settings[setting_idx])
        
        trip_decision = oc_get_trip_decision(net_sc, oc_relay_settings[setting_idx], i_ka)

        trip_decisions.append(trip_decision)
    
    df_trip_decison = pd.DataFrame.from_dict(trip_decisions)
    df_decisions=df_trip_decison[["Switch ID","Switch type","Trip","Fault Current [kA]","Trip time [s]"]]
    
    print(df_decisions)
    
    return trip_decisions, net_sc  

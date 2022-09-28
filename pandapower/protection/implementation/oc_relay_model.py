# -*- coding: utf-8 -*-

# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pandapower.shortcircuit as sc
import copy
import pandapower as pp
import numpy as np
import pandas as pd
from pandapower.protection.implementation.utility_functions import *
from pandapower.plotting import simple_plot

# set the parameters
def oc_parameters(net2, switch_idx,t_ig,t_igg, sc_fraction=None, overload_factor=None,
                  ct_current_factor=None,safety_factor=None, oc_relay_trip_currents=None):  # all the factors can be re-considered
    
    net = copy.deepcopy(net2)

    
    # get the line index and bus index from the line
    
    line_idx = get_line_idx(net, switch_idx)
    
    bus_idx = get_bus_idx(net, switch_idx)

    
    # creat short circuit on the given line and location
    if oc_relay_trip_currents is None:
        
        net_sc = create_sc_bus(net, line_idx, sc_fraction)
        sc.calc_sc(net_sc, bus=max(net_sc.bus.index), branch_results=True)

    
    #net.ext_grid['s_sc_max_mva'] = 1000
    # I_g and I_gg current treshold calculation
        I_g = net_sc.line.max_i_ka.at[line_idx] * overload_factor * ct_current_factor
        I_gg = net_sc.res_line_sc.ikss_ka.at[line_idx] * safety_factor
    
    else:
        I_g= oc_relay_trip_currents[2] # take manual inputs
        I_gg=oc_relay_trip_currents[1]
    # dictionary to store the relay settings for each lines

    settings = {"net": net, "switch_idx": switch_idx, "line_idx": line_idx, "bus_idx": bus_idx,
                "Ig_ka": I_g, "Igg_ka": I_gg, "tg": t_ig, "tgg": t_igg}
    
    return settings

    
    # get the short circuit current values in the lines
def oc_get_measurement_at_relay_location(net, settings):
    switch_idx = settings.get("switch_idx")
    line_idx = get_line_idx(net, switch_idx)
    i_ka = net.res_line_sc.ikss_ka.at[line_idx]

    return i_ka

    # based on the short circuit currents, the trip decision is taken
def oc_get_trip_decision(net, settings, i_ka):
    switch_idx = settings.get("switch_idx")
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

    trip_decision = {"Switch": switch_idx,'Switch type':'OC', "Trip": trip, "Fault Current": i_ka, "Trip Type": trip_type,
                     "Trip time": trip_time, "Ig": max_ig, "Igg": max_igg, 'tg':t_ig, 't_gg':t_igg}
    # show only relevant tripping logic
    
    return trip_decision


    # Time graded protection (backup) for definte over current relay
def time_graded_overcurrent(net,timegrade, relay_trip_times):
      
    #timegrade=[0.07,0.5,0.3]
    
    if timegrade:
    
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
                time_lines[line]=count*timegrade[2]+timegrade[1]
                count+=1
                
        for switch in net.switch[net.switch.closed == True].index:
            
            #line_id=net.switch.element[switch]
            line_id=net.switch[net.switch.closed == True].element.at[switch]

            
            tg = time_lines[line_id] 
            
            time_switches[switch]=tg
            
            tg_sw_setting.append([switch,tg])
    
        # if there is multiple time setting for each switch take only the highest one
            df_protection_settings = pd.DataFrame(tg_sw_setting)
            df_protection_settings = df_protection_settings.sort_values(by=1, ascending = False)
            df_protection_settings = df_protection_settings.groupby(0).head(1)
            df_protection_settings["t_gg"] = [timegrade[0]] * len(df_protection_settings) 
            df_protection_settings.columns=["switch_idx","t_g","t_gg"]
            df_protection_settings= df_protection_settings.sort_values(by=['switch_idx'])
            df_protection_settings=df_protection_settings.reset_index(drop=True)

    if   relay_trip_times is not None:
         df_protection_settings=pd.DataFrame()
         df_protection_settings['switch_idx']=relay_trip_times['switch_idx']
         df_protection_settings['t_g']=relay_trip_times['t_g']
         df_protection_settings['t_gg']=relay_trip_times['t_gg']
    
    return df_protection_settings


def run_fault_scenario_oc(net, sc_line_idx, sc_location,timegrade=None,relay_trip_times=None,
                          sc_fraction=0.95, overload_factor=1.2, ct_current_factor=1.25,
                          safety_factor=1, relay_trip_currents=None,plot_grid=True,
                          plot_annotations=True, i_t_plot=True):

    """
    
    The main function to create fault scenario in network at defined location to get the tripping decisons
    
       INPUT:
           **net** (pandapowerNet) - Pandapower network with switch type as "CB_non_dir" in net.switch.type
           
           **sc_line_idx** (int, index)- Index of the line to create the short circuit.
           
           **sc_location** (float)- Location of short circuit on the on line (between 0 and 1).
           
           **timegrade** (list, float) - Relay tripping time calculated based on topological grid search.
          
            - timegrade =[t_gg, t_g and t_delta]
            - t_gg: instantaneous tripping time in seconds,
            - t_g:  primary backup tripping time in seconds, 
            - t_delta: secondary backup tripping time in seconds
            
           "or" 
                
           **relay_trip_times**- (dataframe, float) - User defined relay trip currents given as a dataframe with columns as 'switch_idx', 't_gg', 't_g'
           
           (Note: either timegrade or the relay_trip_times needed to be provided and not both)
           
           **sc_fraction** (float, 0.95) - Maximum possible extent to which the short circuit can be created on the line
                                 
           **overload_factor** - (float, 1.25)- Allowable over loading on the line used to calculate the backup pick up current
                                 
           **ct_current_factor** -(float, 1.2) - Current mutiplication factor to define the backup pick up current
           
           **safety_factor** -(float, 1) - Safety limit for the instantaneous pick up currents
                    
            
        OPTIONAL:
           
                                    
            **relay_trip_currents** - (DataFrame, None) - User defined relay trip currents given as a dataframe with columns as 'Switch_ID', 'Igg', 'Ig'
            
        
            **plot_grid** (bool, True) - Plot tripped grid based on the trip decisions. 
            
            **plot_annotations** (bool, True) -Plot the annotations in the tripped grid.
            
            **i_t_plot** (bool, True) -Plot the current-time relationshio (i-t plot).
        
        OUTPUT:
            **return** (list(Dict)) - Return trip decision of each relays
        """

    
    df_protection_settings =time_graded_overcurrent(net,timegrade,relay_trip_times)
    oc_relay_settings = []
    for switch_idx in net.switch.index:
        
        if (net.switch.closed.at[switch_idx]) & (net.switch.type.at[switch_idx] == "CB_non_dir") & (
                net.switch.et.at[switch_idx] == "l"):
            
            #time graded plan to be implemented here
      
            t_g=df_protection_settings.loc[df_protection_settings['switch_idx']==switch_idx]['t_g'].item()
            
            t_gg=df_protection_settings.loc[df_protection_settings['switch_idx']==switch_idx]['t_gg'].item()
            
            if relay_trip_currents is not None:
                
                oc_relay_parameter=list(relay_trip_currents.loc[switch_idx])
                settings = oc_parameters(net, switch_idx,t_g,t_gg, sc_fraction, overload_factor,ct_current_factor, safety_factor, oc_relay_parameter)

            else:
                settings = oc_parameters(net, switch_idx,t_g,t_gg, sc_fraction, overload_factor,ct_current_factor, safety_factor)

            oc_relay_settings.append(settings)
            
        
    net_sc = create_sc_bus(net, sc_line_idx, sc_location)
    sc.calc_sc(net_sc, bus=max(net_sc.bus.index), branch_results=True)

    trip_decisions = []
    
    for setting_idx in range(len(oc_relay_settings)):
        
        i_ka = oc_get_measurement_at_relay_location(net_sc, oc_relay_settings[setting_idx])
        
        trip_decision = oc_get_trip_decision(net_sc, oc_relay_settings[setting_idx], i_ka)

        trip_decisions.append(trip_decision)
    
    df_trip_decison = pd.DataFrame.from_dict(trip_decisions)
    df_decisions=df_trip_decison[["Switch","Switch type","Trip", "Fault Current","Trip time" ]]
    
    # Show only the necessary data
    df_decisions.columns = ['Switch ID', 'Switch type','Trip', 'Ikss [kA]', 'Trip time [s]']
    
    print(df_decisions)
    
    if plot_grid:
        #coming from utility functions
        if plot_annotations:
            plot_tripped_grid(net_sc, trip_decisions,sc_location,plot_annotations=True)
        else:
            plot_tripped_grid(net_sc, trip_decisions, sc_location,plot_annotations=False)
    # Plot I-T plot        
    if i_t_plot:
        switch_id=[]
        #plot only the instaneous trip decision
        for trip_idx in range(len(trip_decisions)):
            trip_decision = trip_decisions[trip_idx]
            trip_type = trip_decision.get("Trip Type")
            
            if trip_type == "instantaneous":
                switch_idx = trip_decision.get("Switch")
        switch_id.append(switch_idx)
        
        create_I_t_plot(trip_decisions,switch_id)
            
    return trip_decisions  


if __name__ == "__main__":
    
    from pandaplan.core.protection.implementation.example_grids import *
    
    net= oc_relay_net(open_loop=True)
    
    simple_plot(net, plot_loads=True, plot_sgens=True, plot_line_switches=True)

    # here user can define the time grade according to their choice # by default  Tgg is 0.07 and Tg=0.5 Tdelta=0.3
    trip_decision = run_fault_scenario_oc(net, sc_line_idx =3,sc_location = 0.5,
                                         timegrade=[0.07,0.5,0.3])

    #create_I_t_plot(trip_decision,switch_id=[0,3,4])
                           
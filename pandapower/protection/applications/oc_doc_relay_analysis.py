# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 12:03:28 2022

@author: amadhusoodhanan
"""


import pandapower.shortcircuit as sc
import copy
import pandapower as pp
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore') 
import sys
sys.path.append('C:\Arjun\git\pandaplan-core') 
from pandaplan.core.protection.implementation.utility_functions import *
from pandaplan.core.protection.implementation import doc_relay_model as doc
from pandaplan.core.protection.implementation import oc_relay_model as oc
from pandapower.plotting import simple_plot

def run_fault_scenario_oc_doc(net, sc_line_idx, sc_location,relay_configuration,timegrade=None,relay_trip_times=None,
                           sc_fraction=0.95, overload_factor=1.2,ct_current_factor=1.25,
                           safety_factor=1,relay_trip_currents=None,plot_annotations=True, 
                           plot_grid=True, i_t_plot=True):

    
    #Get trip decisions from doc relay model
    trip_decision_doc=doc.run_fault_scenario_doc(net, sc_line_idx=sc_line_idx, sc_location=sc_location,
                            relay_configuration=relay_configuration,
                            timegrade=timegrade,relay_trip_times=relay_trip_times,
                           sc_fraction= sc_fraction, overload_factor=overload_factor,ct_current_factor=ct_current_factor,
                           safety_factor=safety_factor,relay_trip_currents=relay_trip_currents,plot_annotations=False, 
                           plot_grid=False, i_t_plot=False)
    
    
    # Get trip decisions from oc relay model
    trip_decision_oc = oc.run_fault_scenario_oc(net, sc_line_idx =sc_line_idx, 
                   sc_location = sc_location,relay_trip_times=relay_trip_times,timegrade=timegrade,
                   relay_trip_currents=relay_trip_currents
                   ,plot_grid=False, plot_annotations=False,i_t_plot=False)

    
    # merge two trip decisions
    
    trip_decisions=trip_decision_oc+trip_decision_doc

    net_sc = create_sc_bus(net, sc_line_idx, sc_location)
    sc.calc_sc(net_sc, bus=max(net_sc.bus.index), branch_results=True)
    
    # Plotting functions
    
    if plot_grid:
        #coming from utility functions
        if plot_annotations:
            
            plot_tripped_grid(net_sc, trip_decisions, sc_location,plot_annotations=True)
        else:
            plot_tripped_grid(net_sc, trip_decisions, sc_location,plot_annotations=False)

        #I-T plot needs to be implemented
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
        
        
    # printing required decisions
    df_trip_decison = pd.DataFrame.from_dict(trip_decisions)
    
    df_decisions=df_trip_decison[["Switch", "Switch type","Trip", "Fault Current","Trip time" ]]
    
    # rename the header 
    df_decisions.columns = ['Relay ID', 'Relay type','Trip', 'Ikss [kA]', 'Trip time [ms]']
    
    print(df_decisions)
    
    return trip_decisions
 
 
if __name__ == "__main__":
    

    from pandaplan.core.protection.implementation.example_grids import *
     
    net = load_4bus_net_oc_doc(open_loop = False)
    
    relay_configuration = {"Switch_4": [3,  "CB_dir",     "forward",    86, 45],
                            "Switch_5": [4,  "CB_dir",     "forward",    86, 45]}
    
    relay_trip_times= pd.DataFrame({'Relay ID': [0,1,2,3,4],
                              'Tgg': [100,110,120,130,140],
                              'Tg':[1400,1300,1200,1100,1000]})
    
     
    #simple_plot(net, plot_loads=True, plot_sgens=True, plot_line_switches=True)
    
    trip_decisions= run_fault_scenario_oc_doc(net, sc_line_idx=4, sc_location=0.6 ,relay_configuration=relay_configuration,timegrade=[0.07,0.5,0.3],
                                  relay_trip_times=None,
                               sc_fraction=0.95, overload_factor=1.2,ct_current_factor=1.25,
                               safety_factor=1,relay_trip_currents=None,plot_annotations=True, 
                               plot_grid=True, i_t_plot=True)
    
    create_I_t_plot(trip_decisions,switch_id=[0,3,4])
        
    """
            required inputs:
                            1. net: pandapower network
                            2. if it includes doc relay:
                                relay configuration: if doc, the index, type, biasing, max sector angle and relay chara angle
                                need to be input
                            3. sc_line_idx: index of short circuit need to create
                            4. Sc_location: location of sc on line  (0<sc_fraction<1)
                            5. timegrade: list of Tgg (inst trip time), Tg (backup tim)e, Tdelta (time gap between each backup) 
                                or 
                                input Tgg and Tg for all relays as list
        
            Optional inputs
                            1. if closed network- input :
                                loop_line=line index (index of loop line)
                            2. by default plot and plot annotations are True, else
                                plot=True
                                plot_annotations=True
                            3.  oc paramters can be defined by user
                                by default: 
                                    sc_fraction=0.95,
                                    overload_factor=1.2,
                                    ct_current_factor=1.25,
                                    safety_factor=1
                            4. timegrading plan is predefined based on input (input timegrade)
                                if needed user can predefine tgg and tg of each relays
                            5. relay_trip_currents: 
                                oc relay parameters can be manualy set by user by providing as dataframe of required format,
                                if not will take 
                                                        default values
    """              
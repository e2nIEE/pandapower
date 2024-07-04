
# This function implemenets the protecton module using over current relay
# Copyright (c) 2016-2023 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pandapower.shortcircuit as sc
import copy
import numpy as np
import pandas as pd
import warnings
from math import isnan,nan
warnings.filterwarnings('ignore')
from pandapower.protection.utility_functions import create_sc_bus,bus_path_multiple_ext_bus,get_line_path,get_line_idx,get_bus_idx,\
                                                    parallel_lines,plot_tripped_grid, create_I_t_plot



# set the oc parameters
def oc_parameters(net,relay_type, time_settings,sc_fraction=0.95, overload_factor=1.2, ct_current_factor=1.25,
                          safety_factor=1,inverse_overload_factor=1.2, pickup_current_manual=None, **kwargs):
    """
    The main function is to create relay settings with oc parameters.

       INPUT:
           **net** (pandapowerNet) - Pandapower network and net.switch.type need to be specified as

           - 'CB_DTOC'    (for using Definite Time Over Current Relay)
           - 'CB_IDMT'    (Inverse Definite Minimum Time over current relay)
           - 'CB_IDTOC'  (Inverse Definite Minimum Time over current relay

           **relay_type** (string)- oc relay type need to be specifiied either as

            - DTOC: Definite Time Over Current Relay
            - IDMT: Inverse Definite Minimum Time over current relay
            - IDTOC: Inverse Definite Time Minimum over current relay (combination of DTOC and IDMT)


           **time_settings** (list or DataFrame) - Relay tripping time can be given as a list or a DataFrame

                If given as a list, the time grading will be calculated based on topological grid search, and manual tripping time can be provided as a dataframe by respecting the column names.

                For DTOC:
                time_settings =[t>>, t>, t_diff] or Dataframe columns as 'switch_id', 't_gg', 't_g'

                - t>> (t_gg): instantaneous tripping time in seconds
                - t> (t_g):  primary backup tripping time in seconds,
                - t_diff: time grading delay difference in seconds


                For IDMT:
                time_settings =[tms, t_delta] or Dataframe columns as 'switch_id', 'tms', 't_grade'

                - tms: time multiplier settings in seconds
                - t_grade:  time grading delay difference in seconds

                For IDTOC:
                time_settings =[t>>, t>, t_diff, tms,t_grade] or Dataframe columns as 'switch_id', 't_gg', 't_g','tms', 't_grade'

                - t>> (t_gg): instantaneous tripping time in seconds
                - t> (t_g):  primary backup tripping time in seconds,
                - t_diff: time grading delay difference in seconds
                - tms: time multiplier settings in seconds
                - t_grade:  time grading delay difference in seconds


           **sc_fraction** (float, 0.95) - Maximum possible extent to which the short circuit can be created on the line

           **overload_factor** - (float, 1.25)- Allowable overloading on the line used to calculate the pick-up current

           **ct_current_factor** -(float, 1.2) - Current multiplication factor to calculate the pick-up current

           **safety_factor** -(float, 1) - Safety limit for the instantaneous pick-up current

           **inverse_overload_factor** -(float, 1.2)- Allowable inverse overloading to define the pick-up current in IDMT relay


        OPTIONAL:
            **pickup_current_manual** - (DataFrame, None) - User-defined relay trip currents can be given as a dataframe.

                DTOC: Dataframe with columns as 'switch_id', 'I_gg', 'I_g'

                IDMT: Dataframe with columns as 'switch_id', 'I_s'

                IDTOC: Dataframe with columns as 'switch_id', 'I_gg', 'I_g', 'I_s'


        KWARGS:
            **curve_type**- (String) - Relay trip time will vary depending on the curve slope for inverse Time Relays. The curve is used to coordinate with other protective devices for selectivity (according to IEC60255)

                Curve type can be :

                - 'standard_inverse'
                - 'very_inverse',
                - 'extremely_inverse',
                - 'long_inverse',
        OUTPUT:
            **return** (DataFrame, float) - Return relay setting as a dataframe with required parameters for oc relay (DTOC, IDMT, IDTOC)
        """


    oc_relay_settings= pd.DataFrame(columns = ["switch_id","line_id","bus_id","relay_type", "curve_type","I_g[kA]","I_gg[kA]","I_s[kA]", "t_g[s]","t_gg[s]", "tms[s]",'t_grade[s]' "alpha","k"])
    for switch_id in net.switch.index:
        if (net.switch.closed.at[switch_id])  & (net.switch.et.at[switch_id] == "l"):

            net = copy.deepcopy(net)
            # get the line index and bus index from the line
            line_idx = get_line_idx(net, switch_id)
            bus_idx = get_bus_idx(net, switch_id)
            net_sc = create_sc_bus(net, line_idx, sc_fraction)
            sc.calc_sc(net_sc, bus=max(net_sc.bus.index), branch_results=True)

            if (net.switch.type.at[switch_id] == "CB_DTOC") and  relay_type=='DTOC': # Definite Time Over Current relay

                if pickup_current_manual is None:
                    I_g = net_sc.line.max_i_ka.at[line_idx] * overload_factor * ct_current_factor
                    I_gg = net_sc.res_line_sc.ikss_ka.at[line_idx] * safety_factor
                else:
                    I_g= pickup_current_manual.I_g.at[switch_id] # take manual inputs
                    I_gg=pickup_current_manual.I_gg.at[switch_id]

                df_protection_settings =time_grading(net,time_settings)
                t_gg=df_protection_settings.loc[df_protection_settings['switch_id']==switch_id]['t_gg'].item()
                t_g=df_protection_settings.loc[df_protection_settings['switch_id']==switch_id]['t_g'].item()
                curve_type="Definite time curve"
                t_grade=tms=alpha=k=I_s=float("NaN")

            if  (net.switch.type.at[switch_id] == "CB_IDMT") and relay_type=='IDMT':

                # Inverse over current relay according to  IEC 60255-3/BS142
                if pickup_current_manual is None:
                    I_s=net_sc.line.max_i_ka.at[line_idx]*inverse_overload_factor
                else:
                    I_s= pickup_current_manual.I_s.at[switch_id] # take manual inputs

                #k and alpha are the curve constants for IEC standards
                if kwargs['curve_type']=='standard_inverse':
                    k=0.140;
                    alpha=0.020
                if kwargs['curve_type']=='very_inverse':
                     k=13.5
                     alpha=1
                if kwargs['curve_type']=='extremely_inverse':
                     k=80;
                     alpha=2
                if kwargs['curve_type']=='long_inverse':
                     k=120;
                     alpha=1

                curve_type=kwargs['curve_type']

                time_grading(net,time_settings)

                df_protection_settings =time_grading(net,time_settings)
                t_grade=df_protection_settings.loc[df_protection_settings['switch_id']==switch_id]['t_g'].item()
                tms=df_protection_settings.loc[df_protection_settings['switch_id']==switch_id]['t_gg'].item()
                #tms= t_grade
                I_g=I_gg=t_g=t_gg=float("NaN")


            if  (net.switch.type.at[switch_id] == "CB_IDTOC") and relay_type=='IDTOC' : # Inverse Definite Time relay

                if pickup_current_manual is None:
                    I_g = net_sc.line.max_i_ka.at[line_idx] * overload_factor * ct_current_factor
                    I_gg = net_sc.res_line_sc.ikss_ka.at[line_idx] * safety_factor
                    I_s=inverse_overload_factor* net_sc.line.max_i_ka.at[line_idx]
                else:
                    I_g= pickup_current_manual.I_g.at[switch_id] # take manual inputs
                    I_gg=pickup_current_manual.I_gg.at[switch_id]
                    I_s= pickup_current_manual.I_s.at[switch_id]

                if len(time_settings)!=5:
                    assert "length of time_setting for DTOC is a list of order 5 with t_gg,t_g,t_diff, tms,t_grade"

                if isinstance(time_settings, list):
                    df_protection_settings =time_grading(net,[time_settings[0], time_settings[1], time_settings[2]])
                    t_gg=df_protection_settings.loc[df_protection_settings['switch_id']==switch_id]['t_gg'].item()
                    t_g=df_protection_settings.loc[df_protection_settings['switch_id']==switch_id]['t_g'].item()

                    df_protection_settings =time_grading(net,[time_settings[3], time_settings[4]])
                    tms=df_protection_settings.loc[df_protection_settings['switch_id']==switch_id]['t_gg'].item()
                    t_grade=df_protection_settings.loc[df_protection_settings['switch_id']==switch_id]['t_g'].item()

                if isinstance(time_settings, pd.DataFrame):
                    df_protection_settings =time_grading(net,time_settings)
                    t_gg=df_protection_settings.loc[df_protection_settings['switch_id']==switch_id]['t_gg'].item()
                    t_g=df_protection_settings.loc[df_protection_settings['switch_id']==switch_id]['t_g'].item()
                    tms=df_protection_settings.loc[df_protection_settings['switch_id']==switch_id]['tms'].item()
                    t_grade=df_protection_settings.loc[df_protection_settings['switch_id']==switch_id]['t_grade'].item()

                #k and alpha are the curve constants for IEC standards
                if kwargs['curve_type']=='standard_inverse':
                    k=0.14
                    alpha=0.020
                if kwargs['curve_type']=='very_inverse':
                     k=13.5
                     alpha=1
                if kwargs['curve_type']=='extremely_inverse':
                     k=80;
                     alpha=2
                if kwargs['curve_type']=='long_inverse':
                     k=120;
                     alpha=1
                curve_type=kwargs['curve_type']

            settings = pd.DataFrame([{"switch_id": switch_id, "line_id": line_idx, "bus_id": bus_idx,"relay_type":relay_type, "curve_type":curve_type,"I_g[kA]": I_g, "I_gg[kA]": I_gg, "I_s[kA]":I_s, "t_g[s]": t_g, "t_gg[s]": t_gg,  "tms[s]":tms, "t_grade[s]":t_grade, "alpha":alpha,"k":k}])

            oc_relay_settings = pd.concat([oc_relay_settings,settings],ignore_index=True)
            oc_relay_settings = oc_relay_settings.dropna(how='all', axis=1)


    return  oc_relay_settings

    # get the short circuit current values in the lines
def oc_get_measurement_at_relay_location(net, settings):
    switch_id = settings["switch_id"]
    line_idx = get_line_idx(net, switch_id)
    i_ka = net.res_line_sc.ikss_ka.at[line_idx]
    return i_ka

    # based on the short circuit currents, the trip decision is taken
def oc_get_trip_decision(net, settings, i_ka):
    switch_id = settings["switch_id"]
    if settings["relay_type"] == "DTOC":
        max_ig = settings["I_g[kA]"]
        max_igg = settings["I_gg[kA]"]
        t_g = settings["t_g[s]"]
        t_gg = settings["t_gg[s]"]
        relay_type=settings["relay_type"]
        t=I_s=nan

        if i_ka > max_igg:
            trip = True
            trip_type = "instantaneous"
            trip_time = t_gg

        elif i_ka > max_ig:
            trip = True
            trip_type = "backup"
            trip_time = t_g

        else:
            trip = False
            trip_type = "no trip"
            trip_time = np.inf


    if settings["relay_type"] == "IDMT":

        I_s=settings["I_s[kA]"]
        k=settings["k"]
        alpha=settings["k"]
        relay_type=settings["relay_type"]
        trip_type=settings['curve_type']
        t_grade=settings["t_grade[s]"]
        tms=settings["tms[s]"]
        t=(tms*k)/(((i_ka/I_s)**alpha)-1)+t_grade
        max_ig=max_igg=t_g=t_gg=nan

        if i_ka > I_s:
            trip = True
            trip_type = "instantaneous"
            trip_time = t

        else:
            trip = False
            trip_type = "no trip"
            trip_time = np.inf

      # combination of DTOC and IDMT
    if settings["relay_type"] == "IDTOC":
        max_ig = settings["I_g[kA]"]
        max_igg = settings["I_gg[kA]"]
        t_g = settings["t_g[s]"]
        t_gg = settings["t_gg[s]"]
        t_grade=settings["t_grade[s]"]


        tms=settings["tms[s]"]
        I_s=settings["I_s[kA]"]
        k=settings["k"]
        alpha=settings["k"]
        relay_type=settings["relay_type"]
        trip_type=settings['curve_type']
        t=(tms*k)/(((i_ka/I_s)**alpha)-1)+t_grade

        if i_ka > max_igg:
            trip = True
            trip_type = "instantaneous"
            trip_time = t_gg

        elif i_ka > max_ig:
            trip = True
            trip_type = "backup"
            trip_time = t_g

        elif i_ka > I_s:
            trip = True
            trip_type = "inverse instantaneous"
            trip_time = t

        else:
            trip = False
            trip_type = "no trip"
            trip_time = np.inf


    trip_decision = {"Switch ID": switch_id,'Switch type':relay_type, 'Trip type':trip_type, 'Trip': trip, "Fault Current [kA]": i_ka,
                 "Trip time [s]": trip_time, "Ig": max_ig, "Igg": max_igg, "Is":I_s, 'tg':t_g, 'tgg':t_gg, 't_s':t}
    # filter out the nan values if any from trip decisions
    filter(lambda k: not isnan( trip_decision[k]),  trip_decision)
    return trip_decision

    # Time graded protection (backup) for over current relay

    # Get the bus path from all the ext grids
def time_grading(net,time_settings):

    # Automated time grading calculation
    if isinstance(time_settings,list):
        if len(time_settings)==3:
            time_settings=time_settings
        if len(time_settings)==2:
            time_settings=[time_settings[0], time_settings[1], time_settings[1]]

        # Get the bus path to each lines from all the ext buses
        bus_paths=bus_path_multiple_ext_bus(net)

        # Get all the line paths
        line_paths=[]
        # get sorted line path (last is the longest line path)
        for bus_path in bus_paths:
            line=get_line_path(net, bus_path)
            line_paths.append(line)
            sorted_line_path = sorted(line_paths, key=len)

        #assign tg based on longest line path
        line_length_time={}
        for line_length in range(0,len(sorted_line_path[-1])) :

            distance_counter=len(sorted_line_path[-1])- line_length
            t_g=time_settings[1] + (line_length)* time_settings[2]
            line_length_time[ distance_counter]=t_g


        # line_time gets line id and time
        line_time={}
        for line in sorted_line_path:
            line_length=len(line)
            for length in line_length_time:
                if line_length==length:
                    line_time[line[-1]]=line_length_time[length]
        parallel_line=parallel_lines(net)

        # find the missing lines in line time (due to parallel lines)
        missing_line=[]
        for line in net.line.index:

            linecheck = []
            for key in line_time:

                if line == key:
                    linecheck.append(True)
                else:
                    linecheck.append(False)

            if any(linecheck):

                pass
            else:
                missing_line.append(line)


        # assign time to parallel line from the original time of line

        for parallel in parallel_line:

            for line in missing_line:

                if parallel[0]==line:

                    if parallel[1] not in line_time:
                        pass

                    else:
                        line_time[line]=line_time[parallel[1]]

                if parallel[1]==line:

                    if parallel[0] not in line_time:
                        pass

                    else:
                        line_time[line]=line_time[parallel[0]]


        # Assign time setting to switches
        time_switches={}
        tg_sw_setting=[]
        for switch in net.switch[net.switch.closed == True].index:
            line_id=net.switch[net.switch.closed == True].element.at[switch]

            tg = line_time[line_id]
            time_switches[switch]=tg
            tgg=time_settings[0]
            tg_sw_setting.append([switch,tg,tgg])

            # if there is multiple time setting for each switch take only the highest time
        protection_time_settings = pd.DataFrame(tg_sw_setting)
        protection_time_settings.columns=["switch_id","t_g","t_gg"]
        protection_time_settings= protection_time_settings.sort_values(by=['switch_id'])
        protection_time_settings=protection_time_settings.reset_index(drop=True)

 # Manual time grading settings
    if  isinstance(time_settings, pd.DataFrame):
        protection_time_settings=pd.DataFrame(columns = ["switch_id","t_gg","t_g"])

        if time_settings.columns.values.tolist()==['switch_id', 't_gg', 't_g']:
            protection_time_settings['switch_id']=time_settings['switch_id']
            protection_time_settings['t_g']=time_settings['t_g']
            protection_time_settings['t_gg']=time_settings['t_gg']

        if time_settings.columns.values.tolist()==['switch_id', 'tms', 't_grade']:
            protection_time_settings['switch_id']=time_settings['switch_id']
            protection_time_settings['t_g']=time_settings['t_grade']
            protection_time_settings['t_gg']=time_settings['tms']

        if time_settings.columns.values.tolist()==['switch_id', 't_gg', 't_gg', 'tms','t_grade']:
            protection_time_settings['switch_id']=time_settings['switch_id']
            protection_time_settings['t_g']=time_settings['t_g']
            protection_time_settings['t_gg']=time_settings['t_gg']
            protection_time_settings['t_grade']=time_settings['t_grade']
            protection_time_settings['tms']=time_settings['tms']

    return protection_time_settings


def run_fault_scenario_oc(net, sc_line_id, sc_location,relay_settings):

    """
    The main function is to create fault scenarios in the network at the defined location to get the tripping decisions.

       INPUT:
           **net** (pandapowerNet) - Pandapower network

           **sc_line_id** (int, index)- Index of the line to create the short circuit

           **sc_location** (float)- The short circuit location on the given line id (between 0 and 1).

           **relay_settings** (Dataframe, float)- Relay setting given as a dataframe returned from oc parameters (manual relay settings given as dataframe by respecting the column names)

                - DTOC:

                    Dataframe with columns as 'switch_id', 'line_id', 'bus_id', 'relay_type', 'I_g[kA]', 'I_gg[kA]', 't_g[s]', 't_gg[s]'

                - IDMT:

                    Dataframe with columns as 'switch_id', 'line_id', 'bus_id', 'relay_type', 'curve_type', I_s[kA], 'tms[s]', 't_grade[s], 'k', 'alpha'

                    - k and alpha are the curve constants according  to  IEC-60255

                - IDTOC:

                    Dataframe with columns as 'switch_id', 'line_id', 'bus_id', 'relay_type', 'I_g[kA]', 'I_gg[kA]', 'I_s[kA], 't_g[s]', 't_gg[s]', 'tms[s], 't_grade[s], 'k', 'alpha'

        OUTPUT:
            **return** (list(Dict),net_sc) - Return trip decision of each relay and short circuit net
        """


    net_sc = create_sc_bus(net, sc_line_id, sc_location)
    sc.calc_sc(net_sc, bus=max(net_sc.bus.index), branch_results=True)

    trip_decisions = []

    for index, settings in relay_settings.iterrows():

        i_ka = oc_get_measurement_at_relay_location(net_sc, settings)

        trip_decision = oc_get_trip_decision(net_sc, settings, i_ka)

        trip_decisions.append(trip_decision)

    df_trip_decison = pd.DataFrame.from_dict(trip_decisions)
    df_decisions=df_trip_decison[["Switch ID","Switch type","Trip type","Trip","Fault Current [kA]","Trip time [s]"]]

    print(df_decisions)

    return trip_decisions, net_sc


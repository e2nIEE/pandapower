# this function combines the oc and doc relay for analysis purposes
# Copyright (c) 2016-2022 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.
import pandapower.shortcircuit as sc
import copy
import pandapower as pp
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore') 
from pandapower.protection.implementation.utility_functions import *
from pandapower.protection.implementation import doc_relay_model as doc
from pandapower.protection.implementation import oc_relay_model as oc
from pandapower.plotting import simple_plot

def run_fault_scenario_oc_doc(net, sc_line_id, sc_location,relay_configuration,tripping_time_auto=None,tripping_time_manual=None,
                           sc_fraction=0.95, overload_factor=1.2,ct_current_factor=1.25,
                           safety_factor=1,pickup_current_manual=None):

    
    # Get trip decisions from oc relay model
    trip_decision_oc,net_Sc_oc = oc.run_fault_scenario_oc(net, sc_line_id =sc_line_id, 
                   sc_location = sc_location,tripping_time_manual=tripping_time_manual,tripping_time_auto=tripping_time_auto,
                   pickup_current_manual=pickup_current_manual)
    
    #Get trip decisions from doc relay model
    trip_decision_doc,net_sc_doc=doc.run_fault_scenario_doc(net, sc_line_id=sc_line_id, sc_location=sc_location,
                            relay_configuration=relay_configuration,
                            tripping_time_auto=tripping_time_auto,tripping_time_manual=tripping_time_manual,
                           sc_fraction= sc_fraction, overload_factor=overload_factor,ct_current_factor=ct_current_factor,
                           safety_factor=safety_factor,pickup_current_manual=pickup_current_manual)
    

    
    # merge two trip decisions
    trip_decisions=trip_decision_oc+trip_decision_doc
    # printing required decisions
    df_trip_decison = pd.DataFrame.from_dict(trip_decisions)

    return trip_decisions
 
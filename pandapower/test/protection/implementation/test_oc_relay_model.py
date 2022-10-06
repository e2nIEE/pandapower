
import pandas as pd
import pytest
import pandapower as pp
import pandapower.networks as pn
import pytest
import numpy as np
import pandas as pd
import logging as log
import pandapower
import pandapower.shortcircuit as sc
logger = log.getLogger(__name__)
try:
    from plotly import __version__
    PLOTLY_INSTALLED = True
except ImportError:
    PLOTLY_INSTALLED = False
from unittest.mock import patch 
import matplotlib.pyplot as plt 
from pandas.testing import assert_frame_equal
from pandapower.protection.implementation import oc_relay_model as oc
from pandapower.protection.implementation.example_grids import *
from pandapower.protection.implementation.utility_functions import *

# This function test the if manual configuration of oc parameters is provided
def test_oc_parameters_automated(switch_id = 2, t_gg=0.07, t_g=0.5, sc_fraction=0.95,
                              overload_factor=1.2, ct_current_factor=1.25,safety_factor=0.9):
    
    net=load_6bus_net_directional(open_loop=True)
    # sanity chech of user defined input oc parameters
    
    assert type(net)==pandapower.auxiliary.pandapowerNet, 'net should be pandapower network'
    assert switch_id in net.switch.index, 'given switch id shoulb be in the switch index of the given network'
    assert  t_g> t_gg, 't_g shoulbe be greater than t_gg'
    assert 0<sc_fraction<1 , 'sc fraction should be between 0 and 1'
    assert 1<=overload_factor , 'overload should be between greater than or equal to 1'
    assert 0<ct_current_factor, 'ct current factor should be greater than 0'
    assert safety_factor<=0.95 , 'safety current factor should be less than 0.95'

    
    # get relay parameters from oc relay based on input parameters
    relay_parameters = oc.oc_parameters(net, switch_id, t_g, t_gg, sc_fraction,
                                        overload_factor, ct_current_factor, safety_factor,oc_pickup_current_manual=None)

    #rtol = (B1 - A1)/A1 = 10
    #atol=B1-A1
    assert np.isclose(relay_parameters['Ig_ka'],0.42899999999999994, rtol=0.0001), 'Ig_ka at the given location should be within 0.1 A tolerence'
    
    assert np.isclose(relay_parameters['Igg_ka'], 1.8285431678393391, rtol=0.0001), 'Igg_ka at the given location should be within 0.1 A relative tolerence'

    assert relay_parameters['Igg_ka']>relay_parameters['Ig_ka'],  'Igg should be alwary greater than Ig'


def test_oc_parameters_manual(switch_id=4,t_g=0.5,t_gg=0.07, relay_parameters= pd.DataFrame({'Relay ID': [4],'Igg': [1.7],'Ig':[0.8]})):
    # The default in the function are used for calculating oc parameters if no user defined values are provided
    net=load_6bus_net_directional(open_loop=True)
    # get relay parameters from oc relay based on input parameters
    parameters = oc.oc_parameters(net, switch_id, t_g, t_gg,oc_pickup_current_manual= relay_parameters.loc[0])
    
    # expected result of oc parameters settings
    res_relay_settings = {"net": net, "switch_id": 4, "line_idx": 4, "bus_idx":4,
                    "Ig_ka": 0.8, "Igg_ka": 1.7, "tg": 0.5, "tgg": 0.07}
    
    assert  res_relay_settings.keys()==parameters.keys(), 'oc parameters should be of the given format'
    assert list(res_relay_settings)==list(parameters), 'oc parameters should have same values'

       
        
def test_oc_get_measurement_at_relay_location(sc_line_id=3, sc_location=0.3,  settings = {"switch_id":3}):
    
    net=load_6bus_net_directional(open_loop=True)
    # input sanity check
    assert sc_line_id in net.line.index , 'sc_line_id should be in the given network line index'
    assert  0<sc_location<1,'sc location should be be between 0 and 1'
    assert type(settings)==dict, 'input settings with switch index should be dictionary with id as switch_id'
    assert settings.get("switch_id") in net.switch.index, 'switch id should be in the given network'
    
    # create sc for given line and location
    net_sc = create_sc_bus(net, sc_line_id, sc_location)
    sc.calc_sc(net_sc, bus=max(net_sc.bus.index), branch_results=True)
    
    i_ka= oc.oc_get_measurement_at_relay_location(net_sc,settings)
    
    assert np.isclose(i_ka, 2.5983119767620106, rtol=0.00001), 'I_ka at the given location should be within 0.1 A tolerence with calculated value'


def test_oc_get_trip_decision(settings = {"net": None, "switch_id":7,"Ig_ka":0.429 , "Igg_ka": 1.68, "tg": 0.5, "tgg": 0.07}, i_ka1=1.808, i_ka2=0.808,  i_ka3=0 ):
    
    net=load_6bus_net_directional(open_loop=True)

    settings['net']=net
    
    # input sanity check
    assert settings.get("switch_id") in net.switch.index, 'switch id should be in the given network'
    
    # think how to change ika1, ika2, ika3
    
    inst_trip_decisions=oc.oc_get_trip_decision(net, settings, i_ka1)
    assert  inst_trip_decisions['Trip Type'] == "instantaneous" , 'i_ka >Igg  trip type should be instantaneous'
    assert  inst_trip_decisions['Trip time [s]'] == settings.get("tgg"), 'instantaneous trip have trip time equal to Tgg'
    
    
    backup_trip_decisions=oc.oc_get_trip_decision(net, settings, i_ka2)
    
    assert  backup_trip_decisions['Trip Type'] == "backup" , 'i_ka >Ig  trip type should backup'
    assert  backup_trip_decisions['Trip time [s]'] == settings.get("tg"), 'backup trip should have trip time equal to Tg'
    
    no_trip_trip_decisions=oc.oc_get_trip_decision(net, settings, i_ka3)
    
    assert  no_trip_trip_decisions['Trip Type'] == "no trip" , 'i_ka <Ig  no trip'
    assert  no_trip_trip_decisions['Trip time [s]'] ==np.inf, 'If no trip, trip time should be infinity'    



def test_time_graded_overcurrent_manual(tripping_time_manual= pd.DataFrame({'switch_id': [0,1,2,3,4,5,6,7],
                                                                         't_g':[0.5,0.8,1.1,1.4,1.7,2,2.5,2.8],
                                                                         't_gg': [0.05,0.06,0.07,0.07,0.05,0.05,0.07,0.08]})):
    
    net=load_6bus_net_directional(open_loop=True)

    assert len(net.switch[net.switch.closed == True])==len(tripping_time_manual),'Number of active relays in the network should be equal to the number of relay in relay trip times'
    
    assert tripping_time_manual.columns.values.tolist()==['switch_id', 't_g', 't_gg'], 'Input relay times  should have the given format and header name'
    
    times=oc.time_graded_overcurrent(net,tripping_time_manual=tripping_time_manual, tripping_time_auto=None)
    
    # check input switch time equal to in the times from tripping_time_auto
    assert tripping_time_manual.equals(times), 'given relay time settings should be equal to the values in the tripping_time_auto function'
    

def test_time_graded_overcurrent_automated(tripping_time_auto=[0.07,0.5,0.3]): # tripping_time_auto=[tgg, tg, t_delta]
    
    net=load_6bus_net_directional(open_loop=True)
    # Input sanity check 
    
    times=oc.time_graded_overcurrent(net,tripping_time_auto, tripping_time_manual=None)
    tripping_time_manual= pd.DataFrame({'switch_id': [0,1,2,3,4,5,6,7],
                                    't_g':[1.7,1.1,0.8,1.4,1.1,0.8,0.5,0.5],
                                    't_gg': [0.07,0.07,0.07,0.07,0.07,0.07,0.07,0.07]})
    
    assert tripping_time_manual.equals(times), 'given relay time settings hould be equal from the tripping_time_auto function'

@patch("matplotlib.pyplot.show")
@pytest.mark.slow

def test_plot_tripped_grid(mock_show,sc_line_id =7,sc_location =0.4,tripping_time_auto=[0.07,0.5,0.3] ):
    net=load_6bus_net_directional(open_loop=True)
    # generate trip decisions
    trip_decisions,net_sc = oc.run_fault_scenario_oc(net, sc_line_id, sc_location,
                                           tripping_time_auto=tripping_time_auto)
    
    # test the tripped grid
    oc.plot_tripped_grid(net_sc, trip_decisions,sc_location, plot_annotations=(True))
    plt.close('all')
    
    
@patch("matplotlib.pyplot.show")
@pytest.mark.slow

def test_plot_create_I_t_plot(mock_show,sc_line_id =7,sc_location =0.4,tripping_time_auto=[0.07,0.5,0.3],switch_id=[0,3,4] ):
    net=load_6bus_net_directional(open_loop=True)
    # generate trip decisions
    trip_decisions,net_sc = oc.run_fault_scenario_oc(net, sc_line_id, sc_location,
                                           tripping_time_auto=tripping_time_auto)

    # test the IT plot function working or not
    oc.create_I_t_plot(trip_decisions,switch_id)

    plt.close('all')

if __name__ == '__main__':
    pytest.main(['-x',__file__])
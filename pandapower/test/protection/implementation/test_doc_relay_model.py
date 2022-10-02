
#This function test the function of doc relay model


import pytest
import numpy as np
import pandas as pd
import pandapower as pp
import logging as log
import pandapower
import pandapower.shortcircuit as sc
logger = log.getLogger(__name__)
try:
    from plotly import __version__
    PLOTLY_INSTALLED = True
except ImportError:
    PLOTLY_INSTALLED = False
import warnings
warnings.filterwarnings("ignore")
from unittest.mock import patch 
import matplotlib.pyplot as plt 
from pandas.testing import assert_frame_equal
import pandapower.protection.implementation.doc_relay_model as doc
from pandapower.protection.implementation.example_grids import *
from pandapower.protection.implementation.utility_functions import *


# This function test the if manual configuration of doc parameters
def test_doc_parameters_automated(switch_id = 2, t_igg=0.07, t_ig=0.5, sc_fraction=0.95,
                              overload_factor=1.2, ct_current_factor=1.25,safety_factor=0.9,
                              relay_configuration= {"switch_3": [2,"CB_dir",
                                                    "forward",86, 45]}):
    
    net =  load_4bus_net(open_loop =False)
    
    # sanity chech of user defined input doc parameters
    assert type(net)==pandapower.auxiliary.pandapowerNet,\
        'net should be pandapower network'
    assert switch_id in net.switch.index,\
        'given switch id shoulb be in the switch index of the given network'
    assert  t_ig> t_igg, 't_g shoulbe be greater than t_gg'
    assert 0<sc_fraction<1 , 'sc fraction should be between 0 and 1'
    assert 1<=overload_factor , 'overload should be between greater than or equal to 1'
    assert 0<ct_current_factor, 'ct current factor should be greater than 0'
    assert safety_factor<=0.95 , 'safety current factor should be less than 0.95'

    
    # get relay parameters from doc relay based on input parameters
    relay_parameters = doc.doc_parameters(net, switch_id, t_ig, t_igg,
                                          relay_configuration,sc_fraction,
                                        overload_factor, ct_current_factor, safety_factor,doc_pickup_current_manual=None)

    #rtol = (B1 - A1)/A1 = 10
    #atol=B1-A1
    assert np.isclose(relay_parameters['Ig_ka'],0.42899999999999994, rtol=0.0001),\
        'Ig_ka at the given location should be within 0.0001 A tolerence'
    
    assert np.isclose(relay_parameters['Igg_ka'], 1.1401695752802623, rtol=0.0001),\
        'Igg_ka at the given location should be within 0.0001 A relative tolerence'

    assert relay_parameters['Igg_ka']>relay_parameters['Ig_ka'],\
        'Igg should be alwary greater than Ig'
    
    assert relay_parameters['OSA']==relay_configuration.get("switch_3")[3],\
        'Maximum sector angle should be equal to the input configuration'
    
    assert relay_parameters['RCA']==relay_configuration.get("switch_3")[4],\
        'Relay characteristics angle should be equal to the input configuration'

    assert relay_parameters['direction']==relay_configuration.get("switch_3")[2],\
        'Direction of relay should be equal to the input configuration'

    assert np.isclose(relay_parameters['MTA'],41, rtol=0.0001),\
        'Maximum toque angle at for the relay should be within 0.001 degree tolerence'

    

    # Test the manual configuration of doc parameters
def test_doc_parameters_manual(switch_id=4,t_ig=0.5,t_igg=0.07, 
                               relay_parameters= pd.DataFrame({'switch_id': [4],
                                                               'I_gg': [1.7],'I_g':[0.8]}),
                               relay_configuration= {"switch_5": [4,  "CB_dir",
                                                    "forward",86, 45]}):
    
    net =  load_4bus_net(open_loop =False)

    # get relay parameters from doc relay based on input parameters
    relay_settings = doc.doc_parameters(net, switch_id, t_ig, t_igg,
                     relay_configuration,doc_pickup_current_manual= relay_parameters.iloc[0])
    
    # expected result of doc parameters settings
    res_relay_settings = {"net": net, "switch_id": 4, "line_idx": 4, "bus_idx":4,
                    "Ig_ka": 0.8, "Igg_ka": 1.7,"tg": 0.5, "tgg": 0.07, "MTA": 41,
                    "direction":"forward","OSA":86, "RCA": 45}
    
    #test the expected format
    assert res_relay_settings.keys()==relay_settings.keys(),\
        'oc parameters should be of the given format'
    assert list(res_relay_settings)==list(relay_settings),\
        'oc parameters should have same values'
    
    #test the current values
    assert relay_settings['Igg_ka']==relay_parameters.iloc[0][1],\
        'Igg should be equal to given input'
    assert relay_settings['Ig_ka']==relay_parameters.iloc[0][2],\
        'Ig should be equal to given input'



    #test get measurement at relay location
def test_doc_get_measurement_at_relay_location(sc_line_idx=3, sc_location=0.3,
                                               settings = {"switch_id":3}):
    
    net =  load_4bus_net(open_loop =False)

    # input sanity check
    assert sc_line_idx in net.line.index ,\
        'sc_line_idx should be in the given network line index'
    assert  0<sc_location<1,'sc location should be be between 0 and 1'
    assert type(settings)==dict,\
        'input settings with switch index should be dictionary with id as switch_id'
    assert settings.get("switch_id") in net.switch.index,\
        'switch id should be in the given network'
    
    # create sc for given line and location
    net_sc = create_sc_bus(net, sc_line_idx, sc_location)
    sc.calc_sc(net_sc, bus=max(net_sc.bus.index), branch_results=True)
    
    i_ka,vi_angle= doc.doc_get_measurement_at_relay_location(net_sc,settings)
    
    assert np.isclose(i_ka, 0.33642179104125747, rtol=0.00001),\
        'I_ka at the given location should be within 0.00001 A,\
            tolerence with calculated value'
    
    assert np.isclose(vi_angle, 26.314913191501308, rtol=0.00001),\
        'vi_angle at the given location should be within 0.00001,\
            degree tolerence with calculated value'

    #test the doc get trip decisions
def test_doc_get_trip_decision(switch_id=4,sc_line_idx=4 ,sc_location=0.5, 
                            settings = {"net": None, "switch_id":4,"Ig_ka":0.429 ,
                                        "Igg_ka": 1.68, "tg": 0.5,
                                        "tgg": 0.07,"MTA": 41,
                                        "direction":"forward",
                                        "OSA":86, "RCA": 45},
                             i_ka1=1.808,i_ka2=0.808,i_ka3=0,
                             vi_angle1=21,vi_angle2=21,vi_angle3=150):    
    
    net =  load_4bus_net(open_loop =False)

    settings['net']=net
    
    # input sanity check
    assert settings.get("switch_id") in net.switch.index,\
        'switch id should be in the given network'
    
    # think how to change ika1, ika2, ika3
    net_sc = create_sc_bus(net, switch_id, sc_location)
    sc.calc_sc(net_sc, bus = max(net_sc.bus.index), branch_results = True)
    
    inst_trip_decisions=doc.doc_get_trip_decision(net_sc, settings, i_ka1,vi_angle1)
    assert  inst_trip_decisions['Trip Type'] == "instantaneous" ,\
        'i_ka >Igg, forward zone and forward biased or reverse zone and,\
            reverse biased for trip type instantaneous'
    assert  inst_trip_decisions['Trip time [s]'] == settings.get("tgg"),\
        'instantaneous trip have trip time equal to Tgg'
    
    
    backup_trip_decisions=doc.doc_get_trip_decision(net_sc, settings, i_ka2,vi_angle2)
    assert  backup_trip_decisions['Trip Type'] == "backup" , \
        'i_ka >Ig  forward zone and forward biased or reverse zone and reverse biased,\
            for trip type as backup'
    assert  backup_trip_decisions['Trip time [s]'] == settings.get("tg"),\
        'backup trip should have trip time equal to Tg'

    
    no_trip_trip_decisions=doc.doc_get_trip_decision(net_sc, settings, i_ka3,vi_angle3)
    assert  no_trip_trip_decisions['Trip Type'] == "no trip" ,\
        'i_ka <Ig, forward zone and reverse biased or reverse zone and forward biased then no trip'
    assert  no_trip_trip_decisions['Trip time [s]'] ==np.inf, 'If no trip, trip time should be infinity'


    no_trip_trip_decisions_dir=doc.doc_get_trip_decision(net_sc, settings, i_ka1,vi_angle3)
    assert  no_trip_trip_decisions_dir['Trip Type'] == "no trip" , \
        'i_ka >Igg, forward zone and reverse biased or reverse zone and forward biased  no trip'
    assert  no_trip_trip_decisions_dir['Trip time [s]'] ==np.inf, 'If no trip, trip time should be infinity'

    #Test the manual time grading plan
def test_time_graded_overcurrent_manual(tripping_time_manual= pd.DataFrame({'switch_id': [0,1,2,3,4],
                                        't_g':[0.5,0.8,1.1,1.4,1.7],
                                        't_gg': [0.05,0.06,0.07,0.07,0.05]})):
    # input sanity check
    net =  load_4bus_net(open_loop =False)
    assert len(net.switch[net.switch.closed == True])==len(tripping_time_manual),\
        'Number of active relays in the network should be equal to the number of relay in relay trip times'
    
    assert tripping_time_manual.columns.values.tolist()==['switch_id', 't_g', 't_gg'],\
        'Input relay times  should have the given format and header name'
    
    times=doc.time_graded_overcurrent(net,tripping_time_manual=tripping_time_manual, tripping_time_auto=None)
    
    # check input switch time equal to in the times from tripping_time_auto
    assert tripping_time_manual.equals(times), \
        'given relay time settings should be equal to the values in the tripping_time_auto function'
    
    #Test the autmated time grading plan
def test_time_graded_overcurrent_automated(tripping_time_auto=[0.07,0.5,0.3]): # tripping_time_auto=[tgg, tg, t_delta]
    
    net =  load_4bus_net(open_loop =False)

    # Input sanity check 
    times=doc.time_graded_overcurrent(net,tripping_time_auto, tripping_time_manual=None)
    tripping_time_manual= pd.DataFrame({'switch_id': [0,1,2,3,4],
                                    't_g':[1.1,0.5,0.8,0.5,0.5],
                                    't_gg': [0.07,0.07,0.07,0.07,0.07]})
    
    assert tripping_time_manual.equals(times),\
        'given relay time settings hould be equal from the tripping_time_auto function'


# Test the tripped grid
@patch("matplotlib.pyplot.show")
@pytest.mark.slow
#@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly functions require the plotly package")

def test_plot_tripped_grid(mock_show,sc_line_idx =4,sc_location =0.4,
                           tripping_time_auto=[0.07,0.5,0.3],relay_configuration = {
                    "switch_1": [0,  "CB_dir",     "forward",    86, 45],
                    "switch_2": [1,  "CB_dir",     "forward",    86, 45], 
                    "switch_3": [2,  "CB_dir",     "forward",    86, 45],
                    "switch_4": [3,  "CB_dir",     "forward",    86, 45],
                    "switch_5": [4,  "CB_dir",     "forward",    86, 45]}):
    
    net=load_4bus_net(open_loop =False)
    # generate trip decisions
    trip_decisions,net_sc = doc.run_fault_scenario_doc(net,sc_line_idx, sc_location,relay_configuration,
                                                tripping_time_auto=tripping_time_auto)
    
    # test the tripped grid
    doc.plot_tripped_grid(net_sc, trip_decisions,sc_location, plot_annotations=(True))
    plt.close('all')
    
    
# Test the I-T plot   
@patch("matplotlib.pyplot.show")
@pytest.mark.slow
#@pytest.mark.skipif(not PLOTLY_INSTALLED, reason="plotly functions require the plotly package")

def test_create_I_t_plot(mock_show,sc_line_idx =4,sc_location =0.4,
                         tripping_time_auto=[0.07,0.5,0.3],switch_id=[0,3,4],
                         relay_configuration = {
                        "switch_1": [0,  "CB_dir",     "forward",    86, 45],
                        "switch_2": [1,  "CB_dir",     "forward",    86, 45], 
                        "switch_3": [2,  "CB_dir",     "forward",    86, 45],
                        "switch_4": [3,  "CB_dir",     "forward",    86, 45],
                        "switch_5": [4,  "CB_dir",     "forward",    86, 45]}):
    
    net=load_4bus_net(open_loop =False)
    # generate trip decisions
    trip_decisions,net_sc =doc.run_fault_scenario_doc(net,sc_line_idx, sc_location,relay_configuration, 
                                                tripping_time_auto=tripping_time_auto)

    # test the IT plot function working or not
    doc.create_I_t_plot(trip_decisions,switch_id)

    plt.close('all')


if __name__ == '__main__':
    pytest.main(['-x',__file__])
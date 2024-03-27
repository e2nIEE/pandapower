
# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import pytest
from unittest.mock import patch
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import logging as log

import pandapower as pp
import pandapower.shortcircuit as sc
from pandapower.protection import oc_relay_model as oc
from pandapower.protection.example_grids import *
from pandapower.protection.utility_functions import plot_tripped_grid, create_I_t_plot,create_sc_bus
logger = log.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False

try:
    import mplcursors
    MPLCURSORS_INSTALLED = True
except ImportError:
    MPLCURSORS_INSTALLED = False
    logger.info('could not import mplcursors, plotting test is not possible')


def test_oc_parameters_automated(sc_fraction=0.95, overload_factor=1.2, ct_current_factor=1.25,
                          safety_factor=1,inverse_overload_factor=1.2, pickup_current_manual=None):
    # This function test the automated configuration of oc parameters function

    net_dtoc=dtoc_relay_net(open_loop=True)

    # sanity chech of user defined input oc parameters
    assert type(net_dtoc) == pp.auxiliary.pandapowerNet, 'net should be pandapower network'
    assert 0<sc_fraction<1 , 'sc fraction should be between 0 and 1'
    assert 1<=overload_factor , 'overload should be between greater than or equal to 1'
    assert 1<=inverse_overload_factor, 'inverse overload should be between greater than or equal to 1'
    assert 0<ct_current_factor, 'ct current factor should be greater than 0'
    assert safety_factor<=1 , 'safety current factor should be less than or equal to 0.95'


    # DTOC parameter settings
    settings_DTOC = pd.DataFrame({'switch_id': [0],'line_id':[0],
                                'bus_id': [0],'relay_type':['DTOC'],
                                'curve_type':['Definite time curve'],
                                'I_g[kA]':[0.213],
                                'I_gg[kA]':[2.630155810722129],
                                't_g[s]':[1.4],'t_gg[s]':[0.07]})


    # get relay settings for DTOC relay based on imput parameters
    relay_settings_DTOC = oc.oc_parameters(net_dtoc,relay_type='DTOC', time_settings=[0.07,0.5, 0.3],
                           sc_fraction=sc_fraction, overload_factor=overload_factor,
                           ct_current_factor=ct_current_factor,
                              safety_factor=safety_factor,inverse_overload_factor=inverse_overload_factor)
    # Test DTOC parameter settings
    assert_frame_equal(relay_settings_DTOC.iloc[[0]], settings_DTOC, check_dtype=False),
    'DTOC parameters should be of given format and vale with the given format'


    net_idmt = idmt_relay_net(open_loop=True)

    # IDMT parameter settings
    settings_IDMT = pd.DataFrame({'switch_id': [0],'line_id':[0],
                                'bus_id': [0],'relay_type':['IDMT'],
                                'curve_type':['standard_inverse'],
                                'I_s[kA]':[0.1704],'tms[s]':[1],
                                'k':[0.14],'t_grade[s]':[2.0], 'alpha':0.02})

    # get relay settings for IDMT relay based on imput parameters
    relay_settings_IDMT = oc.oc_parameters(net_idmt,relay_type='IDMT', time_settings=[1, 0.5],
                           sc_fraction=sc_fraction, overload_factor=overload_factor,
                           ct_current_factor=ct_current_factor,safety_factor=safety_factor,
                           inverse_overload_factor=inverse_overload_factor,curve_type='standard_inverse')
    # Test IDMT parameter settings
    assert_frame_equal(relay_settings_IDMT.iloc[[0]], settings_IDMT, check_dtype=False),
    'IDMT parameters should be of given format and vale with the given format'


    net_idtoc = idtoc_relay_net(open_loop=True)

    # Test IDTOC parameter settings
    settings_IDTOC = pd.DataFrame({'switch_id': [0],'line_id':[0],
                                'bus_id': [0],'relay_type':['IDTOC'],
                                'curve_type':['standard_inverse'],'I_g[kA]':[0.213],
                                  'I_gg[kA]':[2.63016],'I_s[kA]':[0.1704],
                                  't_g[s]':[1.4],'t_gg[s]':[0.07],'tms[s]':[1],
                                'k':[0.14],'t_grade[s]':[2.0], 'alpha':0.02})

    # get relay settings for DTOC relay based on imput parameters
    relay_settings_IDTOC = oc.oc_parameters(
        net_idtoc,relay_type='IDTOC',curve_type='standard_inverse', time_settings=[0.07, 0.5, 0.3, 1, 0.5],
        sc_fraction=sc_fraction, overload_factor=overload_factor,ct_current_factor=ct_current_factor,
        safety_factor=safety_factor,inverse_overload_factor=inverse_overload_factor)

    assert_frame_equal(relay_settings_IDTOC.iloc[[0]], settings_IDTOC, check_dtype=False),
    'IDMT parameters should be of given format and vale with the given format'


def test_oc_parameters_manual(pickup_current_manual=pd.DataFrame({
        'switch_id': [0,1,2,3,4,5],
        'I_gg': [1.2,1.3,1.3,1.4,1.4,1.25],
        'I_g': [0.7,0.7,0.5,0.4,0.4,0.4]})):
  # This function test oc parameters by manual input configurations

    net = dtoc_relay_net(open_loop=True)

    # DTOC parameter settings
    settings_DTOC=pd.DataFrame({'switch_id': [0],'line_id':[0],
                                'bus_id': [0],'relay_type':['DTOC'],
                                'curve_type':['Definite time curve'],
                                'I_g[kA]':[0.7],
                                'I_gg[kA]':[1.2],
                                't_g[s]':[1.4],'t_gg[s]':[0.07]})

    # get relay settings for DTOC relay based on imput parameters
    relay_settings_DTOC = oc.oc_parameters(net,relay_type='DTOC', time_settings=[0.07,0.5, 0.3],
                           pickup_current_manual=pickup_current_manual)
    # Test DTOC parameter settings
    assert_frame_equal(relay_settings_DTOC.iloc[[0]], settings_DTOC, check_dtype=False),
    'DTOC parameters manual should be of given format and vale with the given input format pickup_current_manual'


def test_oc_get_measurement_at_relay_location(sc_line_id=3, sc_location=0.3,  settings = pd.DataFrame({'switch_id':[3]})):
    # This function test the measurement function at relay location

    net = dtoc_relay_net(open_loop=True)

    # create sc for given line and location
    net_sc = create_sc_bus(net, sc_line_id, sc_location)
    sc.calc_sc(net_sc, bus=max(net_sc.bus.index), branch_results=True)
    i_ka= oc.oc_get_measurement_at_relay_location(net_sc,settings=settings.iloc[0])

    assert np.isclose(i_ka, 2.410070123674386, rtol=0.001), 'I_ka at the given location should be within 0.001 kA relative tolerence with calculated value'


def test_time_grading_manual(time_settings= pd.DataFrame({'switch_id': [0,1,2,3,4,5],'t_g':[0.5,0.8,1.1,1.4,1.7,2],
                                                                         't_gg': [0.05,0.06,0.07,0.07,0.05,0.05]})):
    net=dtoc_relay_net(open_loop=True)

    assert len(net.switch[net.switch.closed == True]) == len(time_settings), 'Number of active switches in the network should be equal to the number of relay in relay trip times'

    assert time_settings.columns.values.tolist() == ['switch_id', 't_g', 't_gg'], 'Input relay times should have the given format and header name'

    # times_DTOC=oc.time_grading(net,time_settings=time_settings)

    # # check input switch time equal to in the times from tripping_time
    # assert time_settings.equals(times_DTOC), 'given relay time settings should be equal to the values in the tripping_time function'


def test_time_grading_automated(time_setting_DTOC=[0.07,0.5,0.3]):

    net=dtoc_relay_net(open_loop=True)

    times_DTOC=oc.time_grading(net,time_setting_DTOC)
    tripping_time_DTOC= pd.DataFrame({'switch_id': [0,1,2,3,4,5],
                                      't_g':[1.4,1.1,0.8,1.1,0.8,0.5],
                                      't_gg': [0.07,0.07,0.07,0.07,0.07,0.07]})

    assert tripping_time_DTOC.equals(times_DTOC), 'given relay time settings hould be equal from the tripping_time function'


def test_oc_get_trip_decision(i_ka1=2.808, i_ka2=0.808,i_ka3=0):

    net = dtoc_relay_net(open_loop=True)

    settings_DTOC = pd.DataFrame({'switch_id': [0],'line_id':[0],
                                  'bus_id': [0],'relay_type':['DTOC'],
                                  'curve_type':['Definite time curve'],
                                  'I_g[kA]':[0.5415],
                                  'I_gg[kA]':[2.72257],
                                  't_g[s]':[1.4],'t_gg[s]':[0.07]})

    # think how to change ika1, ika2, ika3

    inst_trip_decisions = oc.oc_get_trip_decision(net, settings_DTOC.iloc[0], i_ka1)

    assert  inst_trip_decisions['Trip type'] == "instantaneous" , 'i_ka >Igg  trip type should be instantaneous'
    assert  inst_trip_decisions['Trip time [s]'] == settings_DTOC.iloc[0]['t_gg[s]'], 'instantaneous trip have trip time equal to Tgg'


    backup_trip_decisions = oc.oc_get_trip_decision(net,settings_DTOC.iloc[0], i_ka2)
    assert  backup_trip_decisions['Trip type'] == "backup" , 'i_ka >Ig  trip type should be backup'
    assert  backup_trip_decisions['Trip time [s]'] == settings_DTOC.iloc[0]['t_g[s]'], 'backup trip have trip time equal to Tg'

    no_trip_trip_decisions = oc.oc_get_trip_decision(net, settings_DTOC.iloc[0], i_ka3)
    assert  no_trip_trip_decisions['Trip type'] == 'no trip' , 'i_ka <Ig  No trip'
    assert  no_trip_trip_decisions['Trip time [s]'] ==np.inf, 'If no trip, trip time should be infinity'


    # IDMT trip decisions

    settings_IDMT = pd.DataFrame({'switch_id': [0],'line_id':[0],
                                  'bus_id': [0],'relay_type':['IDMT'],
                                  'curve_type':['standard_inverse'],
                                  'I_s[kA]':[0.4332],'tms[s]':[1],
                                  'k':[0.14],'t_grade[s]':[2.5], 'alpha':0.02})

    # think how to change ika1, ika2, ika3

    inst_trip_decisions = oc.oc_get_trip_decision(net, settings_IDMT.iloc[0], i_ka1)
    assert  inst_trip_decisions['Trip type'] == "instantaneous" , 'i_ka >Is  trip type should be instantaneous'


    no_trip_trip_decisions = oc.oc_get_trip_decision(net, settings_IDMT.iloc[0], i_ka3)
    assert  no_trip_trip_decisions['Trip type'] == 'no trip' , 'i_ka <Ig  No trip'
    assert  no_trip_trip_decisions['Trip time [s]'] ==np.inf, 'If no trip, trip time should be infinity'


@patch("matplotlib.pyplot.show")
@pytest.mark.slow
@pytest.mark.skipif(not MPLCURSORS_INSTALLED, reason='mplcursors must be installed')
@pytest.mark.skipif(not MATPLOTLIB_INSTALLED, reason='matplotlib must be installed')
def test_plot_tripped_grid(mock_show, sc_line_id =0,sc_location =0.4,
                           settings_DTOC=pd.DataFrame({'switch_id': [0],'line_id':[0],
                           'bus_id': [0],'relay_type':['DTOC'],
                           'curve_type':['Definite time curve'],
                           'I_g[kA]':[0.5415],
                           'I_gg[kA]':[2.72257],
                           't_g[s]':[1.4],'t_gg[s]':[0.07]}) ):
    net=dtoc_relay_net(open_loop=True)
    # generate trip decisions
    trip_decisions, net_sc = oc.run_fault_scenario_oc(
        net, sc_line_id, sc_location, relay_settings=settings_DTOC)


    # test the tripped grid
    plot_tripped_grid(net_sc, trip_decisions,sc_location, plot_annotations=(True))
    plt.close('all')


@patch("matplotlib.pyplot.show")
@pytest.mark.slow
@pytest.mark.skipif(not MPLCURSORS_INSTALLED, reason='mplcursors must be installed')
@pytest.mark.skipif(not MATPLOTLIB_INSTALLED, reason='matplotlib must be installed')
def test_plot_create_I_t_plot(mock_show, sc_line_id=0 ,sc_location =0.4,
                              settings_DTOC=pd.DataFrame({'switch_id': [0],'line_id':[0],
                              'bus_id': [0],'relay_type':['DTOC'],
                              'curve_type':['Definite time curve'],
                              'I_g[kA]':[0.5415],
                              'I_gg[kA]':[2.72257],
                              't_g[s]':[1.4],'t_gg[s]':[0.07]}) ):
    net = dtoc_relay_net(open_loop=True)
    # generate trip decisions
    trip_decisions, net_sc  = oc.run_fault_scenario_oc(
        net, sc_line_id, sc_location, relay_settings=settings_DTOC)

    # test the IT plot function working or not
    create_I_t_plot(trip_decisions,switch_id=[0])

    plt.close('all')


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])

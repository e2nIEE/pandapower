# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import copy
import tempfile

import numpy as np
import pandas as pd
import pytest

import pandapower.control as ct
import pandapower as pp
import logging
from pandapower.control import ConstControl
from pandapower.test.timeseries.test_timeseries import create_data_source, simple_test_net
from pandapower.timeseries import DFData, OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries

logger = logging.getLogger(__name__)
ow_logger = logging.getLogger("hp.control.output_writer")
logger.setLevel(logging.ERROR)
ow_logger.setLevel(logging.CRITICAL)


def test_output_writer_log():
    net = simple_test_net()

    # timeseries data
    df = pd.DataFrame([[15, 30, 2], [12, 27, 1.5], [7, 29, 2.1]])
    ds = DFData(df)

    # Create gen controller with datasource
    ct.ConstControl(net, element="load", variable="p_mw", element_index=[0, 2], data_source=ds,
                    profile_name=[0, 2])

    # Create, add output and set outputwriter
    ow = OutputWriter(net, output_path=tempfile.gettempdir())
    orig_index = [0, 1]
    ow.log_variable("res_bus", "vm_pu", orig_index)
    ow.log_variable("res_sgen", "p_mw")
    ow.log_variable("res_sgen", "q_mvar")

    # Run timeseries
    run_timeseries(net, time_steps=range(2), output_writer=ow)


    # --- double logged variables handling
    ow2 = copy.deepcopy(ow)
    new_idx = 2
    ow2.log_variable("res_bus", "vm_pu", new_idx, eval_name="test")
    run_timeseries(net, time_steps=range(2), output_writer=ow2)
    assert all(ow2.output["res_bus.vm_pu"].columns == orig_index + [new_idx])

    ow3 = copy.deepcopy(ow)
    new_idx = [2, 3]
    ow3.log_variable("res_bus", "vm_pu", new_idx)
    run_timeseries(net, time_steps=range(2), output_writer=ow3)
    assert all(ow3.output["res_bus.vm_pu"].columns == orig_index + new_idx)

    ow4 = copy.deepcopy(ow)
    new_idx = [2, 4]
    ow4.log_variable("res_bus", "vm_pu", new_idx, eval_name=["test1", "test2"])
    run_timeseries(net, time_steps=range(2), output_writer=ow4)
    assert all(ow4.output["res_bus.vm_pu"].columns == orig_index + new_idx)


def test_output_writer_with_timesteps_set():
    net = simple_test_net()

    n_timesteps = 10
    profiles, ds = create_data_source(n_timesteps)
    # 1load
    ConstControl(net, element='load', variable='p_mw', element_index=[0, 1, 2],
                 data_source=ds, profile_name=["load1", "load2_mv_p", "load3_hv_p"])

    time_steps = range(0, n_timesteps)
    ow = OutputWriter(net, time_steps, output_path=tempfile.gettempdir(), output_file_type=".json")
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'i_ka')
    run_timeseries(net, time_steps, output_writer=ow)
    assert len(ow.output["res_bus.vm_pu"]) == n_timesteps
    assert len(ow.output["res_line.i_ka"]) == n_timesteps


def test_output_writer_without_timesteps_set():
    net = simple_test_net()

    n_timesteps = 10
    profiles, ds = create_data_source(n_timesteps)
    # 1load
    ConstControl(net, element='load', variable='p_mw', element_index=[0, 1, 2],
                 data_source=ds, profile_name=["load1", "load2_mv_p", "load3_hv_p"])

    time_steps = range(0, n_timesteps)
    ow = OutputWriter(net, output_path=tempfile.gettempdir(), output_file_type=".json")
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'i_ka')
    run_timeseries(net, time_steps, output_writer=ow)
    assert len(ow.output["res_bus.vm_pu"]) == n_timesteps
    assert len(ow.output["res_line.i_ka"]) == n_timesteps


def test_output_writer_without_timesteps_set_repeat():
    # the same outputwriter should be able to run repeated time series
    net = simple_test_net()

    time_steps_to_check = [8, 5, 10]
    profiles, ds = create_data_source(max(time_steps_to_check))
    # 1load
    ConstControl(net, element='load', variable='p_mw', element_index=[0, 1, 2],
                 data_source=ds, profile_name=["load1", "load2_mv_p", "load3_hv_p"])

    ow = OutputWriter(net, output_path=tempfile.gettempdir(), output_file_type=".json")
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'i_ka')

    for n_timesteps in time_steps_to_check:
        time_steps = range(0, n_timesteps)
        run_timeseries(net, time_steps, output_writer=ow)
        assert len(ow.output["res_bus.vm_pu"].index) == n_timesteps


def test_output_writer_short_data_source():
    # outputwriter should fail if data source is shorter than time steps
    net = simple_test_net()

    n_timesteps = 10
    profiles, ds = create_data_source(5)
    # 1load
    ConstControl(net, element='load', variable='p_mw', element_index=[0, 1, 2],
                 data_source=ds, profile_name=["load1", "load2_mv_p", "load3_hv_p"])

    time_steps = range(0, n_timesteps)
    ow = OutputWriter(net, output_path=tempfile.gettempdir(), output_file_type=".json")
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'i_ka')

    with pytest.raises(KeyError):
        run_timeseries(net, time_steps, output_writer=ow)


def test_default_output_writer():
    net = simple_test_net()

    n_timesteps = 10
    profiles, ds = create_data_source(n_timesteps)
    ConstControl(net, element='load', variable='p_mw', element_index=[0, 1, 2],
                 data_source=ds, profile_name=["load1", "load2_mv_p", "load3_hv_p"])

    time_steps = range(0, n_timesteps)
    run_timeseries(net, time_steps)

    pass


def test_output_writer_eval_simple():
    net = simple_test_net()
    n_timesteps = 1
    profiles, ds = create_data_source(n_timesteps)
    # 1load
    ConstControl(net, element='load', variable='p_mw', element_index=[0, 1, 2],
                 data_source=ds, profile_name=["load1", "load2_mv_p", "load3_hv_p"])
    time_steps = range(0, n_timesteps)
    ow = OutputWriter(net, time_steps, output_path=tempfile.gettempdir(), output_file_type=".json")
    ow.log_variable('res_bus', 'vm_pu', eval_function=max, eval_name="max")
    run_timeseries(net, time_steps, output_writer=ow)
    assert len(ow.output["res_bus.vm_pu"]["max"]) == n_timesteps


def test_output_writer_multiple_index_definition():
    net = simple_test_net()

    n_timesteps = 1
    profiles, ds = create_data_source(n_timesteps)
    ConstControl(net, element='load', variable='p_mw', element_index=[0, 1, 2],
                 data_source=ds, profile_name=["load1", "load2_mv_p", "load3_hv_p"])
    time_steps = range(0, n_timesteps)

    ow = OutputWriter(net, time_steps, output_path=tempfile.gettempdir(), output_file_type=".json")
    ow.log_variable('res_bus', 'vm_pu', net.load.bus[[0, 1]])
    ow.log_variable('res_bus', 'vm_pu', index=[1, 2])
    ow.log_variable('res_bus', 'vm_pu', index=[3, 2, 1])
    ow.log_variable('res_bus', 'vm_pu', net.load.bus)
    ow.log_variable('res_bus', 'vm_pu', index=[3, 4])
    ow.log_variable('res_bus', 'vm_pu', net.bus.index)
    ow.log_variable('res_bus', 'vm_pu', 0)

    ow2 = OutputWriter(net, time_steps, output_path=tempfile.gettempdir(), output_file_type=".json")
    ow2.log_variable('res_bus', 'vm_pu', net.bus.index)

    run_timeseries(net, time_steps, output_writer=ow)
    run_timeseries(net, time_steps, output_writer=ow2)
    # assert saving works
    ow.save_results(0, True, True)
    # assert all are considered
    assert len(ow.output["res_bus.vm_pu"].columns) == len(net.bus.index)
    # assert correct order of values
    assert np.allclose(ow2.output["res_bus.vm_pu"].loc[:, net.bus.index],
                       ow.output["res_bus.vm_pu"].loc[:, net.bus.index], atol=0, rtol=0)




if __name__ == '__main__':
    pytest.main(['-s', __file__])

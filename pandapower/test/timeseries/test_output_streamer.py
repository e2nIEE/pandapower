# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import copy
import logging
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from pandapower.control import ConstControl
from pandapower.create import create_pwl_cost
from pandapower.file_io import from_json, to_json
from pandapower.networks import simple_four_bus_system
from pandapower.networks.power_system_test_cases import case5
from pandapower.run import runpp
from pandapower.test.timeseries.test_timeseries import create_data_source, simple_test_net
from pandapower.timeseries.data_sources.frame_data import DFData
from pandapower.timeseries.output_streamer import OutputStreamer
from pandapower.timeseries.run_time_series import run_timeseries

logger = logging.getLogger(__name__)
ow_logger = logging.getLogger("hp.control.output_writer")
logger.setLevel(logging.ERROR)
ow_logger.setLevel(logging.CRITICAL)


def test_output_streamer_log(simple_test_net):
    net = simple_test_net

    # timeseries data
    df = pd.DataFrame([[15, 30, 2], [12, 27, 1.5], [7, 29, 2.1]])
    ds = DFData(df)

    # Create gen controller with datasource
    ConstControl(net, element="load", variable="p_mw", element_index=[0, 2], data_source=ds, profile_name=[0, 2])

    # Create, add output and set OutputStreamer
    output_streamer = OutputStreamer(net, output_path=tempfile.gettempdir())
    output_streamer.remove_log_variable("res_bus")
    orig_index = [0, 1]
    output_streamer.log_variable("res_bus", "vm_pu", orig_index)
    output_streamer.log_variable("res_sgen", "p_mw")
    output_streamer.log_variable("res_sgen", "q_mvar")

    # Run timeseries
    run_timeseries(net, time_steps=range(2), verbose=False)

    # --- double logged variables handling
    output_streamer2 = copy.deepcopy(output_streamer)
    new_idx = 2
    output_streamer2.log_variable("res_bus", "vm_pu", new_idx, eval_name="test")
    run_timeseries(net, time_steps=range(2), output_writer=output_streamer2, verbose=False)
    assert all(output_streamer2.output["res_bus.vm_pu"].columns == orig_index + [new_idx])

    output_streamer4 = copy.deepcopy(output_streamer)
    new_idx = [2, 4]
    output_streamer4.log_variable("res_bus", "vm_pu", new_idx, eval_name=["test1", "test2"])
    run_timeseries(net, time_steps=range(2), output_writer=output_streamer4, verbose=False)
    assert all(output_streamer4.output["res_bus.vm_pu"].columns == orig_index + new_idx)


def test_output_streamer_with_timesteps_set(simple_test_net):
    net = simple_test_net

    n_timesteps = 10
    _, ds = create_data_source(n_timesteps)
    # 1load
    ConstControl(
        net,
        element="load",
        variable="p_mw",
        element_index=[0, 1, 2],
        data_source=ds,
        profile_name=["load1", "load2_mv_p", "load3_hv_p"],
    )

    time_steps = range(0, n_timesteps)
    output_streamer = OutputStreamer(net, time_steps, output_path=tempfile.gettempdir(), output_file_type=".json")
    output_streamer.log_variable("res_bus", "vm_pu")
    output_streamer.log_variable("res_line", "i_ka")
    run_timeseries(net, time_steps, verbose=False)
    assert len(output_streamer.output["res_bus.vm_pu"]) == n_timesteps
    assert len(output_streamer.output["res_line.i_ka"]) == n_timesteps


def test_output_streamer_without_timesteps_set(simple_test_net):
    net = simple_test_net
    n_timesteps = 5
    _, ds = create_data_source(n_timesteps)
    # 1load
    ConstControl(
        net,
        element="load",
        variable="p_mw",
        element_index=[0, 1, 2],
        data_source=ds,
        profile_name=["load1", "load2_mv_p", "load3_hv_p"],
    )

    time_steps = range(0, n_timesteps)
    output_streamer = OutputStreamer(net, output_path=tempfile.gettempdir(), output_file_type=".json")
    output_streamer.log_variable("res_bus", "vm_pu")
    output_streamer.log_variable("res_line", "i_ka")
    run_timeseries(net, time_steps, verbose=False)
    assert len(output_streamer.output["res_bus.vm_pu"]) == n_timesteps
    assert len(output_streamer.output["res_line.i_ka"]) == n_timesteps


def test_output_streamer_without_timesteps_set_repeat(simple_test_net):
    net = simple_test_net
    # the same OutputStreamer should be able to run repeated time series

    time_steps_to_check = [8, 5, 10]
    _, ds = create_data_source(max(time_steps_to_check))
    # 1load
    ConstControl(
        net,
        element="load",
        variable="p_mw",
        element_index=[0, 1, 2],
        data_source=ds,
        profile_name=["load1", "load2_mv_p", "load3_hv_p"],
    )

    output_streamer = OutputStreamer(net, output_path=tempfile.gettempdir(), output_file_type=".json")
    output_streamer.log_variable("res_bus", "vm_pu")
    output_streamer.log_variable("res_line", "i_ka")

    for n_timesteps in time_steps_to_check:
        time_steps = range(0, n_timesteps)
        run_timeseries(net, time_steps, verbose=False)
        assert len(output_streamer.output["res_bus.vm_pu"].index) == n_timesteps


def test_output_streamer_short_data_source(simple_test_net):
    net = simple_test_net
    # OutputStreamer should fail if data source is shorter than time steps

    n_timesteps = 10
    _, ds = create_data_source(5)
    # 1load
    ConstControl(
        net,
        element="load",
        variable="p_mw",
        element_index=[0, 1, 2],
        data_source=ds,
        profile_name=["load1", "load2_mv_p", "load3_hv_p"],
    )

    time_steps = range(0, n_timesteps)
    output_streamer = OutputStreamer(net, output_path=tempfile.gettempdir(), output_file_type=".json")
    output_streamer.log_variable("res_bus", "vm_pu")
    output_streamer.log_variable("res_line", "i_ka")

    with pytest.raises(KeyError):
        run_timeseries(net, time_steps, verbose=False)


def test_default_output_streamer(simple_test_net):
    net = simple_test_net

    n_timesteps = 5
    _, ds = create_data_source(n_timesteps)
    ConstControl(
        net,
        element="load",
        variable="p_mw",
        element_index=[0, 1, 2],
        data_source=ds,
        profile_name=["load1", "load2_mv_p", "load3_hv_p"],
    )

    time_steps = range(0, n_timesteps)
    run_timeseries(net, time_steps, verbose=False)
    output_streamer = net.output_writer.iloc[0, 0]
    loading_percent = output_streamer.output["res_line.loading_percent"]
    vm_pu = output_streamer.output["res_bus.vm_pu"]
    assert loading_percent.shape[0] == n_timesteps and loading_percent.shape[1] == len(net.line)
    assert vm_pu.shape[0] == n_timesteps and vm_pu.shape[1] == len(net.bus)


def test_output_streamer_eval_simple(simple_test_net):
    net = simple_test_net

    n_timesteps = 1
    _, ds = create_data_source(n_timesteps)
    # 1load
    ConstControl(
        net,
        element="load",
        variable="p_mw",
        element_index=[0, 1, 2],
        data_source=ds,
        profile_name=["load1", "load2_mv_p", "load3_hv_p"],
    )
    time_steps = range(0, n_timesteps)
    output_streamer = OutputStreamer(net, time_steps, output_path=tempfile.gettempdir(), output_file_type=".json")
    output_streamer.log_variable("res_bus", "vm_pu", eval_function=max, eval_name="max")
    run_timeseries(net, time_steps, verbose=False)
    assert len(output_streamer.output["res_bus.vm_pu"]["max"]) == n_timesteps


def test_output_streamer_multiple_index_definition(simple_test_net):
    net = simple_test_net

    n_timesteps = 1
    _, ds = create_data_source(n_timesteps)
    ConstControl(
        net,
        element="load",
        variable="p_mw",
        element_index=[0, 1, 2],
        data_source=ds,
        profile_name=["load1", "load2_mv_p", "load3_hv_p"],
    )
    time_steps = range(0, n_timesteps)

    output_streamer = OutputStreamer(net, time_steps, output_path=tempfile.gettempdir(), output_file_type=".json")
    output_streamer.log_variable("res_bus", "vm_pu", net.load.bus[[0, 1]])
    output_streamer.log_variable("res_bus", "vm_pu", index=[1, 2])
    output_streamer.log_variable("res_bus", "vm_pu", index=[3, 2, 1])
    output_streamer.log_variable("res_bus", "vm_pu", net.load.bus)
    output_streamer.log_variable("res_bus", "vm_pu", index=[3, 4])
    output_streamer.log_variable("res_bus", "vm_pu", net.bus.index)
    output_streamer.log_variable("res_bus", "vm_pu", 0)
    run_timeseries(net, time_steps, verbose=False)
    backup_result = copy.deepcopy(output_streamer.output["res_bus.vm_pu"].loc[:, net.bus.index])
    del output_streamer

    output_streamer = OutputStreamer(net, time_steps, output_path=tempfile.gettempdir(), output_file_type=".json")
    output_streamer.log_variable("res_bus", "vm_pu", net.bus.index)

    run_timeseries(net, time_steps, verbose=False)
    # assert all are considered
    assert len(output_streamer.output["res_bus.vm_pu"].columns) == len(net.bus.index)
    # assert correct order of values
    assert np.allclose(backup_result, output_streamer.output["res_bus.vm_pu"].loc[:, net.bus.index])


def test_remove_variable(simple_test_net):
    net = simple_test_net

    output_streamer = OutputStreamer(net)
    # test printing
    logger.info(output_streamer)
    assert len(output_streamer.log_variables) == 2
    assert output_streamer.log_variables[0][0] == "res_bus" and output_streamer.log_variables[0][1] == "vm_pu"
    assert (
        output_streamer.log_variables[1][0] == "res_line" and output_streamer.log_variables[1][1] == "loading_percent"
    )
    output_streamer.remove_log_variable("res_bus")
    assert (
        output_streamer.log_variables[0][0] == "res_line" and output_streamer.log_variables[0][1] == "loading_percent"
    )
    output_streamer.remove_log_variable("res_line", "loading_percent")
    assert len(output_streamer.log_variables) == 0


def test_store_and_load(simple_test_net):
    net = simple_test_net

    n_timesteps = 2
    _, ds = create_data_source(n_timesteps)
    ConstControl(
        net,
        element="load",
        variable="p_mw",
        element_index=[0, 1, 2],
        data_source=ds,
        profile_name=["load1", "load2_mv_p", "load3_hv_p"],
    )
    dirname = tempfile.gettempdir()
    output_streamer = OutputStreamer(net, output_path=dirname, output_file_type=".json")
    output_streamer.remove_log_variable("res_bus")
    tmp_file = os.path.join(dirname, "net.json")
    to_json(net, tmp_file)
    del net
    del output_streamer
    res_line_file = os.path.join(dirname, "res_line", "loading_percent.json")
    # del result file is one is present
    if os.path.isfile(res_line_file):
        os.remove(res_line_file)
    net = from_json(tmp_file)
    output_streamer = net.output_writer.iat[0, 0]
    assert len(output_streamer.log_variables) == 1
    assert output_streamer.output_path == dirname
    time_steps = range(0, n_timesteps)
    run_timeseries(net, time_steps=time_steps, verbose=False)
    # check if results were written
    assert os.path.isfile(res_line_file)


def test_ppc_log(simple_test_net):
    net = simple_test_net
    n_timesteps = 5
    _, ds = create_data_source(n_timesteps)
    # 1load
    ConstControl(
        net,
        element="load",
        variable="p_mw",
        element_index=[0, 1, 2],
        data_source=ds,
        profile_name=["load1", "load2_mv_p", "load3_hv_p"],
        recycle=True,
    )

    time_steps = range(0, n_timesteps)
    output_streamer = OutputStreamer(net, output_path=tempfile.gettempdir(), output_file_type=".json", log_variables=[])
    output_streamer.log_variable("ppc_bus", "vm")
    output_streamer.log_variable("ppc_bus", "va")
    runpp(net, only_v_results=True, recycle={"bus_pq": True, "gen": False, "trafo": False})
    run_timeseries(
        net, time_steps, recycle={"bus_pq": True, "gen": False, "trafo": False}, only_v_results=True, verbose=False
    )
    assert len(output_streamer.output["ppc_bus.vm"]) == n_timesteps
    assert len(output_streamer.output["ppc_bus.va"]) == n_timesteps


def test_output_streamer_index():
    net = simple_four_bus_system()
    steps = [3, 5, 7]
    p_data = pd.DataFrame(
        index=steps,
        columns=["0", "1"],
        data=[
            [0.01, 0.02],
            [0.03, 0.04],
            [0.05, 0.06],
        ],
    )
    v_data = pd.DataFrame(index=steps, columns=["0"], data=[1.01, 1.03, 1.02])

    ds_p = DFData(p_data)
    ds_v = DFData(v_data)

    ConstControl(
        net,
        element="load",
        variable="p_mw",
        element_index=net.load.index.tolist(),
        data_source=ds_p,
        profile_name=p_data.columns,
    )
    ConstControl(net, element="ext_grid", variable="vm_pu", element_index=0, data_source=ds_v, profile_name="0")

    output_streamer = OutputStreamer(net)
    output_streamer.log_variable("res_bus", "vm_pu")
    output_streamer.log_variable("res_line", "loading_percent")

    run_timeseries(net, time_steps=p_data.index, verbose=False)

    assert np.all(output_streamer.output["res_line.loading_percent"].index == p_data.index)


def test_output_streamer_save_interval(simple_test_net, tmp_path):
    net = simple_test_net

    n_timesteps = 6
    _, ds = create_data_source(n_timesteps)

    ConstControl(
        net,
        element="load",
        variable="p_mw",
        element_index=[0, 1, 2],
        data_source=ds,
        profile_name=["load1", "load2_mv_p", "load3_hv_p"],
    )

    output_streamer = OutputStreamer(
        net,
        output_path=str(tmp_path),
        output_file_type=".csv",
        save_interval=2,
    )

    output_streamer.log_variable("res_bus", "vm_pu")

    time_steps = range(n_timesteps)
    run_timeseries(net, time_steps, verbose=False)

    result_file = tmp_path / "res_bus" / "vm_pu.csv"

    assert result_file.exists()

    result = pd.read_csv(result_file, sep=";")

    # Alle 6 Zeitschritte müssen am Ende vorhanden sein
    assert len(result) == n_timesteps


def test_get_data_since_last_save(simple_test_net):
    net = simple_test_net

    output_streamer = OutputStreamer(
        net,
        time_steps=range(10),
        save_interval=3,
    )

    data = pd.DataFrame(
        {"value": range(10)},
        index=range(10),
    )

    # Simuliere ersten Save bei timestep 3
    output_streamer.time_step = 3
    output_streamer.last_time_step = 0

    result = output_streamer._get_data_since_last_save(data)

    assert list(result["value"]) == [0, 1, 2]


def test_get_data_since_last_save_second_chunk(simple_test_net):
    net = simple_test_net

    output_streamer = OutputStreamer(
        net,
        time_steps=range(10),
        save_interval=3,
    )

    data = pd.DataFrame(
        {"value": range(10)},
        index=range(10),
    )

    output_streamer.time_step = 6
    output_streamer.last_time_step = 3

    result = output_streamer._get_data_since_last_save(data)

    assert list(result["value"]) == [3, 4, 5]


def test_get_data_since_last_save_last_chunk(simple_test_net):
    net = simple_test_net

    output_streamer = OutputStreamer(
        net,
        time_steps=range(10),
        save_interval=3,
    )

    data = pd.DataFrame(
        {"value": range(10)},
        index=range(10),
    )

    output_streamer.time_step = 9
    output_streamer.last_time_step = 9

    result = output_streamer._get_data_since_last_save(data)

    assert list(result["value"]) == [9]


def test_output_streamer_save_interval_zero(simple_test_net, tmp_path):
    net = simple_test_net

    output_streamer = OutputStreamer(
        net,
        output_path=str(tmp_path),
        output_file_type=".csv",
        save_interval=0,
    )

    output_streamer.log_variable("res_bus", "vm_pu")

    _, ds = create_data_source(3)

    ConstControl(
        net,
        element="load",
        variable="p_mw",
        element_index=[0, 1, 2],
        data_source=ds,
        profile_name=["load1", "load2_mv_p", "load3_hv_p"],
    )

    run_timeseries(
        net,
        time_steps=range(3),
        verbose=False,
    )

    result_file = tmp_path / "res_bus" / "vm_pu.csv"

    assert result_file.exists()

    result = pd.read_csv(result_file, sep=";")

    assert len(result) == 3


def test_equal_eval_name_warning_and_costs():
    net = case5()
    net.poly_cost = net.poly_cost.iloc[0:0]
    create_pwl_cost(net, 0, "sgen", [[0, 20, 1], [20, 30, 2]])
    create_pwl_cost(net, 0, "gen", [[0, 20, 1], [20, 30, 2]])
    create_pwl_cost(net, 1, "gen", [[0, 20, 1], [20, 30, 2]])
    create_pwl_cost(net, 2, "gen", [[0, 20, 1], [20, 30, 2]])
    df = pd.DataFrame({0: [200, 300, 400, 500], 1: [400, 300, 100, 50], 2: [100, 300, 200, 100]})
    ds = DFData(df.astype(np.float64))
    _ = ConstControl(net, "load", "p_mw", net.load.index, profile_name=net.load.index, data_source=ds)
    output_streamer = OutputStreamer(net, output_path=None)
    output_streamer.log_variable("res_sgen", "p_mw", None, np.max, "warnme")
    output_streamer.log_variable("res_load", "p_mw", None, np.max, "warnme")
    output_streamer.log_variable("pwl_cost", "points", eval_function=cost_logging)

    output_streamer.remove_log_variable("res_bus", "vm_pu")
    output_streamer.remove_log_variable("res_line", "loading_percent")
    run_timeseries(net, verbose=False)

    p_sgen = output_streamer.output["res_sgen.p_mw"]
    p_load = output_streamer.output["res_load.p_mw"]
    cost = output_streamer.output["pwl_cost.points"]
    assert not np.all(p_sgen.values == p_load.values)
    assert cost.shape == (4, 4)
    assert len(output_streamer.np_results) == 3


def cost_logging(result, n_columns=4):
    return np.array([result[i][0][2] for i in range(len(result))])


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

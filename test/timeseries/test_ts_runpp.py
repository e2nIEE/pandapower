# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.

import copy
import tempfile

import numpy as np
import pytest

from pandapower.control import Characteristic, CharacteristicControl, ConstControl, ContinuousTapControl
from pandapower.run import runpp
from pandapower.timeseries import OutputWriter
from pandapower.timeseries.run_time_series import run_timeseries
from pandapower.timeseries.ts_runpp import TimeSeriesRunpp
from test.timeseries.test_timeseries import create_data_source, simple_test_net  # noqa: F401

n_timesteps = 5
time_steps = range(n_timesteps)

# logged variables that are written by both the recycled and the regular time series run
log_variables = [
    ("res_bus", "vm_pu"),
    ("res_bus", "va_degree"),
    ("res_line", "loading_percent"),
    ("res_trafo", "loading_percent"),
    ("res_trafo3w", "loading_percent"),
    ("res_trafo3w", "p_hv_mw"),
    ("res_ext_grid", "p_mw"),
    ("res_load", "p_mw"),
]


def add_const(net, ds, recycle):
    return ConstControl(
        net,
        element="load",
        variable="p_mw",
        element_index=[0, 1, 2],
        data_source=ds,
        profile_name=["load1", "load2_mv_p", "load3_hv_p"],
        recycle=recycle,
    )


def create_output_writer(net):
    ow = OutputWriter(net, time_steps, output_path=tempfile.gettempdir(), output_file_type=".json")
    for table, variable in log_variables:
        ow.log_variable(table, variable)
    return ow


def clear_controllers(net):
    """Remove controllers and output writers so a fresh time series setup can be created."""
    net.controller = net.controller.drop(index=net.controller.index)
    if "output_writer" in net:
        net.output_writer = net.output_writer.drop(index=net.output_writer.index)
    return net


def run_ts_runpp(net, ds, setup=None):
    net_ts = clear_controllers(copy.deepcopy(net))
    add_const(net_ts, ds, recycle=True)
    if setup is not None:
        setup(net_ts, recycle=True)
    ow = create_output_writer(net_ts)
    ts_runpp = TimeSeriesRunpp(net_ts)
    run_timeseries(net_ts, time_steps, run=ts_runpp.ts_runpp, verbose=False)
    return ow, ts_runpp


def run_reference(net, ds, setup=None):
    net_ref = clear_controllers(copy.deepcopy(net))
    add_const(net_ref, ds, recycle=False)
    if setup is not None:
        setup(net_ref, recycle=False)
    ow = create_output_writer(net_ref)
    run_timeseries(net_ref, time_steps, recycle=False, verbose=False)
    return ow


def assert_same_results(ow_ts, ow_ref):
    for table, variable in log_variables:
        key = f"{table}.{variable}"
        assert np.allclose(ow_ts.output[key], ow_ref.output[key], rtol=0, atol=1e-6), (
            f"ts_runpp results differ from runpp for {key}"
        )


def test_update_flags_const_pq(simple_test_net):
    net = simple_test_net
    _, ds = create_data_source(n_timesteps)
    net = clear_controllers(net)
    add_const(net, ds, recycle=True)
    create_output_writer(net)

    ts_runpp = TimeSeriesRunpp(net)
    assert ts_runpp.update_pq
    assert not ts_runpp.update_trafo


def test_update_flags_trafo_tap(simple_test_net):
    net = clear_controllers(simple_test_net)
    add_const(net, create_data_source(n_timesteps)[1], recycle=True)
    ContinuousTapControl(net, 0, 1.005, tol=1e-4, recycle=True)
    create_output_writer(net)

    ts_runpp = TimeSeriesRunpp(net)
    assert ts_runpp.update_pq
    assert ts_runpp.update_trafo


def test_const_pq(simple_test_net):
    _, ds = create_data_source(n_timesteps)

    ow_ts, _ = run_ts_runpp(simple_test_net, ds)
    ow_ref = run_reference(simple_test_net, ds)

    assert_same_results(ow_ts, ow_ref)


def test_trafo_tap(simple_test_net):
    net = copy.deepcopy(simple_test_net)
    net.trafo.loc[0, "tap_pos"] = 0
    _, ds = create_data_source(n_timesteps)

    def setup(net_to_setup, recycle):
        ContinuousTapControl(net_to_setup, 0, 1.005, tol=1e-4, recycle=recycle)

    ow_ts, ts_runpp = run_ts_runpp(net, ds, setup=setup)
    assert ts_runpp.update_trafo
    ow_ref = run_reference(net, ds, setup=setup)

    assert_same_results(ow_ts, ow_ref)


def test_out_of_service(simple_test_net):
    net = copy.deepcopy(simple_test_net)
    _, ds = create_data_source(n_timesteps)
    net.load.loc[0, "in_service"] = False

    ow_ts, _ = run_ts_runpp(net, ds)
    ow_ref = run_reference(net, ds)

    assert_same_results(ow_ts, ow_ref)


def test_unsupported_controller_raises(simple_test_net):
    net = clear_controllers(simple_test_net)
    create_output_writer(net)
    characteristic = Characteristic(net, x_values=[0, 1], y_values=[0, 1])
    CharacteristicControl(
        net,
        output_element="load",
        output_variable="p_mw",
        output_element_index=0,
        input_element="res_load",
        input_variable="p_mw",
        input_element_index=0,
        characteristic_index=characteristic.index,
    )

    with pytest.raises(TypeError, match="controller class not supported for recycle"):
        TimeSeriesRunpp(net)


def test_cleanup_resets_newton_variables(simple_test_net):
    net = clear_controllers(simple_test_net)
    _, ds = create_data_source(n_timesteps)
    add_const(net, ds, recycle=True)
    create_output_writer(net)

    ts_runpp = TimeSeriesRunpp(net)
    assert ts_runpp.ppc is not None
    assert ts_runpp.Ybus is not None

    ts_runpp.cleanup()

    for attribute in ("baseMVA", "ref", "pv", "pq", "V", "ppc", "ppci", "Ybus", "Yf", "Yt", "Ibus"):
        assert getattr(ts_runpp, attribute) is None, f"{attribute} was not reset by cleanup()"

    # the net must still be usable with a regular power flow afterwards
    runpp(net)
    assert net.converged


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

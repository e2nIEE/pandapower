import copy
import itertools
import tempfile

import numpy as np
import pandas as pd
import pytest

from pandapower.control.controller.trafo.continuous_tap_control import (
    ContinuousTapControl,
)
from pandapower.create import (
    create_buses,
    create_gen,
    create_lines,
    create_transformer,
    create_transformer3w,
)
from pandapower.run import rundcpp, runpp
from pandapower.timeseries.data_sources.frame_data import DFData
from pandapower.timeseries.read_batch_results import (
    get_batch_line_results,
    get_batch_trafo3w_results,
    get_batch_trafo_results,
    polar_to_rad,
    v_to_i_s,
)
from test.timeseries.test_output_writer import (
    ConstControl,
    OutputWriter,
    create_data_source,
    run_timeseries,
)
from test.timeseries.test_timeseries import simple_test_net  # noqa: F401

n_timesteps = 5
time_steps = range(n_timesteps)


@pytest.fixture(params=[runpp, rundcpp])
def run_function(request):
    return request.param


def add_const(net, ds, recycle, element="load", variable="p_mw", element_index=None, profile_name=None):
    if element_index is None or profile_name is None:
        element_index = [0, 1, 2]
        profile_name = ["load1", "load2_mv_p", "load3_hv_p"]
    return ConstControl(
        net,
        element=element,
        variable=variable,
        element_index=element_index,
        data_source=ds,
        profile_name=profile_name,
        recycle=recycle,
    )


def test_batch_output_reader(simple_test_net):
    net = simple_test_net
    _, ds = create_data_source(n_timesteps)
    # 1load
    c = add_const(net, ds, recycle=True)

    ow = OutputWriter(net, output_path=tempfile.gettempdir(), output_file_type=".json", log_variables=[])
    ow.log_variable("ppc_bus", "vm")
    ow.log_variable("ppc_bus", "va")

    recycle = {"trafo": False, "gen": False, "bus_pq": True}
    only_v = True
    runpp(net, only_v_results=only_v, recycle=recycle)
    run_timeseries(net, time_steps, recycle=recycle, only_v_results=only_v, verbose=False)

    vm, va = ow.output["ppc_bus.vm"], ow.output["ppc_bus.va"]
    # ppc was removed (set to None) in cleanup() after the timeseries was over. now we have to remake the net._ppc by
    # running the power flow again
    runpp(net, only_v_results=only_v, recycle=recycle)
    _, s_abs, i_abs = v_to_i_s(net, vm, va)

    del net, ow, c
    net = simple_test_net
    net.output_writer = net.output_writer.drop(index=net.output_writer.index)
    net.controller = net.controller.drop(index=net.controller.index)
    add_const(net, ds, recycle=False)
    ow = OutputWriter(net, output_path=tempfile.gettempdir(), output_file_type=".json", log_variables=[])
    ow.log_variable("res_line", "i_from_ka")
    ow.log_variable("res_line", "loading_percent")
    ow.log_variable("res_line", "i_to_ka")
    ow.log_variable("res_line", "loading_percent")
    ow.log_variable("res_trafo", "loading_percent")
    ow.log_variable("res_trafo", "i_hv_ka")
    ow.log_variable("res_trafo", "i_lv_ka")
    ow.log_variable("res_trafo3w", "loading_percent")
    ow.log_variable("res_trafo3w", "i_hv_ka")
    ow.log_variable("res_trafo3w", "i_mv_ka")
    ow.log_variable("res_trafo3w", "i_lv_ka")
    ow.log_variable("res_bus", "vm_pu")
    ow.log_variable("res_bus", "va_degree")

    runpp(net)
    run_timeseries(net, time_steps, trafo_loading="current", verbose=False)

    vm_pu, va_degree = ow.output["res_bus.vm_pu"], ow.output["res_bus.va_degree"]
    i_from_ka_normal, i_to_ka_normal = ow.output["res_line.i_from_ka"], ow.output["res_line.i_to_ka"]
    i_hv_ka_normal, i_lv_ka_normal = ow.output["res_trafo.i_hv_ka"], ow.output["res_trafo.i_lv_ka"]
    i3_hv_ka, i3_lm_ka, i3_lv_ka = (
        ow.output["res_trafo3w.i_hv_ka"],
        ow.output["res_trafo3w.i_mv_ka"],
        ow.output["res_trafo3w.i_lv_ka"],
    )
    t3_loading_percent_normal = ow.output["res_trafo3w.loading_percent"]
    loading_percent_normal = ow.output["res_line.loading_percent"]
    t_loading_percent_normal = ow.output["res_trafo.loading_percent"]

    v_normal = polar_to_rad(vm_pu, va_degree)
    v_batch = polar_to_rad(vm, va)
    _, i_from_ka, i_to_ka, loading_percent = get_batch_line_results(net, i_abs)
    _, i_hv_ka, i_lv_ka, _, ld_trafo = get_batch_trafo_results(net, i_abs, s_abs)
    i_h, i_m, i_l, ld3_trafo = get_batch_trafo3w_results(net, i_abs, s_abs)

    # v batch contains aux buses of trafo3w. we can only compare non-aux bus voltages
    assert np.allclose(v_normal, v_batch[:, : v_normal.shape[1]])
    # line loading
    assert np.allclose(i_from_ka_normal, i_from_ka)
    assert np.allclose(i_to_ka_normal, i_to_ka)
    assert np.allclose(loading_percent_normal, loading_percent)
    # trafo
    assert np.allclose(i_hv_ka_normal, i_hv_ka)
    assert np.allclose(i_lv_ka_normal, i_lv_ka)
    assert np.allclose(t_loading_percent_normal, ld_trafo)
    # trafo3w
    assert np.allclose(i3_hv_ka, i_h)
    assert np.allclose(i3_lm_ka, i_m)
    assert np.allclose(i3_lv_ka, i_l)
    assert np.allclose(t3_loading_percent_normal, ld3_trafo)


def _v_var(run, full=True):
    # for runpp, vm_pu is important, and for rundcpp va_degree is important
    var = "vm_pu" if run.__name__ == "runpp" else "va_degree"
    if full:
        return f"res_bus.{var}"
    else:
        return var


def _run_recycle(net, run):
    # default log variables are res_bus.vm_pu, res_line.loading_percent
    log_variables = [
        ("res_bus", "vm_pu") if run.__name__ == "runpp" else ("res_bus", "va_degree"),
        ("res_line", "loading_percent"),
    ]
    ow = OutputWriter(net, output_path=tempfile.gettempdir(), output_file_type=".json", log_variables=log_variables)
    run_timeseries(net, time_steps, verbose=False, run=run)
    vm_pu = copy.deepcopy(ow.output[_v_var(run)])
    ll = copy.deepcopy(ow.output["res_line.loading_percent"])
    net.controller = net.controller.drop(index=net.controller.index)
    net.output_writer = net.output_writer.drop(index=net.output_writer.index)
    del ow
    return vm_pu, ll


def _run_normal(net, run, time_steps=time_steps):
    log_variables = [
        ("res_bus", "vm_pu") if run.__name__ == "runpp" else ("res_bus", "va_degree"),
        ("res_line", "loading_percent"),
    ]
    ow = OutputWriter(net, output_path=tempfile.gettempdir(), output_file_type=".json", log_variables=log_variables)
    ow.log_variable("res_bus", "va_degree")
    run_timeseries(net, time_steps, recycle=False, verbose=False, run=run)
    return ow


def test_const_pq(simple_test_net, run_function):
    # allows to use recycle = {"bus_pq"} and fast output read
    net = simple_test_net

    _, ds = create_data_source(n_timesteps)
    # 1load
    c = add_const(net, ds, recycle=None)
    vm_pu, ll = _run_recycle(net, run_function)
    del c

    # calculate the same results without recycle
    add_const(net, ds, recycle=False)
    ow = _run_normal(net, run_function)
    assert np.allclose(vm_pu, ow.output[_v_var(run_function)])
    assert np.allclose(ll, ow.output["res_line.loading_percent"])


def test_const_gen(simple_test_net, run_function):
    # allows to use recycle = {"gen"} and fast output read
    net = simple_test_net
    create_gen(net, 1, p_mw=2.0)
    profiles, _ = create_data_source(n_timesteps)
    profiles["gen"] = np.random.random(n_timesteps) * 2e1
    ds = DFData(profiles)
    # 1load
    c1 = add_const(net, ds, recycle=None)
    c2 = add_const(net, ds, recycle=None, profile_name="gen", element_index=0, element="gen")

    vm_pu, ll = _run_recycle(net, run_function)
    del c1, c2

    # calculate the same results without recycle
    add_const(net, ds, recycle=False)
    add_const(net, ds, recycle=None, profile_name="gen", element_index=0, element="gen")
    ow = _run_normal(net, run_function)
    assert np.allclose(vm_pu, ow.output[_v_var(run_function)])
    assert np.allclose(ll, ow.output["res_line.loading_percent"])


def test_const_ext_grid(simple_test_net, run_function):
    # allows to use recycle = {"gen"} and fast output read
    net = simple_test_net
    profiles, _ = create_data_source(n_timesteps)
    profiles["ext_grid"] = np.ones(n_timesteps) + np.arange(0, n_timesteps) * 1e-2

    ds = DFData(profiles)
    # 1load
    c1 = add_const(net, ds, recycle=None)
    c2 = add_const(
        net,
        ds,
        recycle=None,
        profile_name="ext_grid",
        variable=_v_var(run_function, False),
        element_index=0,
        element="ext_grid",
    )

    vm_pu, ll = _run_recycle(net, run_function)
    del c1, c2

    # calculate the same results without recycle
    add_const(net, ds, recycle=False)
    add_const(
        net,
        ds,
        recycle=False,
        profile_name="ext_grid",
        variable=_v_var(run_function, False),
        element_index=0,
        element="ext_grid",
    )
    ow = _run_normal(net, run_function)
    assert np.allclose(vm_pu, ow.output[_v_var(run_function)])
    assert np.allclose(ll, ow.output["res_line.loading_percent"])
    assert np.allclose(ow.output[_v_var(run_function)].values[:, 0], profiles["ext_grid"])
    assert np.allclose(vm_pu.values[:, 0], profiles["ext_grid"])


def test_trafo_tap(simple_test_net, run_function=runpp):
    # allows to use recycle = {"trafo"} but not fast output read
    net = simple_test_net
    _, ds = create_data_source(n_timesteps)

    # 1load
    c1 = add_const(net, ds, recycle=None)
    c2 = ContinuousTapControl(net, 0, 0.99, tol=1e-9)

    vm_pu, ll = _run_recycle(net, run_function)
    del c1, c2

    # calculate the same results without recycle
    add_const(net, ds, recycle=False)
    ContinuousTapControl(net, 0, 0.99, recycle=False, tol=1e-9)
    ow = _run_normal(net, run_function)
    assert np.allclose(vm_pu, ow.output[_v_var(run_function)])
    assert np.allclose(ll, ow.output["res_line.loading_percent"])


def test_trafo_tap_dc(simple_test_net, run_function=rundcpp):
    # allows to use recycle = {"trafo"} but not fast output read
    net = simple_test_net
    _, ds = create_data_source(n_timesteps)

    # 1load
    c1 = add_const(net, ds, recycle=None)
    c2 = add_const(
        net, ds, recycle=None, profile_name="trafo_tap", variable="tap_pos", element_index=0, element="trafo"
    )

    vm_pu, ll = _run_recycle(net, run_function)
    del c1, c2

    # calculate the same results without recycle
    add_const(net, ds, recycle=False)
    add_const(net, ds, recycle=None, profile_name="trafo_tap", variable="tap_pos", element_index=0, element="trafo")
    ow = _run_normal(net, run_function)
    assert np.allclose(vm_pu, ow.output[_v_var(run_function)])
    assert np.allclose(ll, ow.output["res_line.loading_percent"])


def test_const_pq_gen_trafo_tap(simple_test_net, run_function=runpp):
    # allows to use recycle = {"bus_pq", "gen", "trafo"}
    net = simple_test_net
    profiles, _ = create_data_source(n_timesteps)
    profiles["ext_grid"] = np.ones(n_timesteps) + np.arange(0, n_timesteps) * 1e-2
    ds = DFData(profiles)
    # 1load
    c1 = add_const(net, ds, recycle=None)
    c2 = add_const(
        net, ds, recycle=None, profile_name="ext_grid", variable="vm_pu", element_index=0, element="ext_grid"
    )
    c3 = ContinuousTapControl(net, 0, 1.01, tol=1e-9)

    vm_pu, ll = _run_recycle(net, run_function)
    del c1, c2, c3

    # calculate the same results without recycle
    add_const(net, ds, recycle=False)
    add_const(net, ds, recycle=False, profile_name="ext_grid", variable="vm_pu", element_index=0, element="ext_grid")
    ContinuousTapControl(net, 0, 1.01, recycle=False, tol=1e-9)
    ow = _run_normal(net, run_function)
    assert np.allclose(vm_pu, ow.output[_v_var(run_function)])
    assert np.allclose(ll, ow.output["res_line.loading_percent"])


def test_const_pq_gen_trafo_tap_dc(simple_test_net, run_function=rundcpp):
    # allows to use recycle = {"bus_pq", "gen", "trafo"}
    net = simple_test_net
    profiles, _ = create_data_source(n_timesteps)
    profiles["ext_grid"] = np.ones(n_timesteps) + np.arange(0, n_timesteps) * 1e-2
    ds = DFData(profiles)
    # 1load
    c1 = add_const(net, ds, recycle=None)
    c2 = add_const(
        net, ds, recycle=None, profile_name="ext_grid", variable="vm_pu", element_index=0, element="ext_grid"
    )
    c3 = add_const(
        net, ds, recycle=None, profile_name="trafo_tap", variable="tap_pos", element_index=0, element="trafo"
    )

    vm_pu, ll = _run_recycle(net, run_function)
    del c1, c2, c3

    # calculate the same results without recycle
    add_const(net, ds, recycle=False)
    add_const(net, ds, recycle=False, profile_name="ext_grid", variable="vm_pu", element_index=0, element="ext_grid")
    add_const(net, ds, recycle=None, profile_name="trafo_tap", variable="tap_pos", element_index=0, element="trafo")
    ow = _run_normal(net, run_function)
    assert np.allclose(vm_pu, ow.output[_v_var(run_function)], rtol=0, atol=1e-6)
    assert np.allclose(ll, ow.output["res_line.loading_percent"], rtol=0, atol=1e-6)


def test_const_pq_gen_trafo_tap_ideal(simple_test_net, run_function):
    # allows to use recycle = {"bus_pq", "gen", "trafo"}
    net = simple_test_net
    net.trafo.loc[0, ["tap_step_percent", "tap_step_degree", "tap_changer_type"]] = 0, 5, "Ideal"
    profiles, _ = create_data_source(n_timesteps)
    profiles["ext_grid"] = np.ones(n_timesteps) + np.arange(0, n_timesteps) * 1e-2
    ds = DFData(profiles)
    # 1load
    c1 = add_const(net, ds, recycle=None)
    c2 = add_const(
        net,
        ds,
        recycle=None,
        profile_name="ext_grid",
        variable=_v_var(run_function, False),
        element_index=0,
        element="ext_grid",
    )
    c3 = add_const(
        net, ds, recycle=None, profile_name="trafo_tap", variable="tap_pos", element_index=0, element="trafo"
    )

    vm_pu, ll = _run_recycle(net, run_function)
    del c1, c2, c3

    # calculate the same results without recycle
    add_const(net, ds, recycle=False)
    add_const(
        net,
        ds,
        recycle=False,
        profile_name="ext_grid",
        variable=_v_var(run_function, False),
        element_index=0,
        element="ext_grid",
    )
    add_const(net, ds, recycle=None, profile_name="trafo_tap", variable="tap_pos", element_index=0, element="trafo")
    ow = _run_normal(net, run_function)
    assert np.allclose(vm_pu, ow.output[_v_var(run_function)], rtol=0, atol=1e-6)
    assert np.allclose(ll, ow.output["res_line.loading_percent"], rtol=0, atol=1e-6)


def test_const_pq_gen_trafo_tap_shifter(simple_test_net, run_function):
    # allows to use recycle = {"bus_pq", "gen", "trafo"}
    net = simple_test_net
    net.trafo.loc[0, ["tap_step_percent", "tap_step_degree", "tap_changer_type"]] = 1, 10, "Ratio"
    profiles, _ = create_data_source(n_timesteps)
    profiles["ext_grid"] = np.ones(n_timesteps) + np.arange(0, n_timesteps) * 1e-2
    ds = DFData(profiles)
    # 1load
    c1 = add_const(net, ds, recycle=None)
    c2 = add_const(
        net,
        ds,
        recycle=None,
        profile_name="ext_grid",
        variable=_v_var(run_function, False),
        element_index=0,
        element="ext_grid",
    )
    c3 = add_const(
        net, ds, recycle=None, profile_name="trafo_tap", variable="tap_pos", element_index=0, element="trafo"
    )

    vm_pu, ll = _run_recycle(net, run_function)
    del c1, c2, c3

    # calculate the same results without recycle
    add_const(net, ds, recycle=False)
    add_const(
        net,
        ds,
        recycle=False,
        profile_name="ext_grid",
        variable=_v_var(run_function, False),
        element_index=0,
        element="ext_grid",
    )
    add_const(net, ds, recycle=None, profile_name="trafo_tap", variable="tap_pos", element_index=0, element="trafo")
    ow = _run_normal(net, run_function)
    assert np.allclose(vm_pu, ow.output[_v_var(run_function)], rtol=0, atol=1e-6)
    assert np.allclose(ll, ow.output["res_line.loading_percent"], rtol=0, atol=1e-6)


def test_const_pq_out_of_service(simple_test_net, run_function):
    # allows to use recycle = {"bus_pq"} and fast output read
    net = simple_test_net
    for _ in range(3):
        (b,) = create_buses(net, 1, 20.0, in_service=False)
        create_lines(net, 2, b, line_params="149-AL1/24-ST1A 20.0", length_km=1.0, in_service=False)
        create_transformer(net, 2, b, std_type="25 MVA 110/20 kV", in_service=False)
        create_transformer3w(net, 1, 2, b, std_type="63/25/38 MVA 110/20/10 kV", in_service=False)
    _, ds = create_data_source(n_timesteps)
    # 1load
    c = add_const(net, ds, recycle=None)
    vm_pu, ll = _run_recycle(net, run_function)
    del c

    # calculate the same results without recycle
    add_const(net, ds, recycle=False)
    ow = _run_normal(net, run_function)
    in_service = net.bus.loc[net.bus.in_service].index
    assert np.allclose(vm_pu.loc[:, in_service], ow.output[_v_var(run_function)].loc[:, in_service])
    in_service = net.line.loc[net.line.in_service].index
    assert np.allclose(ll.loc[:, in_service], ow.output["res_line.loading_percent"].loc[:, in_service])


def _tdpf_two_bus_net():
    """A small two winding net whose loading varies with the constant control."""
    from pandapower.create import create_ext_grid, create_load, create_transformer_from_parameters
    from pandapower.network import pandapowerNet

    net = pandapowerNet(name="tdpf_recycle")
    hv_bus, lv_bus = create_buses(net, 2, vn_kv=[110.0, 10.0])
    create_ext_grid(net, hv_bus)
    create_transformer_from_parameters(
        net,
        hv_bus,
        lv_bus,
        sn_mva=40.0,
        vn_hv_kv=110.0,
        vn_lv_kv=10.0,
        vk_percent=10.0,
        vkr_percent=0.5,
        pfe_kw=20.0,
        i0_percent=0.1,
    )
    create_load(net, lv_bus, p_mw=10.0)
    return net


def test_recycle_disabled_when_tdpf_is_requested():
    """The voltage-only recycle path must not be used when TDPF produces transformer temperatures.

    ``_recycled_powerflow`` returns before the results are extracted, so the thermal post-processing
    never runs. The network would then report the temperature of an earlier step even though the time
    step completed. This must not depend on whether the output writer happens to log a thermal variable.
    """
    pytest.importorskip("transformer_thermal_model")
    from pandapower.timeseries.run_time_series import get_recycle_settings

    net = _tdpf_two_bus_net()
    # 2-tuples are required for the voltage-only batch path to be selectable at all
    OutputWriter(
        net,
        output_path=tempfile.gettempdir(),
        output_file_type=".json",
        log_variables=[("res_bus", "vm_pu")],
    )

    # `get_recycle_settings` rewrites `ow.log_variables`, so re-create the writer for the second call
    assert get_recycle_settings(net, run=runpp)["only_v_results"] is True

    net = _tdpf_two_bus_net()
    OutputWriter(
        net,
        output_path=tempfile.gettempdir(),
        output_file_type=".json",
        log_variables=[("res_bus", "vm_pu")],
    )
    assert get_recycle_settings(net, run=runpp, tdpf=True)["only_v_results"] is False


def test_recycle_disabled_when_tdpf_is_set_in_user_pf_options():
    """A configured TDPF must be honoured as well, not only the keyword argument."""
    pytest.importorskip("transformer_thermal_model")
    from pandapower.run import set_user_pf_options
    from pandapower.timeseries.run_time_series import get_recycle_settings

    net = _tdpf_two_bus_net()
    OutputWriter(
        net,
        output_path=tempfile.gettempdir(),
        output_file_type=".json",
        log_variables=[("res_bus", "vm_pu")],
    )

    set_user_pf_options(net, tdpf=True)
    try:
        assert get_recycle_settings(net, run=runpp)["only_v_results"] is False
    finally:
        set_user_pf_options(net)


def test_timeseries_trafo_temperature_updates_with_tdpf():
    """Each time step must produce a new transformer temperature, even when only voltages are logged."""
    pytest.importorskip("transformer_thermal_model")
    net = _tdpf_two_bus_net()
    _, ds = create_data_source(n_timesteps)
    ConstControl(
        net,
        element="load",
        variable="p_mw",
        element_index=[0],
        data_source=ds,
        profile_name=["load1"],
    )
    # only a voltage is logged, so the voltage-only batch path would be chosen without the TDPF gate
    OutputWriter(
        net,
        output_path=tempfile.gettempdir(),
        output_file_type=".json",
        log_variables=[("res_bus", "vm_pu")],
    )

    temperatures = []

    def recording_runpp(inner_net, **kwargs):
        runpp(inner_net, **kwargs)
        temperatures.append(float(inner_net.res_trafo.at[0, "temperature_degree_celsius"]))

    run_timeseries(net, time_steps, run_control_fct=recording_runpp, tdpf=True, verbose=False)

    assert len(temperatures) == n_timesteps
    # the load profile varies, so every step must yield its own temperature
    assert len({round(value, 6) for value in temperatures}) == n_timesteps


def _constant_load_net(p_mw=40.0, n_steps=6):
    """A two bus net with a constant load, so only the thermal inertia can change the temperature."""
    from pandapower.create import create_ext_grid, create_load, create_transformer_from_parameters
    from pandapower.network import pandapowerNet

    net = pandapowerNet(name="tdpf_chaining")
    hv_bus, lv_bus = create_buses(net, 2, vn_kv=[110.0, 10.0])
    create_ext_grid(net, hv_bus)
    create_transformer_from_parameters(
        net,
        hv_bus,
        lv_bus,
        sn_mva=40.0,
        vn_hv_kv=110.0,
        vn_lv_kv=10.0,
        vk_percent=10.0,
        vkr_percent=0.5,
        pfe_kw=20.0,
        i0_percent=0.1,
    )
    create_load(net, lv_bus, p_mw=p_mw)
    ds = DFData(pd.DataFrame({"p": [p_mw] * n_steps}))
    ConstControl(net, "load", "p_mw", element_index=[0], data_source=ds, profile_name=["p"])
    OutputWriter(
        net,
        time_steps=range(n_steps),
        output_path=tempfile.gettempdir(),
        output_file_type=".json",
        log_variables=[("res_trafo", "top_oil_temperature_degree_celsius")],
    )
    return net


def test_timeseries_top_oil_temperature_is_chained_across_steps():
    """The default runpp path must carry the top-oil temperature from one step into the next.

    Without the tracker every step would restart from the cold start condition, so a constant load
    would produce the same cold temperature at every step instead of heating up towards the steady
    state. The thermal inertia only shows up when the previous top-oil temperature is fed back in.
    """
    pytest.importorskip("transformer_thermal_model")
    n_steps = 6
    net = _constant_load_net(n_steps=n_steps)
    run_timeseries(net, time_steps=range(n_steps), tdpf=True, tdpf_delay_s=3600.0, verbose=False)

    profile = list(net.output_writer.iat[0, 0].output["res_trafo.top_oil_temperature_degree_celsius"].values.ravel())

    # a steady state solve of the same net is the limit the transient must approach
    steady = copy.deepcopy(net)
    for controller in steady.controller["object"]:
        controller.data_source = None
    steady.load["p_mw"] = 40.0
    runpp(steady, tdpf=True)
    steady_state = float(steady.res_trafo.at[0, "top_oil_temperature_degree_celsius"])

    assert len(profile) == n_steps
    # strictly heating up towards the steady state instead of staying at the cold start value
    assert all(later > earlier for earlier, later in itertools.pairwise(profile))
    assert profile[-1] < steady_state
    # the cold start value must not repeat, which is what the missing tracker produced
    assert profile[-1] > profile[0]


def test_timeseries_top_oil_temperature_steady_state_without_delay():
    """Without thermal inertia every step is the steady state, so chaining must not change anything."""
    pytest.importorskip("transformer_thermal_model")
    n_steps = 4
    net = _constant_load_net(n_steps=n_steps)
    run_timeseries(net, time_steps=range(n_steps), tdpf=True, verbose=False)

    profile = list(net.output_writer.iat[0, 0].output["res_trafo.top_oil_temperature_degree_celsius"].values.ravel())

    assert np.allclose(profile, profile[0])
    assert np.isfinite(profile[0])


def test_timeseries_without_tdpf_has_no_tracker():
    """A run without TDPF must not create a thermal tracker at all."""
    from pandapower.timeseries.run_time_series import init_time_series

    net = _constant_load_net(n_steps=2)
    ts_variables = init_time_series(net, range(2), verbose=False)

    assert ts_variables["transformer_thermal_tracker"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

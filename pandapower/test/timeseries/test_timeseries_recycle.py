import tempfile

import numpy as np
import pytest

import pandapower as pp
from pandapower.test.timeseries.test_output_writer import create_data_source, OutputWriter, simple_test_net, \
    ConstControl, run_timeseries
from pandapower.timeseries.read_batch_results import get_batch_line_results, get_batch_trafo_results, \
    get_batch_trafo3w_results, v_to_i_s, polar_to_rad

n_timesteps = 5
time_steps = range(0, n_timesteps)
def add_const(net, ds, recycle):
    return ConstControl(net, element='load', variable='p_mw', element_index=[0, 1, 2],
                        data_source=ds, profile_name=["load1", "load2_mv_p", "load3_hv_p"],
                        recycle=recycle)


def test_batch_output_reader(simple_test_net):
    net = simple_test_net
    _, ds = create_data_source(n_timesteps)
    # 1load
    c = add_const(net, ds, recycle=True)

    ow = OutputWriter(net, output_path=tempfile.gettempdir(), output_file_type=".json", log_variables=list())
    ow.log_variable('ppc_bus', 'vm')
    ow.log_variable('ppc_bus', 'va')

    recycle = dict(trafo=False, gen=False, bus_pq=True)
    only_v = True
    pp.runpp(net, only_v_results=only_v, recycle=recycle)
    run_timeseries(net, time_steps, recycle=recycle, only_v_results=only_v)

    vm, va = ow.output["ppc_bus.vm"], ow.output["ppc_bus.va"]
    s, s_abs, i_abs = v_to_i_s(net, vm, va)

    del net, ow, c
    net = simple_test_net
    net.output_writer.drop(index=net.output_writer.index, inplace=True)
    net.controller.drop(index=net.controller.index, inplace=True)
    add_const(net, ds, recycle=False)
    ow = OutputWriter(net, output_path=tempfile.gettempdir(), output_file_type=".json", log_variables=list())
    ow.log_variable('res_line', 'i_from_ka')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_line', 'i_to_ka')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_trafo', 'loading_percent')
    ow.log_variable('res_trafo', 'i_hv_ka')
    ow.log_variable('res_trafo', 'i_lv_ka')
    ow.log_variable('res_trafo3w', 'loading_percent')
    ow.log_variable('res_trafo3w', 'i_hv_ka')
    ow.log_variable('res_trafo3w', 'i_mv_ka')
    ow.log_variable('res_trafo3w', 'i_lv_ka')
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_bus', 'va_degree')

    pp.runpp(net)
    run_timeseries(net, time_steps, trafo_loading="current")

    vm_pu, va_degree = ow.output["res_bus.vm_pu"], ow.output["res_bus.va_degree"]
    i_from_ka_normal, i_to_ka_normal = ow.output["res_line.i_from_ka"], ow.output["res_line.i_to_ka"]
    i_hv_ka_normal, i_lv_ka_normal = ow.output["res_trafo.i_hv_ka"], ow.output["res_trafo.i_lv_ka"]
    i3_hv_ka, i3_lm_ka, i3_lv_ka = ow.output["res_trafo3w.i_hv_ka"], ow.output["res_trafo3w.i_mv_ka"], \
                                   ow.output["res_trafo3w.i_lv_ka"]
    t3_loading_percent_normal = ow.output["res_trafo3w.loading_percent"]
    loading_percent_normal = ow.output["res_line.loading_percent"]
    t_loading_percent_normal = ow.output["res_trafo.loading_percent"]

    v_normal = polar_to_rad(vm_pu, va_degree)
    v_batch = polar_to_rad(vm, va)
    i_ka, i_from_ka, i_to_ka, loading_percent = get_batch_line_results(net, i_abs)
    i_ka, i_hv_ka, i_lv_ka, s_mva, ld_trafo = get_batch_trafo_results(net, i_abs, s_abs)
    i_h, i_m, i_l, ld3_trafo = get_batch_trafo3w_results(net, i_abs, s_abs)

    # v batch contains aux buses of trafo3w. we can only compare non aux bus voltages
    assert np.allclose(v_normal, v_batch[:, :v_normal.shape[1]])
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

def test_const_pq(simple_test_net):
    # allows to use recycle = {"bus_pq"} and fast output read
    net = simple_test_net

    _, ds = create_data_source(n_timesteps)
    # 1load
    c = add_const(net, ds, recycle=None)
    # default log variables are res_bus.vm_pu, res_line.loading_percent
    ow = OutputWriter(net, output_path=tempfile.gettempdir(), output_file_type=".json")

    run_timeseries(net, time_steps)


def test_const_gen(simple_test_net):
    # allows to use recycle = {"gen"} and fast output read
    pass

def test_const_ext_grid(simple_test_net):
    # allows to use recycle = {"gen"} and fast output read
    pass

def test_trafo_tap(simple_test_net):
    # allows to use recycle = {"trafo"} but not fast output read
    pass

def test_const_pq_gen_trafo_tap(simple_test_net):
    # allows to use recycle = {"bus_pq", "gen", "trafo"}
    pass

if __name__ == "__main__":
    # pytest.main(['-s', __file__])
    test_const_pq(simple_test_net())
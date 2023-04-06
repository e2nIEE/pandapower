import copy

import numpy as np
import pytest

import pandapower as pp
from pandapower.toolbox import nets_equal
from pandapower.test import add_grid_connection
from pandapower.test.consistency_checks import runpp_with_consistency_checks, rundcpp_with_consistency_checks


@pytest.fixture
def recycle_net():
    net = pp.create_empty_network()
    b1, b2, ln = add_grid_connection(net)
    pl = 1.2
    ql = 1.1
    ps = 0.5
    u_set = 1.0

    b3 = pp.create_bus(net, vn_kv=.4)
    pp.create_bus(net, vn_kv=.4, in_service=False)
    pp.create_line_from_parameters(net, b2, b3, 12.2, r_ohm_per_km=0.08, x_ohm_per_km=0.12,
                                   c_nf_per_km=300, max_i_ka=.2, df=.8)
    pp.create_load(net, b3, p_mw=pl, q_mvar=ql)
    pp.create_gen(net, b2, p_mw=ps, vm_pu=u_set)
    return net


def test_recycle_pq(recycle_net):
    # Calls recycle functions and tests if load is update
    net = recycle_net
    pl = 1.2
    ql = 0.
    net["load"].at[0, "q_mvar"] = ql
    runpp_with_consistency_checks(net, recycle=dict(trafo=False, gen=False, bus_pq=True))
    assert np.allclose(net.res_load.at[0, "p_mw"], pl)
    assert np.allclose(net.res_load.at[0, "q_mvar"], ql)
    pl = 0.8
    ql = 0.55
    net.load.at[0, "p_mw"] = pl
    net["load"].at[0, "q_mvar"] = ql
    runpp_with_consistency_checks(net, recycle=dict(bus_pq=True, trafo=False, gen=False))
    assert np.allclose(net.res_load.p_mw.iloc[0], pl)
    assert np.allclose(net.res_load.q_mvar.iloc[0], ql)


def test_recycle_gen(recycle_net):
    net = recycle_net
    # update values of gens
    ps = 0.25
    u_set = 0.98

    net["gen"].at[0, "p_mw"] = ps
    net["gen"].at[0, "vm_pu"] = u_set

    runpp_with_consistency_checks(net, recycle=dict(trafo=False, bus_pq=False, gen=True))
    assert np.allclose(net.res_gen.at[0, "p_mw"], ps)
    assert np.allclose(net.res_gen.at[0, "vm_pu"], u_set)

    ps = 0.5
    u_set = 0.99
    net["gen"].at[0, "p_mw"] = ps
    net["gen"].at[0, "vm_pu"] = u_set
    runpp_with_consistency_checks(net, recycle=dict(trafo=False, bus_pq=False, gen=True))

    assert np.allclose(net.res_gen.at[0, "vm_pu"], u_set)
    assert np.allclose(net.res_gen.at[0, "p_mw"], ps)


def test_recycle_trafo(recycle_net):
    # test trafo tap change
    net = recycle_net
    b4 = pp.create_bus(net, vn_kv=20.)
    pp.create_transformer(net, 3, b4, std_type="0.4 MVA 10/0.4 kV")

    net["trafo"].at[0, "tap_pos"] = 0
    runpp_with_consistency_checks(net, recycle=dict(trafo=True, bus_pq=False, gen=False))
    vm_pu = net.res_bus.at[b4, "vm_pu"]

    net["trafo"].at[0, "tap_pos"] = 5
    runpp_with_consistency_checks(net, recycle=dict(trafo=True, bus_pq=False, gen=False))
    assert not np.allclose(vm_pu, net.res_bus.at[b4, "vm_pu"])


def test_recycle_trafo_bus_gen(recycle_net):
    # test trafo tap change
    net = recycle_net
    b4 = pp.create_bus(net, vn_kv=20.)
    pp.create_transformer(net, 3, b4, std_type="0.4 MVA 10/0.4 kV")

    ps = 0.25
    u_set = 0.98
    pl = 1.2
    ql = 0.
    net["load"].at[0, "p_mw"] = pl
    net["load"].at[0, "q_mvar"] = ql
    net["gen"].at[0, "p_mw"] = ps
    net["gen"].at[0, "vm_pu"] = u_set
    net["trafo"].at[0, "tap_pos"] = 0
    runpp_with_consistency_checks(net, recycle=dict(trafo=True, bus_pq=True, gen=True))
    assert np.allclose(net.res_gen.at[0, "p_mw"], ps)
    assert np.allclose(net.res_gen.at[0, "vm_pu"], u_set)
    assert np.allclose(net.res_load.at[0, "p_mw"], pl)
    assert np.allclose(net.res_load.at[0, "q_mvar"], ql)
    vm_pu = net.res_bus.at[b4, "vm_pu"]

    ps = 0.5
    u_set = 0.99
    pl = 1.
    ql = 0.5
    net["load"].at[0, "p_mw"] = pl
    net["load"].at[0, "q_mvar"] = ql
    net["gen"].at[0, "p_mw"] = ps
    net["gen"].at[0, "vm_pu"] = u_set
    net["trafo"].at[0, "tap_pos"] = 5
    runpp_with_consistency_checks(net, recycle=dict(trafo=True, bus_pq=True, gen=True))

    assert not np.allclose(vm_pu, net.res_bus.at[b4, "vm_pu"])
    assert np.allclose(net.res_gen.at[0, "p_mw"], ps)
    assert np.allclose(net.res_gen.at[0, "vm_pu"], u_set)
    assert np.allclose(net.res_load.at[0, "p_mw"], pl)
    assert np.allclose(net.res_load.at[0, "q_mvar"], ql)


def test_result_index_unsorted():
    net = pp.create_empty_network()
    pp.set_user_pf_options(net, tolerance_mva=1e-9)

    b1 = pp.create_bus(net, vn_kv=0.4, index=4)
    b2 = pp.create_bus(net, vn_kv=0.4, index=2)
    b3 = pp.create_bus(net, vn_kv=0.4, index=3)

    pp.create_gen(net, b1, p_mw=0.01, vm_pu=0.4)
    pp.create_load(net, b2, p_mw=0.01)
    pp.create_ext_grid(net, b3)

    pp.create_line(net, from_bus=b1, to_bus=b2, length_km=0.5, std_type="NAYY 4x120 SE")
    pp.create_line(net, from_bus=b1, to_bus=b3, length_km=0.5, std_type="NAYY 4x120 SE")
    net_recycle = copy.deepcopy(net)
    pp.runpp(net_recycle)
    pp.runpp(net_recycle, recycle=dict(trafo=True, bus_pq=True, gen=True))
    pp.runpp(net)

    assert nets_equal(net, net_recycle, atol=1e-12)


def test_recycle_dc(recycle_net):
    net = recycle_net
    pl = 1.2
    ql = 0.
    net["load"].at[0, "q_mvar"] = ql
    rundcpp_with_consistency_checks(net, recycle=dict(trafo=False, gen=False, bus_pq=True))
    assert np.allclose(net.res_load.at[0, "p_mw"], pl)
    pl = 0.8
    ql = 0.55
    net.load.at[0, "p_mw"] = pl
    net["load"].at[0, "q_mvar"] = ql
    rundcpp_with_consistency_checks(net, recycle=dict(bus_pq=True, trafo=False, gen=False))
    assert np.allclose(net.res_load.p_mw.iloc[0], pl)


def test_recycle_dc_trafo_shift(recycle_net):
    net = recycle_net
    pp.set_user_pf_options(net, calculate_voltage_angles=True)
    b4 = pp.create_bus(net, vn_kv=20.)
    pp.create_transformer(net, 3, b4, std_type="0.4 MVA 10/0.4 kV")
    net["trafo"].at[0, "tap_pos"] = 0
    net["trafo"].at[0, "tap_step_percent"] = 1
    #net["trafo"].at[0, "tap_phase_shifter"] = True
    net["trafo"].at[0, "tap_step_degree"] = 30
    net2 = net.deepcopy()
    pl = 1.2
    ql = 0.
    net["load"].at[0, "q_mvar"] = ql
    rundcpp_with_consistency_checks(net, recycle=dict(bus_pq=True, trafo=True, gen=False))
    assert np.allclose(net.res_load.at[0, "p_mw"], pl)
    pl = 0.8
    ql = 0.55
    net.load.at[0, "p_mw"] = pl
    net.load.at[0, "q_mvar"] = ql
    net.trafo.at[0, "tap_pos"] = 3
    # here: because tap_step_percent is not the same, recycle will not be triggered
    rundcpp_with_consistency_checks(net, recycle=dict(bus_pq=True, trafo=True, gen=False))
    assert np.allclose(net.res_load.p_mw.iloc[0], pl)
    net2.load.at[0, "p_mw"] = pl
    net2.load.at[0, "q_mvar"] = ql
    net2.trafo.at[0, "tap_pos"] = 3
    pp.rundcpp(net2)
    assert np.allclose(net.res_bus.va_degree, net2.res_bus.va_degree, rtol=0, atol=1e-9, equal_nan=True)


def test_recycle_dc_trafo_ideal(recycle_net):
    net = recycle_net
    pp.set_user_pf_options(net, calculate_voltage_angles=True)
    v = net.bus.vn_kv.at[3]
    b4 = pp.create_bus(net, vn_kv=v)
    pp.create_transformer_from_parameters(net, 3, b4, 10, v, v, 0.5, 12, 10, 0.1, 0,
                                          tap_side='hv', tap_neutral=0, tap_max=10, tap_min=-10,
                                          tap_step_percent=0, tap_step_degree=30, tap_pos=0, tap_phase_shifter=True)
    net2 = net.deepcopy()
    pl = 1.2
    ql = 0.
    net["load"].at[0, "q_mvar"] = ql
    rundcpp_with_consistency_checks(net, recycle=dict(bus_pq=True, trafo=True, gen=False))
    assert np.allclose(net.res_load.at[0, "p_mw"], pl)
    pl = 0.8
    ql = 0.55
    net.load.at[0, "p_mw"] = pl
    net.load.at[0, "q_mvar"] = ql
    net.trafo.at[0, "tap_pos"] = 3
    # here: ideal phase shifter, so tap is the same and shift is different
    rundcpp_with_consistency_checks(net, recycle=dict(bus_pq=True, trafo=True, gen=False))
    assert np.allclose(net.res_load.p_mw.iloc[0], pl)
    net2.load.at[0, "p_mw"] = pl
    net2.load.at[0, "q_mvar"] = ql
    net2.trafo.at[0, "tap_pos"] = 3
    pp.rundcpp(net2)
    assert np.allclose(net.res_bus.va_degree, net2.res_bus.va_degree, rtol=0, atol=1e-9, equal_nan=True)


if __name__ == "__main__":
    pytest.main([__file__, "-xs"])
    # test_recycle_gen(recycle_net())

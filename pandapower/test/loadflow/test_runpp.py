# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import copy
import os
import re

import numpy as np
import pandas as pd
import pytest

from pandapower import pp_dir
from pandapower.auxiliary import _check_connectivity, _add_ppc_options, lightsim2grid_available
from pandapower.create import (
    create_bus, create_ext_grid, create_dcline, create_load, create_sgen, create_switch, create_transformer,
    create_xward, create_transformer3w, create_gen, create_shunt, create_line_from_parameters, create_line,
    create_impedance, create_storage, create_buses, create_transformer_from_parameters,
    create_transformer3w_from_parameters, create_poly_cost
)
from pandapower.file_io import from_json
from pandapower.network import pandapowerNet
from pandapower.networks import create_cigre_network_mv, four_loads_with_branches_out, \
    example_simple, simple_four_bus_system, example_multivoltage, case118
from pandapower.pd2ppc import _pd2ppc
from pandapower.pf.create_jacobian import _create_J_without_numba
from pandapower.pf.run_newton_raphson_pf import _get_pf_variables_from_ppci
from pandapower.powerflow import LoadflowNotConverged
from pandapower.pypower.idx_brch import BR_R, BR_X, BR_B, BR_G
from pandapower.pypower.makeYbus import makeYbus as makeYbus_pypower
from pandapower.results import reset_results
from pandapower.run import set_user_pf_options, runpp, runopp
from pandapower.test.consistency_checks import runpp_with_consistency_checks
from pandapower.test.control.test_shunt_control import simple_test_net_shunt_control
from pandapower.test.helper_functions import add_grid_connection, create_test_line, assert_net_equal, assert_res_equal
from pandapower.test.loadflow.result_test_network_generator import add_test_xward, add_test_trafo3w, \
    add_test_line, add_test_oos_bus_with_is_element, result_test_network_generator, add_test_trafo
from pandapower.toolbox import nets_equal, drop_elements
from pandapower.control.util.auxiliary import create_q_capability_characteristics_object

import logging

logger = logging.getLogger(__name__)

try:
    from pandapower.pf.makeYbus_numba import makeYbus as makeYbus_numba

    numba_installed = True
except ImportError:
    numba_installed = False


def test_minimal_net(**kwargs):
    # tests corner-case when the grid only has 1 bus and an ext-grid
    net = pandapowerNet(name="test_minimal_net")
    b = create_bus(net, 110)
    create_ext_grid(net, b)
    runpp_with_consistency_checks(net, **kwargs)

    create_load(net, b, p_mw=0.1)
    runpp_with_consistency_checks(net, **kwargs)

    b2 = create_bus(net, 110)
    create_switch(net, b, b2, 'b')
    create_sgen(net, b2, p_mw=0.2)
    runpp_with_consistency_checks(net, **kwargs)


def test_set_user_pf_options():
    net = example_simple()
    runpp(net)

    old_options = net._options.copy()
    test_options = {key: i for i, key in enumerate(old_options.keys())}

    set_user_pf_options(net, hello='bye', **test_options)
    test_options.update({'hello': 'bye'})

    assert net.user_pf_options == test_options

    # remove what is in user_pf_options and add hello=world
    set_user_pf_options(net, overwrite=True, hello='world')
    assert net.user_pf_options == {'hello': 'world'}

    # check if 'hello' is added to net._options, but other options are untouched
    runpp(net)
    assert 'hello' in net._options.keys() and net._options['hello'] == 'world'
    net._options.pop('hello')
    assert net._options == old_options

    # check if user_pf_options can be deleted and net._options is as it was before
    set_user_pf_options(net, overwrite=True, hello='world')
    set_user_pf_options(net, overwrite=True)
    assert net.user_pf_options == {}
    runpp(net)
    assert 'hello' not in net._options.keys()

    # see if user arguments overrule user_pf_options, but other user_pf_options still have the
    # priority
    set_user_pf_options(net, tolerance_mva=1e-6, max_iteration=20)
    runpp(net, tolerance_mva=1e-2)
    assert np.isclose(net.user_pf_options['tolerance_mva'], 1e-6)
    assert np.isclose(net._options['tolerance_mva'], 1e-2)
    assert net._options['max_iteration'] == 20


def test_kwargs_with_user_options():
    net = example_simple()
    runpp(net)
    assert net._options["trafo3w_losses"] == "hv"
    set_user_pf_options(net, trafo3w_losses="lv")
    runpp(net)
    assert net._options["trafo3w_losses"] == "lv"

    # check providing the kwargs options in runpp overrides user_pf_options
    set_user_pf_options(net, init_vm_pu="results")
    runpp(net, init_vm_pu="flat")
    assert net.user_pf_options["init_vm_pu"] == "results"
    assert net._options["init_vm_pu"] == "flat"


@pytest.mark.xfail(reason="Until now there was no way found to dynamically identify "
                          "the arguments passed to runpp, so if the user options are "
                          "overwritten with the default values, this is not recognized.")
def test_overwrite_default_args_with_user_options():
    net = example_simple()
    runpp(net)
    assert net._options["check_connectivity"] is True
    set_user_pf_options(net, check_connectivity=False)
    runpp(net)
    assert net._options["check_connectivity"] is False
    runpp(net, check_connectivity=True)
    assert net._options["check_connectivity"] is True


def test_runpp_init():
    net = pandapowerNet(name="test_runpp_init")
    b1, b2, l1 = add_grid_connection(net)
    b3 = create_bus(net, vn_kv=0.4)
    tidx = create_transformer(net, hv_bus=b2, lv_bus=b3, std_type="0.25 MVA 20/0.4 kV")
    net.trafo.at[tidx, "shift_degree"] = 70
    runpp(net, calculate_voltage_angles="auto")
    va = net.res_bus.va_degree.at[4]
    runpp(net, calculate_voltage_angles=True, init_va_degree="dc")
    assert np.allclose(va - net.trafo.shift_degree.at[tidx], net.res_bus.va_degree.at[4])
    runpp(net, calculate_voltage_angles=True, init_va_degree="results")
    assert np.allclose(va - net.trafo.shift_degree.at[tidx], net.res_bus.va_degree.at[4])


def test_runpp_init_auxiliary_buses():
    net = pandapowerNet(name="test_runpp_init_auxiliary_buses")
    b1, b2, l1 = add_grid_connection(net, vn_kv=110.)
    b3 = create_bus(net, vn_kv=20.)
    b4 = create_bus(net, vn_kv=10.)
    tidx = create_transformer3w(net, b2, b3, b4, std_type='63/25/38 MVA 110/20/10 kV')
    create_load(net, b3, p_mw=5)
    create_load(net, b4, p_mw=5)
    create_xward(net, b4, ps_mw=1, qs_mvar=1, pz_mw=1, qz_mvar=1, r_ohm=0.1, x_ohm=0.1,
                 vm_pu=1.0)
    net.trafo3w.at[tidx, "shift_lv_degree"] = 120
    net.trafo3w.at[tidx, "shift_mv_degree"] = 80
    runpp(net)
    va = net.res_bus.va_degree.at[b2]
    runpp(net, calculate_voltage_angles=True, init_va_degree="dc")
    assert np.allclose(va - net.trafo3w.shift_mv_degree.at[tidx], net.res_bus.va_degree.at[b3],
                       atol=2)
    assert np.allclose(va - net.trafo3w.shift_lv_degree.at[tidx], net.res_bus.va_degree.at[b4],
                       atol=2)
    runpp(net, calculate_voltage_angles=True, init_va_degree="results")
    assert np.allclose(va - net.trafo3w.shift_mv_degree.at[tidx], net.res_bus.va_degree.at[b3],
                       atol=2)
    assert np.allclose(va - net.trafo3w.shift_lv_degree.at[tidx], net.res_bus.va_degree.at[b4],
                       atol=2)


def test_result_iter():
    for net in result_test_network_generator():
        try:
            runpp_with_consistency_checks(net, enforce_q_lims=True)
        except (AssertionError):
            raise UserWarning("Consistency Error after adding %s" % net.last_added_case)
        except(LoadflowNotConverged):
            raise UserWarning("Power flow did not converge after adding %s" % net.last_added_case)


@pytest.fixture
def bus_bus_net():
    net = pandapowerNet(name="bus_bus_net")
    add_grid_connection(net)
    for _u in range(4):
        create_bus(net, vn_kv=.4)
    create_load(net, 5, p_mw=0.01)
    create_switch(net, 3, 6, et="b")
    create_switch(net, 4, 5, et="b")
    create_switch(net, 6, 5, et="b")
    create_switch(net, 0, 7, et="b")
    create_test_line(net, 4, 7)
    create_load(net, 4, p_mw=0.01)
    return net


def test_bus_bus_switches(bus_bus_net):
    net = bus_bus_net
    runpp(net)
    assert net.res_bus.vm_pu.at[3] == net.res_bus.vm_pu.at[4] == net.res_bus.vm_pu.at[5] == \
           net.res_bus.vm_pu.at[6]
    assert net.res_bus.vm_pu.at[0] == net.res_bus.vm_pu.at[7]

    net.bus.at[5, "in_service"] = False
    runpp(net)
    assert net.res_bus.vm_pu.at[3] == net.res_bus.vm_pu.at[6]
    assert net.res_bus.vm_pu.at[0] == net.res_bus.vm_pu.at[7]
    assert pd.isnull(net.res_bus.vm_pu.at[5])
    assert net.res_bus.vm_pu.at[6] != net.res_bus.vm_pu.at[4]


def test_bus_bus_switches_merges_two_gens(bus_bus_net):
    "buses should not be fused if two gens are connected"
    net = bus_bus_net
    net.bus.at[5, "in_service"] = False
    create_gen(net, 6, 10)
    create_gen(net, 4, 10)
    net.bus.at[5, "in_service"] = True
    runpp(net)
    assert net.converged


def test_bus_bus_switches_throws_exception_for_two_gen_with_diff_vm(bus_bus_net):
    "buses should not be fused if two gens are connected"
    net = bus_bus_net
    create_gen(net, 6, 10, 1.)
    create_gen(net, 4, 10, 1.1)
    with pytest.raises(UserWarning):
        runpp(net)


@pytest.fixture
def z_switch_net():
    net = pandapowerNet(name="z_switch_net")
    for i in range(3):
        create_bus(net, vn_kv=.4)
        create_load(net, i, p_mw=0.1)
    create_ext_grid(net, 0, vm_pu=1.0)
    create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=0.1 / np.sqrt(2),
                                x_ohm_per_km=0.1 / np.sqrt(2),
                                c_nf_per_km=0, max_i_ka=.2)
    create_switch(net, 0, 2, et="b", z_ohm=0.1)
    return net


@pytest.mark.parametrize("numba", [True, False])
def test_z_switch(z_switch_net, numba):
    net = z_switch_net
    runpp(net, numba=numba, switch_rx_ratio=1)
    assert pytest.approx(net.res_bus.vm_pu.at[1], abs=1e-9) == net.res_bus.vm_pu.at[2]

    net_zero_z_switch = copy.deepcopy(net)
    net_zero_z_switch.switch.z_ohm = 0
    runpp(net_zero_z_switch, numba=numba, switch_rx_ratio=1)
    assert pytest.approx(net_zero_z_switch.res_bus.vm_pu.at[0], abs=1e-9) == net_zero_z_switch.res_bus.vm_pu.at[2]


@pytest.fixture
def z_switch_net_4bus_parallel():
    net = pandapowerNet(name="z_switch_net_4bus_parallel")
    for i in range(4):
        create_bus(net, vn_kv=.4)
        create_load(net, i, p_mw=0.1)
    create_ext_grid(net, 0, vm_pu=1.0)
    create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=0.1 / np.sqrt(2),
                                x_ohm_per_km=0.1 / np.sqrt(2),
                                c_nf_per_km=0, max_i_ka=.2)
    create_line_from_parameters(net, 1, 3, 1, r_ohm_per_km=0.1 / np.sqrt(2),
                                x_ohm_per_km=0.1 / np.sqrt(2),
                                c_nf_per_km=0, max_i_ka=.2)
    create_switch(net, 0, 2, et="b", z_ohm=0.1)
    create_switch(net, 0, 2, et="b", z_ohm=0)
    return net


@pytest.fixture
def z_switch_net_4bus():
    net = pandapowerNet(name="z_switch_net_4bus")
    for i in range(4):
        create_bus(net, vn_kv=.4)
        create_load(net, i, p_mw=0.01)
    create_ext_grid(net, 0, vm_pu=1.0)
    create_line_from_parameters(net, 0, 1, 1, r_ohm_per_km=0.1 / np.sqrt(2),
                                x_ohm_per_km=0.1 / np.sqrt(2),
                                c_nf_per_km=0, max_i_ka=.2)
    create_switch(net, 1, 2, et="b", z_ohm=0.1)
    create_switch(net, 2, 3, et="b", z_ohm=0)
    return net


@pytest.mark.parametrize("numba", [True, False])
def test_switch_fuse_z_ohm_0(z_switch_net_4bus_parallel, z_switch_net_4bus, numba):
    net = z_switch_net_4bus_parallel
    runpp(net, numba=numba)
    assert net.res_bus.vm_pu[0] == net.res_bus.vm_pu[2]
    assert net.res_switch.i_ka[0] == 0

    net = z_switch_net_4bus
    runpp(net, numba=numba)
    assert net.res_bus.vm_pu[2] == net.res_bus.vm_pu[3]
    assert net.res_bus.vm_pu[1] != net.res_bus.vm_pu[2]


@pytest.mark.parametrize("numba", [True, False])
def test_switch_z_ohm_different(z_switch_net_4bus_parallel, z_switch_net_4bus, numba):
    net = z_switch_net_4bus_parallel
    net.switch.at[1, 'z_ohm'] = 0.2
    runpp(net, numba=numba)
    assert net.res_bus.vm_pu[0] != net.res_bus.vm_pu[2]
    assert np.all(net.res_switch.i_ka > 0)

    net = z_switch_net_4bus
    net.switch.at[1, 'z_ohm'] = 0.2
    runpp(net, numba=numba)
    assert net.res_bus.vm_pu[2] != net.res_bus.vm_pu[3]
    assert net.res_bus.vm_pu[1] != net.res_bus.vm_pu[2]


def test_two_open_switches():
    net = pandapowerNet(name="test_two_open_switches")
    b1, b2, l1 = add_grid_connection(net)
    b3 = create_bus(net, vn_kv=20.)
    l2 = create_test_line(net, b2, b3)
    create_test_line(net, b3, b1)
    create_switch(net, b2, l2, et="l", closed=False)
    create_switch(net, b3, l2, et="l", closed=False)
    runpp(net)
    assert np.isnan(net.res_line.i_ka.at[l2]) or net.res_line.i_ka.at[l2] == 0


def test_oos_bus():
    net = pandapowerNet(name="test_oos_bus")
    add_test_oos_bus_with_is_element(net)
    assert runpp_with_consistency_checks(net)

    #    test for pq-node result
    create_shunt(net, 6, q_mvar=0.8)
    assert runpp_with_consistency_checks(net)

    #   1test for pv-node result
    create_gen(net, 4, p_mw=0.5)
    assert runpp_with_consistency_checks(net)


def get_isolated(net):
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=False,
                     trafo_model="t", check_connectivity=False,
                     mode="pf", switch_rx_ratio=2, init_vm_pu="flat",
                     init_va_degree="flat",
                     enforce_q_lims=False, recycle=None)

    ppc, ppci = _pd2ppc(net)
    return _check_connectivity(ppc)


def test_connectivity_check_island_without_pv_bus():
    # Network with islands without pv bus -> all buses in island should be set out of service
    net = create_cigre_network_mv(with_der=False)
    iso_buses, iso_p, iso_q, *_ = get_isolated(net)
    assert len(iso_buses) == 0
    assert np.isclose(iso_p, 0)
    assert np.isclose(iso_q, 0)

    isolated_bus1 = create_bus(net, vn_kv=20., name="isolated Bus1")
    isolated_bus2 = create_bus(net, vn_kv=20., name="isolated Bus2")
    create_line(net, isolated_bus2, isolated_bus1, length_km=1,
                std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV",
                name="IsolatedLine")
    iso_buses, iso_p, iso_q, *_ = get_isolated(net)
    assert len(iso_buses) == 2
    assert np.isclose(iso_p, 0)
    assert np.isclose(iso_q, 0)

    create_load(net, isolated_bus1, p_mw=0.2, q_mvar=0.02)
    create_sgen(net, isolated_bus2, p_mw=0.15, q_mvar=0.01)

    # with pytest.warns(UserWarning):
    iso_buses, iso_p, iso_q, *_ = get_isolated(net)
    assert len(iso_buses) == 2
    assert np.isclose(iso_p, 350)
    assert np.isclose(iso_q, 30)
    # with pytest.warns(UserWarning):
    runpp_with_consistency_checks(net, check_connectivity=True)


def test_connectivity_check_island_with_one_pv_bus():
    # Network with islands with one PV bus -> PV bus should be converted to the reference bus
    net = create_cigre_network_mv(with_der=False)
    iso_buses, iso_p, iso_q, *_ = get_isolated(net)
    assert len(iso_buses) == 0
    assert np.isclose(iso_p, 0)
    assert np.isclose(iso_q, 0)

    isolated_bus1 = create_bus(net, vn_kv=20., name="isolated Bus1")
    isolated_bus2 = create_bus(net, vn_kv=20., name="isolated Bus2")
    isolated_gen = create_bus(net, vn_kv=20., name="isolated Gen")
    isolated_pv_bus = create_gen(net, isolated_gen, p_mw=0.35, vm_pu=1.0, name="isolated PV bus")
    create_line(net, isolated_bus2, isolated_bus1, length_km=1,
                std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV", name="IsolatedLine")
    create_line(net, isolated_gen, isolated_bus1, length_km=1,
                std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV", name="IsolatedLineToGen")
    # with pytest.warns(UserWarning):
    iso_buses, iso_p, iso_q, *_ = get_isolated(net)

    # assert len(iso_buses) == 0
    # assert np.isclose(iso_p, 0)
    # assert np.isclose(iso_q, 0)
    #
    # create_load(net, isolated_bus1, p_mw=0.200., q_mvar=0.020)
    # create_sgen(net, isolated_bus2, p_mw=0.0150., q_mvar=-0.010)
    #
    # iso_buses, iso_p, iso_q = get_isolated(net)
    # assert len(iso_buses) == 0
    # assert np.isclose(iso_p, 0)
    # assert np.isclose(iso_q, 0)

    # with pytest.warns(UserWarning):
    runpp_with_consistency_checks(net, check_connectivity=True)


def test_connectivity_check_island_with_multiple_pv_buses():
    # Network with islands an multiple PV buses in the island -> Error should be thrown since it
    # would be random to choose just some PV bus as the reference bus
    net = create_cigre_network_mv(with_der=False)
    iso_buses, iso_p, iso_q, *_ = get_isolated(net)
    assert len(iso_buses) == 0
    assert np.isclose(iso_p, 0)
    assert np.isclose(iso_q, 0)

    isolated_bus1 = create_bus(net, vn_kv=20., name="isolated Bus1")
    isolated_bus2 = create_bus(net, vn_kv=20., name="isolated Bus2")
    isolated_pv_bus1 = create_bus(net, vn_kv=20., name="isolated PV bus1")
    isolated_pv_bus2 = create_bus(net, vn_kv=20., name="isolated PV bus2")
    create_gen(net, isolated_pv_bus1, p_mw=0.3, vm_pu=1.0, name="isolated PV bus1")
    create_gen(net, isolated_pv_bus2, p_mw=0.05, vm_pu=1.0, name="isolated PV bus2")

    create_line(net, isolated_pv_bus1, isolated_bus1, length_km=1,
                std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV",
                name="IsolatedLineToGen1")
    create_line(net, isolated_pv_bus2, isolated_bus2, length_km=1,
                std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV",
                name="IsolatedLineToGen2")
    create_line(net, isolated_bus2, isolated_bus1, length_km=1,
                std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV",
                name="IsolatedLine")
    # ToDo with pytest.warns(UserWarning):
    iso_buses, iso_p, iso_q, *_ = get_isolated(net)


def test_isolated_in_service_bus_at_oos_line():
    net = pandapowerNet(name="test_isolated_in_service_bus_at_oos_line")
    b1, b2, l1 = add_grid_connection(net)
    b = create_bus(net, vn_kv=135)
    l = create_line(net, b2, b, 0.1, std_type="NAYY 4x150 SE")
    net.line.loc[l, "in_service"] = False
    assert runpp_with_consistency_checks(net, init="flat")


def test_isolated_in_service_line():
    # ToDo: Fix this
    net = pandapowerNet(name="test_isolated_in_service_line")
    _, b2, l1 = add_grid_connection(net)
    b = create_bus(net, vn_kv=20.)
    create_line(net, b2, b, 0.1, std_type="NAYY 4x150 SE")
    net.line.loc[l1, "in_service"] = False
    assert runpp_with_consistency_checks(net, init="flat")


def test_makeYbus():
    # tests if makeYbus fails for nets where every bus is connected to each other
    net = pandapowerNet(name="test_makeYbus")
    b1, b2, l1 = add_grid_connection(net)

    # number of buses to create
    n_bus = 20
    bus_list = []
    # generate buses and connect them
    for _ in range(n_bus):
        bus_list.append(create_bus(net, vn_kv=20.))

    # connect the first bus to slack node
    create_test_line(net, bus_list[0], b2)
    # iterate over every bus and add connection to every other bus
    for bus_1 in bus_list:
        for bus_2 in bus_list:
            # add no connection to itself
            if bus_1 == bus_2:
                continue
            create_test_line(net, bus_1, bus_2)

    assert runpp_with_consistency_checks(net)


def test_test_sn_mva():
    test_net_gen1 = result_test_network_generator(sn_mva=1)
    test_net_gen2 = result_test_network_generator(sn_mva=2)
    for net1, net2 in zip(test_net_gen1, test_net_gen2):
        runpp(net1)
        runpp(net2)
        try:
            assert_net_equal(net1, net2, exclude_elms=["sn_mva"])
        except:
            raise UserWarning("Result difference due to sn_mva after adding %s" %
                              net1.last_added_case)


def test_bsfw_algorithm():
    net = example_simple()

    runpp(net)
    vm_nr = copy.copy(net.res_bus.vm_pu)
    va_nr = copy.copy(net.res_bus.va_degree)

    runpp(net, algorithm='bfsw')
    vm_alg = net.res_bus.vm_pu
    va_alg = net.res_bus.va_degree

    assert np.allclose(vm_nr, vm_alg)
    assert np.allclose(va_nr, va_alg)


@pytest.mark.xfail(reason="unknown")
def test_bsfw_algorithm_multi_net():
    net = example_simple()
    add_grid_connection(net, vn_kv=110., zone="second")

    runpp(net)
    vm_nr = copy.copy(net.res_bus.vm_pu)
    va_nr = copy.copy(net.res_bus.va_degree)

    runpp(net, algorithm='bfsw')
    vm_alg = net.res_bus.vm_pu
    va_alg = net.res_bus.va_degree

    assert np.allclose(vm_nr, vm_alg)
    assert np.allclose(va_nr, va_alg)


def test_bsfw_algorithm_with_trafo_shift_and_voltage_angles():
    net = example_simple()
    net["trafo"].loc[:, "shift_degree"] = 180.

    runpp(net, calculate_voltage_angles=True)
    vm_nr = net.res_bus.vm_pu
    va_nr = net.res_bus.va_degree

    runpp(net, algorithm='bfsw', calculate_voltage_angles=True)
    vm_alg = net.res_bus.vm_pu
    va_alg = net.res_bus.va_degree
    assert np.allclose(vm_nr, vm_alg)
    assert np.allclose(va_nr, va_alg)


def test_bsfw_algorithm_with_enforce_q_lims():
    net = example_simple()
    net.ext_grid["max_q_mvar"] = [0.1]
    net.ext_grid["min_q_mvar"] = [-0.1]
    net.gen["max_q_mvar"] = [5.]
    net.gen["min_q_mvar"] = [4.]

    runpp(net, enforce_q_lims=True)
    vm_nr = net.res_bus.vm_pu
    va_nr = net.res_bus.va_degree

    runpp(net, algorithm='bfsw', enforce_q_lims=True)
    vm_alg = net.res_bus.vm_pu
    va_alg = net.res_bus.va_degree
    assert np.allclose(vm_nr, vm_alg)
    assert np.allclose(va_nr, va_alg)


def test_bsfw_algorithm_with_branch_loops():
    net = example_simple()
    create_line(net, 0, 6, length_km=2.5,
                std_type="NA2XS2Y 1x240 RM/25 12/20 kV", name="Line meshed")
    net.switch.loc[:, "closed"] = True

    runpp(net, calculate_voltage_angles="auto")
    vm_nr = net.res_bus.vm_pu
    va_nr = net.res_bus.va_degree

    runpp(net, algorithm='bfsw', calculate_voltage_angles="auto")
    vm_alg = net.res_bus.vm_pu
    va_alg = net.res_bus.va_degree
    assert np.allclose(vm_nr, vm_alg)
    assert np.allclose(va_nr, va_alg)


@pytest.mark.slow
def test_pypower_algorithms_iter():
    alg_to_test = ['fdbx', 'fdxb', 'gs']
    for alg in alg_to_test:
        for net in result_test_network_generator(skip_test_impedance=True):
            try:
                runpp_with_consistency_checks(net, enforce_q_lims=True, algorithm=alg, calculate_voltage_angles="auto")
                runpp_with_consistency_checks(net, enforce_q_lims=False, algorithm=alg, calculate_voltage_angles="auto")
            except (AssertionError):
                raise UserWarning("Consistency Error after adding %s" % net.last_added_case)
            except(LoadflowNotConverged):
                raise UserWarning("Power flow did not converge after adding %s" %
                                  net.last_added_case)


@pytest.mark.xfail
def test_zip_loads_gridcal():
    # Tests newton power flow considering zip loads against GridCal's pf result

    # Results used for benchmarking are obtained using GridCal with the following code:
    # from GridCal.grid.CalculationEngine import *
    #
    # np.set_printoptions(precision=4)
    # grid = MultiCircuit()
    #
    # # Add buses
    # bus1 = Bus('Bus 1', vnom=20)
    #
    # bus1.controlled_generators.append(ControlledGenerator('Slack Generator', voltage_module=1.0))
    # grid.add_bus(bus1)
    #
    # bus2 = Bus('Bus 2', vnom=20)
    # bus2.loads.append(Load('load 2',
    #                        power=0.2 * complex(40, 20),
    #                        impedance=1 / (0.40 * (40. - 20.j)),
    #                        current=np.conj(0.40 * (40. + 20.j)) / (20 * np.sqrt(3)),
    #                        ))
    # grid.add_bus(bus2)
    #
    # bus3 = Bus('Bus 3', vnom=20)
    # bus3.loads.append(Load('load 3', power=complex(25, 15)))
    # grid.add_bus(bus3)
    #
    # bus4 = Bus('Bus 4', vnom=20)
    # bus4.loads.append(Load('load 4', power=complex(40, 20)))
    # grid.add_bus(bus4)
    #
    # bus5 = Bus('Bus 5', vnom=20)
    # bus5.loads.append(Load('load 5', power=complex(50, 20)))
    # grid.add_bus(bus5)
    #
    # # add branches (Lines in this case)
    # grid.add_branch(Branch(bus1, bus2, 'line 1-2', r=0.05, x=0.11, b=0.02))
    #
    # grid.add_branch(Branch(bus1, bus3, 'line 1-3', r=0.05, x=0.11, b=0.02))
    #
    # grid.add_branch(Branch(bus1, bus5, 'line 1-5', r=0.03, x=0.08, b=0.02))
    #
    # grid.add_branch(Branch(bus2, bus3, 'line 2-3', r=0.04, x=0.09, b=0.02))
    #
    # grid.add_branch(Branch(bus2, bus5, 'line 2-5', r=0.04, x=0.09, b=0.02))
    #
    # grid.add_branch(Branch(bus3, bus4, 'line 3-4', r=0.06, x=0.13, b=0.03))
    #
    # grid.add_branch(Branch(bus4, bus5, 'line 4-5', r=0.04, x=0.09, b=0.02))
    #
    # grid.compile()
    #
    # print('Ybus:\n', grid.circuits[0].power_flow_input.Ybus.todense())
    #
    # options = PowerFlowOptions(SolverType.NR, robust=False)
    # power_flow = PowerFlow(grid, options)
    # power_flow.run()
    #
    # print('\n\n', grid.name)
    # print('\t|V|:', abs(grid.power_flow_results.voltage))
    # print('\tVang:', np.rad2deg(np.angle(grid.power_flow_results.voltage)))

    vm_pu_gridcal = np.array([1., 0.9566486349, 0.9555640318, 0.9340468428, 0.9540542172])
    va_degree_gridcal = np.array([0., -2.3717973886, -2.345654238, -3.6303651197, -2.6713716569])

    Ybus_gridcal = np.array(
        [[10.9589041096 - 25.9973972603j, -3.4246575342 + 7.5342465753j,
          -3.4246575342 + 7.5342465753j,
          0.0000000000 + 0.j, -4.1095890411 + 10.9589041096j],
         [-3.4246575342 + 7.5342465753j, 11.8320802147 - 26.1409476063j,
          -4.1237113402 + 9.2783505155j,
          0.0000000000 + 0.j, -4.1237113402 + 9.2783505155j],
         [-3.4246575342 + 7.5342465753j, -4.1237113402 + 9.2783505155j,
          10.4751981427 - 23.1190605054j,
          -2.9268292683 + 6.3414634146j, 0.0000000000 + 0.j],
         [0.0000000000 + 0.j, 0.0000000000 + 0.j, -2.9268292683 + 6.3414634146j,
          7.0505406085 - 15.5948139301j,
          -4.1237113402 + 9.2783505155j],
         [-4.1095890411 + 10.9589041096j, -4.1237113402 + 9.2783505155j, 0.0000000000 + 0.j,
          -4.1237113402 + 9.2783505155j, 12.3570117215 - 29.4856051405j]])

    losses_gridcal = 4.69773448916 - 2.710430515j

    abs_path = os.path.join(pp_dir, 'networks', 'power_system_test_case_jsons',
                            'case5_demo_gridcal.json')
    net = from_json(abs_path)

    runpp(net, voltage_depend_loads=True, recycle=None)

    # Test Ybus matrix
    Ybus_pp = net["_ppc"]['internal']['Ybus'].todense()
    bus_ord = net["_pd2ppc_lookups"]["bus"]
    Ybus_pp = Ybus_pp[bus_ord, :][:, bus_ord]

    assert np.allclose(Ybus_pp, Ybus_gridcal)

    # Test Results
    assert np.allclose(net.res_bus.vm_pu, vm_pu_gridcal)
    assert np.allclose(net.res_bus.va_degree, va_degree_gridcal)

    # Test losses
    losses_pp = net.res_bus.p_mw.sum() + 1.j * net.res_bus.q_mvar.sum()
    assert np.isclose(losses_gridcal, - losses_pp / 1.e3)

    # Test bfsw algorithm
    runpp(net, voltage_depend_loads=True, algorithm='bfsw')
    assert np.allclose(net.res_bus.vm_pu, vm_pu_gridcal)
    assert np.allclose(net.res_bus.va_degree, va_degree_gridcal)


def test_zip_loads_consistency(**kwargs):
    net = four_loads_with_branches_out()
    net.load['const_i_p_percent'] = 40
    net.load['const_i_q_percent'] = 40
    net.load['const_z_p_percent'] = 40
    net.load['const_z_q_percent'] = 40

    assert runpp_with_consistency_checks(net, **kwargs)


def test_zip_loads_pf_algorithms():
    net = four_loads_with_branches_out()
    net.load['const_i_p_percent'] = 40
    net.load['const_i_q_percent'] = 40
    net.load['const_z_p_percent'] = 40
    net.load['const_z_q_percent'] = 40

    alg_to_test = ['bfsw']
    for alg in alg_to_test:
        runpp(net, algorithm='nr')
        vm_nr = net.res_bus.vm_pu
        va_nr = net.res_bus.va_degree

        runpp(net, algorithm=alg)
        vm_alg = net.res_bus.vm_pu
        va_alg = net.res_bus.va_degree

        assert np.allclose(vm_nr, vm_alg, rtol=1e-6)
        assert np.allclose(va_nr.values, va_alg.values, rtol=1e-4)


def test_zip_loads_with_voltage_angles():
    net = pandapowerNet(name="test_zip_loads_with_voltage_angles")
    b1 = create_bus(net, vn_kv=1.)
    b2 = create_bus(net, vn_kv=1.)
    create_ext_grid(net, b1)
    create_line_from_parameters(net, b1, b2, length_km=1, r_ohm_per_km=0.3,
                                x_ohm_per_km=0.3, c_nf_per_km=10, max_i_ka=1)
    create_load(net, b2, p_mw=0.002, const_z_p_percent=0, const_z_q_percent=0,
                const_i_p_percent=100, const_i_q_percent=100)

    set_user_pf_options(net, calculate_voltage_angles=True, init='dc')

    runpp(net)

    res_load = net.res_load.copy()
    net.ext_grid.va_degree = 100

    runpp(net)

    assert np.allclose(net.res_load.values, res_load.values)


def test_zip_loads_out_of_service():
    # example from https://github.com/e2nIEE/pandapower/issues/1504

    # test net
    net = pandapowerNet(name="test_zip_loads_out_of_service")
    bus1 = create_bus(net, vn_kv=20., name="Bus 1")
    bus2 = create_bus(net, vn_kv=0.4, name="Bus 2")
    bus3 = create_bus(net, vn_kv=0.4, name="Bus 3")

    # create bus elements
    create_ext_grid(net, bus=bus1, vm_pu=1.02, name="Grid Connection")
    create_load(net, bus=bus3, p_mw=0.100, q_mvar=0.05, name="Load",
                const_i_p_percent=0, const_i_q_percent=0, const_z_p_percent=0, const_z_q_percent=0)

    # create branch elements
    create_transformer(net, hv_bus=bus1, lv_bus=bus2,
                       std_type="0.4 MVA 20/0.4 kV", name="Trafo")
    create_line(net, from_bus=bus2, to_bus=bus3, length_km=0.1,
                std_type="NAYY 4x50 SE", name="Line")

    net1 = copy.deepcopy(net)
    oos_load = create_load(
        net1, bus=bus3, p_mw=0.100, q_mvar=0.05, in_service=False,
        const_i_p_percent=0, const_i_q_percent=0, const_z_p_percent=100, const_z_q_percent=100)

    runpp(net, tolerance_mva=1e-8)
    runpp(net1, tolerance_mva=1e-8)
    assert np.allclose(net1.res_load.loc[oos_load].fillna(0), 0)
    net1.res_load = net1.res_load.drop(oos_load)
    assert nets_equal(net, net1, check_only_results=True)

def test_zip_loads_mixed_voltage_dependencies():
    net = pandapowerNet(name="test_zip_loads_mixed_voltage_dependencies")

    bus1 = create_bus(net, vn_kv=110., name="Bus 1")
    bus2 = create_bus(net, vn_kv=110., name="Bus 2")

    # create bus elements
    create_ext_grid(net, bus=bus1, vm_pu=1.00, name="Grid Connection")
    create_load(net, bus=bus2, p_mw=50, q_mvar=100, name="Load",
                const_i_p_percent=0, const_i_q_percent=0, 
                const_z_p_percent=0, const_z_q_percent=0)
    create_line(net, from_bus=bus1, to_bus=bus2, length_km=50,
                std_type="N2XS(FL)2Y 1x120 RM/35 64/110 kV", name="Line")
    
    const_i_p_percent = [0, 0, 0, 0, 100, 35]
    const_i_q_percent = [0, 0, 0, 100, 0, 7]
    const_z_p_percent = [0, 0, 100, 0, 0, 45]
    const_z_q_percent = [0, 100, 0, 0, 0, 13]
    # results from PowerFactory 2024
    res_load_p_mw = [50, 50, 40.62979, 50, 44.905873, 44.066154]
    res_load_q_mvar = [100, 82.701731, 100, 90.279771, 100, 96.877779]
    res_bus_vm_pu = [0.894107, 0.909405, 0.901441, 0.902798, 0.898117, 0.901571]
    
    for c_i_p, c_i_q, c_z_p, c_z_q, res_load_p, res_load_q, res_bus_v in zip(const_i_p_percent, const_i_q_percent, 
                                                                             const_z_p_percent, const_z_q_percent, 
                                                                             res_load_p_mw, res_load_q_mvar, 
                                                                             res_bus_vm_pu):
        net.load.const_i_p_percent.at[0] = c_i_p
        net.load.const_i_q_percent.at[0] = c_i_q
        net.load.const_z_p_percent.at[0] = c_z_p
        net.load.const_z_q_percent.at[0] = c_z_q

        runpp(net, tolerance_mva=1e-6)
        
        assert np.allclose(net.res_load.p_mw.at[0], res_load_p)
        assert np.allclose(net.res_load.q_mvar.at[0], res_load_q)
        assert np.allclose(net.res_bus.vm_pu.at[1], res_bus_v)

def test_invalid_zip_percentage_sum():
    net = pandapowerNet(name="test_invalid_zip_percentage_sum")
    create_bus(net, 20.0)
    create_ext_grid(net, 0)
    create_load(net, 0, p_mw=1.0, q_mvar=0.5)

    err_msg = "const_z_p_percent + const_i_p_percent need to be less or equal to 100%! The same applies to const_z_q_percent + const_i_q_percent!"

    with pytest.raises(ValueError, match=re.escape(err_msg)):
        net.load.const_z_p_percent.at[0] = 60
        net.load.const_i_p_percent.at[0] = 50
        net.load.const_z_q_percent.at[0] = 30
        net.load.const_i_q_percent.at[0] = 20
        runpp(net, voltage_depend_loads=True)

    with pytest.raises(ValueError, match=re.escape(err_msg)):
        net.load.const_z_q_percent.at[0] = 60
        net.load.const_i_q_percent.at[0] = 50 
        runpp(net, voltage_depend_loads=True)
    
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        net.load.const_z_p_percent.at[0] = 30
        net.load.const_i_p_percent.at[0] = 20 
        runpp(net, voltage_depend_loads=True)

def test_xward_buses():
    """
    Issue: xward elements create dummy buses for the load flow, that are cleaned up afterwards.
    However, if the load flow does not converge, those buses end up staying in the net and don't get
    removed. This can potentially lead to thousands of dummy buses in net.
    """
    net = pandapowerNet(name="test_xward_buses")
    bus_sl = create_bus(net, 110, name='ExtGrid')
    create_ext_grid(net, bus_sl, vm_pu=1)
    bus_x = create_bus(net, 110, name='XWARD')
    create_xward(net, bus_x, 0, 0, 0, 0, 0, 10, 1.1)
    iid = create_impedance(net, bus_sl, bus_x, 0.2, 0.2, 1e3)

    bus_num1 = len(net.bus)

    runpp(net)

    bus_num2 = len(net.bus)

    assert bus_num1 == bus_num2

    # now - make sure that the loadflow doesn't converge:
    net.impedance.at[iid, 'rft_pu'] = 1
    create_load(net, bus_x, 1e6, 0)
    with pytest.raises(LoadflowNotConverged):
        # here the load flow doesn't converge and there is an extra bus in net
        runpp(net)

    bus_num3 = len(net.bus)
    assert bus_num3 == bus_num1


def test_pvpq_lookup():
    net = pandapowerNet(name="test_pvpq_lookup")

    b1 = create_bus(net, vn_kv=0.4, index=4)
    b2 = create_bus(net, vn_kv=0.4, index=2)
    b3 = create_bus(net, vn_kv=0.4, index=3)

    create_gen(net, b1, p_mw=0.01, vm_pu=0.4)
    create_load(net, b2, p_mw=0.01)
    create_ext_grid(net, b3)

    create_line(net, from_bus=b1, to_bus=b2, length_km=0.5, std_type="NAYY 4x120 SE")
    create_line(net, from_bus=b1, to_bus=b3, length_km=0.5, std_type="NAYY 4x120 SE")
    net_numba = copy.deepcopy(net)
    runpp(net_numba, numba=True)
    runpp(net, numba=False)

    assert nets_equal(net, net_numba)


def test_get_internal():
    net = example_simple()
    # for Newton raphson
    runpp(net)
    J_intern = net._ppc["internal"]["J"]

    ppc = net._ppc
    V_mag = ppc["bus"][:, 7][:-2]
    V_ang = ppc["bus"][:, 8][:-2]
    V = V_mag * np.exp(1j * V_ang / 180 * np.pi)

    # Get stored Ybus in ppc
    Ybus = ppc["internal"]["Ybus"]

    _, ppci = _pd2ppc(net)
    baseMVA, bus, gen, branch, svc, tcsc, ssc, vsc, ref, pv, pq, _, _, V0, _ = _get_pf_variables_from_ppci(ppci)

    pvpq = np.r_[pv, pq]
    dist_slack = False
    slack_weights = np.zeros(shape=V.shape)
    slack_weights[ref] = 1

    J = _create_J_without_numba(Ybus, V, ref, pvpq, pq, slack_weights=slack_weights, dist_slack=dist_slack)

    assert np.allclose(J.toarray(), J_intern.toarray(), atol=1e-4, rtol=0)
    # get J for all other algorithms


def test_Ybus_format():
    net = example_simple()
    runpp(net)
    _, ppci = _pd2ppc(net)

    Ybus, Yf, Yt = makeYbus_pypower(ppci["baseMVA"], ppci["bus"], ppci["branch"])
    for Y in (Ybus, Yf, Yt):
        assert Y.has_sorted_indices
        assert Y.has_canonical_format

    if numba_installed:
        Ybus, Yf, Yt = makeYbus_numba(ppci["baseMVA"], ppci["bus"], ppci["branch"])
        for Y in (Ybus, Yf, Yt):
            assert Y.has_sorted_indices
            assert Y.has_canonical_format


def test_storage_pf():
    net = pandapowerNet(name="test_storage_pf")

    b1 = create_bus(net, vn_kv=0.4)
    b2 = create_bus(net, vn_kv=0.4)

    create_line(net, b1, b2, length_km=5, std_type="NAYY 4x50 SE")

    create_ext_grid(net, b2)
    create_load(net, b1, p_mw=0.010)
    create_sgen(net, b1, p_mw=0.010)

    # test generator behaviour
    create_storage(net, b1, p_mw=-0.010, max_e_mwh=0.010)
    create_sgen(net, b1, p_mw=0.010, in_service=False)

    res_gen_beh = runpp_with_consistency_checks(net)
    res_ll_stor = net["res_line"].loading_percent.iloc[0]

    net["storage"].loc[0, 'in_service'] = False
    net["sgen"].loc[1, 'in_service'] = True

    runpp_with_consistency_checks(net)
    res_ll_sgen = net["res_line"].loading_percent.iloc[0]

    assert np.isclose(res_ll_stor, res_ll_sgen)

    # test load behaviour
    create_load(net, b1, p_mw=0.01, in_service=False)
    net["storage"].loc[0, 'in_service'] = True
    net["storage"].loc[0, 'p_mw'] = 0.01
    net["sgen"].loc[1, 'in_service'] = False

    res_load_beh = runpp_with_consistency_checks(net)
    res_ll_stor = net["res_line"].loading_percent.iloc[0]

    net["storage"].loc[0, 'in_service'] = False
    net["load"].loc[1, 'in_service'] = True

    runpp_with_consistency_checks(net)
    res_ll_load = net["res_line"].loading_percent.iloc[0]

    assert np.isclose(res_ll_stor, res_ll_load)

    assert res_gen_beh and res_load_beh


def test_add_element_and_init_results():
    net = simple_four_bus_system()
    runpp(net, init="flat", calculate_voltage_angles="auto")
    create_bus(net, vn_kv=20.)
    create_line(net, from_bus=2, to_bus=3, length_km=1, name="new line" + str(1),
                std_type="NAYY 4x150 SE")
    try:
        runpp(net, init="results", calculate_voltage_angles="auto")
        assert False
    except UserWarning:
        pass


def test_pp_initialization():
    net = pandapowerNet(name="test_pp_initialization")

    b1 = create_bus(net, vn_kv=0.4)
    b2 = create_bus(net, vn_kv=0.4)

    create_ext_grid(net, b1, vm_pu=0.7)
    create_line(net, b1, b2, 0.5, std_type="NAYY 4x50 SE", index=4)
    create_load(net, b2, p_mw=0.01)

    runpp(net, init_va_degree="flat", init_vm_pu=1.02)
    assert net._ppc["iterations"] == 5

    runpp(net, init_va_degree="dc", init_vm_pu=0.8)
    assert net._ppc["iterations"] == 4

    runpp(net, init_va_degree="flat", init_vm_pu=np.array([0.75, 0.7]))
    assert net._ppc["iterations"] == 3

    runpp(net, init_va_degree="dc", init_vm_pu=[0.75, 0.7])
    assert net._ppc["iterations"] == 3

    runpp(net, init_va_degree="flat", init_vm_pu="auto")
    assert net._ppc["iterations"] == 3

    runpp(net, init_va_degree="dc")
    assert net._ppc["iterations"] == 3


def test_equal_indices_res():
    # tests if res_bus indices of are the same as the ones in bus.
    # If this is not the case and you init from results, the PF will fail
    net = pandapowerNet(name="test_equal_indices_res")

    b1 = create_bus(net, vn_kv=10., index=3)
    b2 = create_bus(net, vn_kv=0.4, index=1)
    b3 = create_bus(net, vn_kv=0.4, index=2)

    create_ext_grid(net, b1)
    create_transformer(net, b1, b2, std_type="0.63 MVA 20/0.4 kV")
    create_line(net, b2, b3, 0.5, std_type="NAYY 4x50 SE", index=4)
    create_load(net, b3, p_mw=0.010)
    runpp(net)
    net["bus"] = net["bus"].sort_index()
    try:
        # This should raise a UserWarning since index has changed!!
        runpp(net, init_vm_pu="results", init_va_degree="results")
        assert False
    except UserWarning:
        pass


def test_ext_grid_and_gen_at_one_bus(**kwargs):
    net = pandapowerNet(name="test_ext_grid_and_gen_at_one_bus")
    b1 = create_bus(net, vn_kv=110)
    b2 = create_bus(net, vn_kv=110)
    create_ext_grid(net, b1, vm_pu=1.01)
    create_line(net, b1, b2, 1., std_type="305-AL1/39-ST1A 110.0")
    create_load(net, bus=b2, p_mw=3.5, q_mvar=1)

    runpp_with_consistency_checks(net, **kwargs)
    q = net.res_ext_grid.q_mvar.sum()

    ##create two gens at the slack bus
    g1 = create_gen(net, b1, vm_pu=1.01, p_mw=1)
    g2 = create_gen(net, b1, vm_pu=1.01, p_mw=1)
    runpp_with_consistency_checks(net, **kwargs)

    # all the reactive power previously provided by the ext_grid is now provided by the generators
    assert np.isclose(net.res_ext_grid.q_mvar.values, 0)
    assert np.isclose(net.res_gen.q_mvar.sum(), q)
    # since no Q-limits were set, reactive power is distributed equally to both generators
    assert np.isclose(net.res_gen.q_mvar.at[g1], net.res_gen.q_mvar.at[g2])

    # set reactive power limits at the generators
    net.gen["max_q_mvar"] = [0.1, 0.01]
    net.gen["min_q_mvar"] = [-0.1, -0.01]
    runpp_with_consistency_checks(net, **kwargs)
    # g1 now has 10 times the reactive power of g2 in accordance with the different Q ranges
    assert np.isclose(net.res_gen.q_mvar.at[g1], net.res_gen.q_mvar.at[g2] * 10)
    # all the reactive power is still provided by the generators, because Q-lims are not enforced
    assert np.allclose(net.res_ext_grid.q_mvar.values, [0])
    assert np.isclose(net.res_gen.q_mvar.sum(), q)

    # now enforce Q-lims
    runpp_with_consistency_checks(net, enforce_q_lims=True, **kwargs)
    # both generators are at there lower limit with regard to the reactive power
    assert np.allclose(net.res_gen.q_mvar.values, net.gen.max_q_mvar.values)
    # the total reactive power remains unchanged, but the rest of the power is now provided by the ext_grid
    assert np.isclose(net.res_gen.q_mvar.sum() + net.res_ext_grid.q_mvar.sum(), q)

    # second ext_grid at the slack bus
    create_ext_grid(net, b1, vm_pu=1.01)
    runpp_with_consistency_checks(net, enforce_q_lims=False, **kwargs)
    # gens still have the correct active power
    assert np.allclose(net.gen.p_mw.values, net.res_gen.p_mw.values)
    # slack active power is evenly distributed to both ext_grids
    assert np.isclose(net.res_ext_grid.p_mw.values[0], net.res_ext_grid.p_mw.values[1])

    # q limits at the ext_grids are not enforced
    net.ext_grid["max_q_mvar"] = [0.1, 0.01]
    net.ext_grid["min_q_mvar"] = [-0.1, -0.01]
    runpp_with_consistency_checks(net, enforce_q_lims=True, **kwargs)
    assert net.res_ext_grid.q_mvar.values[0] > net.ext_grid.max_q_mvar.values[0]
    assert np.allclose(net.res_gen.q_mvar.values, net.gen.max_q_mvar.values)


def two_ext_grids_at_one_bus():
    net = pandapowerNet(name="two_ext_grids_at_one_bus")
    b1 = create_bus(net, vn_kv=110, index=3)
    b2 = create_bus(net, vn_kv=110, index=5)
    create_ext_grid(net, b1, vm_pu=1.01, index=2)
    create_line(net, b1, b2, 1., std_type="305-AL1/39-ST1A 110.0")
    create_load(net, bus=b2, p_mw=3.5, q_mvar=1)
    create_gen(net, b1, vm_pu=1.01, p_mw=1)
    runpp_with_consistency_checks(net)
    assert net.converged

    # connect second ext_grid to b1 with different angle but out of service
    eg2 = create_ext_grid(net, b1, vm_pu=1.01, va_degree=20, index=5, in_service=False)
    runpp_with_consistency_checks(net)  # power flow still converges since eg2 is out of service
    assert net.converged

    # error is raised after eg2 is set in service
    net.ext_grid.at[eg2, "in_service"] = True
    with pytest.raises(UserWarning):
        runpp(net)

    #  error is also raised when eg2 is connected to first ext_grid through bus-bus switch
    b3 = create_bus(net, vn_kv=110)
    create_switch(net, b1, b3, et="b")
    net.ext_grid.at[eg2, "bus"] = b3
    with pytest.raises(UserWarning):
        runpp(net)

    # no error is raised when voltage angles are not calculated
    runpp_with_consistency_checks(net, calculate_voltage_angles=False)
    assert net.converged

    # same angle but different voltage magnitude also raises an error
    net.ext_grid.at[eg2, "vm_pu"] = 1.02
    net.ext_grid.at[eg2, "va_degree"] = 0
    with pytest.raises(UserWarning):
        runpp(net)


def test_dc_with_ext_grid_at_one_bus():
    net = pandapowerNet(name="test_dc_with_ext_grid_at_one_bus")
    b1 = create_bus(net, vn_kv=110)
    b2 = create_bus(net, vn_kv=110)

    create_ext_grid(net, b1, vm_pu=1.01)
    create_ext_grid(net, b2, vm_pu=1.01)

    create_dcline(net, from_bus=b1, to_bus=b2, p_mw=10,
                  loss_percent=0, loss_mw=0, vm_from_pu=1.01, vm_to_pu=1.01)

    create_sgen(net, b1, p_mw=10)
    create_load(net, b2, p_mw=10)

    runpp_with_consistency_checks(net)
    assert np.allclose(net.res_ext_grid.p_mw.values, [0, 0])


def test_no_branches():
    net = pandapowerNet(name="test_no_branches")
    create_buses(net, 3, 110)
    create_ext_grid(net, 0)
    create_sgen(net, 1, 10)
    create_load(net, 2, 10)
    runpp(net)
    assert np.isclose(net.res_ext_grid.p_mw.at[0], 0.0)
    assert np.isclose(net.res_ext_grid.q_mvar.at[0], 0.0)
    assert np.isclose(net.res_bus.vm_pu.at[0], 1.0)
    assert np.isclose(net.res_bus.va_degree.at[0], 0.0)
    assert np.all(pd.isnull(net.res_bus.loc[[1, 2], 'vm_pu']))


def test_only_ref_buses():
    net = pandapowerNet(name="test_only_ref_buses")
    create_buses(net, nr_buses=2, vn_kv=1)
    create_line_from_parameters(net, from_bus=0, to_bus=1, length_km=1,
                                r_ohm_per_km=1, x_ohm_per_km=1,
                                c_nf_per_km=0, max_i_ka=1)
    create_ext_grid(net, bus=0, vm_pu=1)
    create_ext_grid(net, bus=1, vm_pu=1)
    runpp(net)
    assert np.all(np.isclose(net.res_bus.vm_pu, 1.0))
    assert np.all(np.isclose(net.res_bus.va_degree, 0.0))
    assert np.isclose(net.res_line.loading_percent.at[0], 0.0)
    assert np.all(np.isclose(net.res_ext_grid.p_mw, 0.0))
    assert np.all(np.isclose(net.res_ext_grid.q_mvar, 0.0))

    net.ext_grid.at[1, "vm_pu"] = 0.5
    runpp(net)
    assert np.allclose(net.res_ext_grid.p_mw.values, np.array([0.25, -0.125]), rtol=0, atol=1e-12)
    assert np.allclose(net.res_ext_grid.q_mvar.values, np.array([0.25, -0.125]), rtol=0, atol=1e-12)
    assert abs(net.res_line.p_from_mw.at[0] - 0.25) < 1e-12
    assert abs(net.res_line.q_from_mvar.at[0] - 0.25) < 1e-12
    assert abs(net.res_line.p_to_mw.at[0] + 0.125) < 1e-12
    assert abs(net.res_line.q_to_mvar.at[0] + 0.125) < 1e-12
    assert abs(net.res_line.i_ka.at[0] - 0.20412415) < 1e-6


def test_init_results_without_results():
    # should switch to "auto" mode and not fail
    net = example_multivoltage()
    reset_results(net)
    runpp(net, init="results")
    assert net.converged
    reset_results(net)
    runpp(net, init_vm_pu="results")
    assert net.converged
    reset_results(net)
    runpp(net, init_va_degree="results")
    assert net.converged
    reset_results(net)
    runpp(net, init_va_degree="results", init_vm_pu="results")
    assert net.converged


def test_init_results():
    net = pandapowerNet(name="test_init_results")
    add_test_line(net)  # line network with switch at to bus
    assert_init_results(net)
    net.switch.at[0, "bus"] = 0  # switch at from bus
    assert_init_results(net)

    add_test_trafo(net)  # trafo network with switch at lv bus
    assert_init_results(net)
    net.switch.at[0, "bus"] = 7  # switch at hv bus
    assert_init_results(net)

    add_test_xward(net)  # xward with internal node
    assert_init_results(net)
    add_test_trafo3w(net)  # trafo3w with internal node
    assert_init_results(net)
    t3idx = net.trafo3w.index[0]
    t3_switch = create_switch(net, bus=net.trafo3w.hv_bus.at[t3idx],
                              element=t3idx, et="t3", closed=False)  # trafo3w switch at hv side
    assert_init_results(net)
    net.switch.at[t3_switch, "bus"] = net.trafo3w.mv_bus.at[t3idx]  # trafo3w switch at mv side
    assert_init_results(net)
    net.switch.at[t3_switch, "bus"] = net.trafo3w.lv_bus.at[t3idx]  # trafo3w switch at lv side
    assert_init_results(net)


def assert_init_results(net):
    runpp(net, init="auto")
    assert net._ppc["iterations"] > 0
    runpp(net, init="results")
    assert net._ppc["iterations"] == 0


def test_wye_delta():
    net = pandapowerNet(name="test_wye_delta")
    create_bus(net, vn_kv=110)
    create_buses(net, nr_buses=4, vn_kv=20)
    trafo = create_transformer(net, hv_bus=0, lv_bus=1, std_type='25 MVA 110/20 kV')
    create_line(net, 1, 2, length_km=2.0, std_type="NAYY 4x50 SE")
    create_line(net, 2, 3, length_km=6.0, std_type="NAYY 4x50 SE")
    create_line(net, 3, 4, length_km=10.0, std_type="NAYY 4x50 SE")
    create_ext_grid(net, 0)
    create_load(net, 4, p_mw=0.1)
    create_sgen(net, 2, p_mw=4.)
    create_sgen(net, 3, p_mw=4.)

    runpp(net, trafo_model="pi")
    f, t = net._pd2ppc_lookups["branch"]["trafo"]
    assert np.isclose(net.res_trafo.p_hv_mw.at[trafo], -7.560996, rtol=1e-7)
    assert np.allclose(net._ppc["branch"][f:t, [BR_R, BR_X, BR_B, BR_G]].flatten(),
                       np.array([0.0001640, 0.0047972, -0.0105000, 0.014]),
                       rtol=1e-7)

    runpp(net, trafo_model="t")
    assert np.allclose(net._ppc["branch"][f:t, [BR_R, BR_X, BR_B, BR_G]].flatten(),
                       np.array([0.00016392, 0.00479726, -0.01050009, 0.01399964]))
    assert np.isclose(net.res_trafo.p_hv_mw.at[trafo], -7.561001, rtol=1e-7)


def test_line_temperature():
    net = four_loads_with_branches_out()
    r_init = net.line.r_ohm_per_km.values.copy()

    # r_ohm_per_km is not in line results by default
    runpp(net)
    v_init = net.res_bus.vm_pu.values.copy()
    va_init = net.res_bus.va_degree.values.copy()
    assert "r_ohm_per_km" not in net.res_line.columns

    # no temperature adjustment performed if not explicitly set in options/arguments to runpp
    net.line["temperature_degree_celsius"] = 20
    runpp(net)
    assert "r_ohm_per_km" not in net.res_line.columns
    assert np.allclose(net.res_bus.vm_pu, v_init, rtol=0, atol=1e-16)
    assert np.allclose(net.res_bus.va_degree, va_init, rtol=0, atol=1e-16)

    # argument in runpp is considered
    runpp(net, consider_line_temperature=True)
    assert "r_ohm_per_km" in net.res_line.columns
    assert np.allclose(net.res_line.r_ohm_per_km, r_init, rtol=0, atol=1e-16)
    assert np.allclose(net.res_bus.vm_pu, v_init, rtol=0, atol=1e-16)
    assert np.allclose(net.res_bus.va_degree, va_init, rtol=0, atol=1e-16)

    # check results of r adjustment, check that user_pf_options works, alpha is 4e-3 by default
    t = np.arange(0, 80, 10)
    net.line.temperature_degree_celsius = t
    set_user_pf_options(net, consider_line_temperature=True)
    runpp(net)
    alpha = 4e-3
    r_temp = r_init * (1 + alpha * (t - 20))
    assert np.allclose(net.res_line.r_ohm_per_km, r_temp, rtol=0, atol=1e-16)
    assert not np.allclose(net.res_bus.vm_pu, v_init, rtol=0, atol=1e-4)
    assert not np.allclose(net.res_bus.va_degree, va_init, rtol=0, atol=1e-2)

    # check reults with user-defined alpha
    alpha = np.arange(3e-3, 5e-3, 2.5e-4)
    net.line['alpha'] = alpha
    runpp(net)
    r_temp = r_init * (1 + alpha * (t - 20))
    assert np.allclose(net.res_line.r_ohm_per_km, r_temp, rtol=0, atol=1e-16)
    assert not np.allclose(net.res_bus.vm_pu, v_init, rtol=0, atol=1e-4)
    assert not np.allclose(net.res_bus.va_degree, va_init, rtol=0, atol=1e-2)

    # not anymore in net if not considered
    set_user_pf_options(net, overwrite=True)
    runpp(net)
    assert np.allclose(net.res_bus.vm_pu, v_init, rtol=0, atol=1e-16)
    assert np.allclose(net.res_bus.va_degree, va_init, rtol=0, atol=1e-16)
    assert "r_ohm_per_km" not in net.res_line.columns


def test_results_for_line_temperature():
    net = pandapowerNet(name="test_results_for_line_temperature")
    create_bus(net, 0.4)
    create_buses(net, 2, 0.4)

    create_ext_grid(net, 0)
    create_load(net, 1, 5e-3, 10e-3)
    create_load(net, 2, 5e-3, 10e-3)

    create_line_from_parameters(net, 0, 1, 0.5, 0.642, 0.083, 210, 1, alpha=0.00403)
    create_line_from_parameters(net, 1, 2, 0.5, 0.642, 0.083, 210, 1, alpha=0.00403)

    vm_res_20 = [1, 0.9727288676, 0.95937328755]
    va_res_20 = [0, 2.2103403814, 3.3622612327]
    vm_res_80 = [1, 0.96677572771, 0.95062498477]
    va_res_80 = [0, 2.7993156134, 4.2714451629]

    runpp(net)

    assert np.allclose(net.res_bus.vm_pu, vm_res_20, rtol=0, atol=1e-6)
    assert np.allclose(net.res_bus.va_degree, va_res_20, rtol=0, atol=1e-6)

    net.line["temperature_degree_celsius"] = 80
    set_user_pf_options(net, consider_line_temperature=True)
    runpp(net)

    assert np.allclose(net.res_bus.vm_pu, vm_res_80, rtol=0, atol=1e-6)
    assert np.allclose(net.res_bus.va_degree, va_res_80, rtol=0, atol=1e-6)


def test_tap_dependent_impedance():
    net = pandapowerNet(name="test_tap_dependent_impedance")
    b1, b2, l1 = add_grid_connection(net)
    b3 = create_bus(net, vn_kv=0.4)
    create_transformer(net, hv_bus=b2, lv_bus=b3, std_type="0.25 MVA 20/0.4 kV")
    create_transformer(net, hv_bus=b2, lv_bus=b3, std_type="0.25 MVA 20/0.4 kV")

    b4 = create_bus(net, vn_kv=0.9)
    b5 = create_bus(net, vn_kv=0.4)
    create_transformer3w_from_parameters(net, hv_bus=b2, mv_bus=b4, lv_bus=b5,
                                         vn_hv_kv=20., vn_mv_kv=0.9, vn_lv_kv=0.45, sn_hv_mva=0.6, sn_mv_mva=0.5,
                                         sn_lv_mva=0.4, vk_hv_percent=1., vk_mv_percent=1., vk_lv_percent=1.,
                                         vkr_hv_percent=0.3, vkr_mv_percent=0.3, vkr_lv_percent=0.3,
                                         pfe_kw=0.2, i0_percent=0.3, tap_neutral=0., tap_side='hv',
                                         tap_pos=0, tap_step_percent=0., tap_min=-2, tap_max=2,
                                         tap_changer_type="Ratio")

    net_backup = copy.deepcopy(net)

    net["trafo_characteristic_table"] = pd.DataFrame(
        {'id_characteristic': [0, 0, 0, 0, 0], 'step': [-2, -1, 0, 1, 2], 'voltage_ratio': [0.95, 0.975, 1, 1.025, 1.05],
         'angle_deg': [0, 0, 0, 0, 0], 'vk_percent': [5.5, 5.8, 6, 6.2, 6.5],
         'vkr_percent': [1.4, 1.42, 1.44, 1.46, 1.48], 'vk_hv_percent': np.nan, 'vkr_hv_percent': np.nan,
         'vk_mv_percent': np.nan, 'vkr_mv_percent': np.nan, 'vk_lv_percent': np.nan, 'vkr_lv_percent': np.nan})
    net.trafo['id_characteristic_table'].at[0] = 0
    net.trafo['tap_dependency_table'].at[0] = True
    net.trafo['tap_dependency_table'].at[1] = False

    new_rows = pd.DataFrame(
        {'id_characteristic': [1, 1, 1, 1, 1], 'step': [-2, -1, 0, 1, 2], 'voltage_ratio': [1, 1, 1, 1, 1],
         'angle_deg': [0, 0, 0, 0, 0], 'vk_hv_percent': [0.95, 0.98, 1, 1.02, 1.05],
         'vkr_hv_percent': [0.3, 0.3, 0.3, 0.3, 0.3], 'vk_mv_percent': [1, 1, 1, 1, 1],
         'vkr_mv_percent': [0.3, 0.3, 0.3, 0.3, 0.3], 'vk_lv_percent': [1, 1, 1, 1, 1],
         'vkr_lv_percent': [0.3, 0.3, 0.3, 0.3, 0.3]})
    net["trafo_characteristic_table"] = pd.concat([net["trafo_characteristic_table"], new_rows], ignore_index=True)
    net.trafo3w['id_characteristic_table'].at[0] = 1
    net.trafo3w['tap_dependency_table'].at[0] = True

    runpp(net)
    runpp(net_backup)
    assert_res_equal(net, net_backup)

    net.trafo.at[0, "tap_pos"] = 2
    net_backup.trafo.at[0, "tap_pos"] = 2
    net_backup.trafo.at[0, "vk_percent"] = 6.5
    net_backup.trafo.at[0, "vkr_percent"] = 1.48

    runpp(net)
    runpp(net_backup)
    assert_res_equal(net, net_backup)

    net.trafo3w.at[0, "tap_pos"] = 2
    net_backup.trafo3w.at[0, "tap_pos"] = 2
    net_backup.trafo3w.at[0, "vk_hv_percent"] = 1.05

    runpp(net)
    runpp(net_backup)
    assert_res_equal(net, net_backup)

def test_tap_table_order():
    net = pandapowerNet(name="test_tap_table_order")
    b1, b2, l1 = add_grid_connection(net)
    b3 = create_bus(net, vn_kv=0.4)
    b4 = create_bus(net, vn_kv=0.4)
    create_transformer(net, hv_bus=b2, lv_bus=b3, std_type="0.25 MVA 20/0.4 kV")
    create_transformer(net, hv_bus=b2, lv_bus=b4, std_type="0.25 MVA 20/0.4 kV")

    b5 = create_bus(net, vn_kv=0.9)
    b6 = create_bus(net, vn_kv=0.4)
    create_transformer3w_from_parameters(net, hv_bus=b2, mv_bus=b5, lv_bus=b6,
                                            vn_hv_kv=20., vn_mv_kv=0.9, vn_lv_kv=0.45, sn_hv_mva=0.6, sn_mv_mva=0.5,
                                            sn_lv_mva=0.4, vk_hv_percent=1., vk_mv_percent=1., vk_lv_percent=1.,
                                            vkr_hv_percent=0.3, vkr_mv_percent=0.3, vkr_lv_percent=0.3,
                                            pfe_kw=0.2, i0_percent=0.3, tap_neutral=0., tap_side='hv',
                                            tap_pos=0, tap_step_percent=0., tap_min=-2, tap_max=2,
                                            tap_changer_type="Ratio")

    net["trafo_characteristic_table"] = pd.DataFrame(
        {'id_characteristic': [0, 0, 0, 0, 0], 'step': [-2, -1, 0, 1, 2], 'voltage_ratio': [1, 1, 1, 1, 1],
         'angle_deg': [0, 0, 0, 0, 0], 'vk_percent': np.nan, 'vkr_percent': np.nan,
         'vk_hv_percent': [0.95, 0.98, 1, 1.02, 1.05], 'vkr_hv_percent': [0.3, 0.3, 0.3, 0.3, 0.3],
         'vk_mv_percent': [1, 1, 1, 1, 1], 'vkr_mv_percent': [0.3, 0.3, 0.3, 0.3, 0.3],
         'vk_lv_percent': [1, 1, 1, 1, 1], 'vkr_lv_percent': [0.3, 0.3, 0.3, 0.3, 0.3]})
    net.trafo3w['id_characteristic_table'].at[0] = 0
    net.trafo3w['tap_dependency_table'].at[0] = True

    new_rows = pd.DataFrame(
        {'id_characteristic': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2], 'step': [-2, -1, 0, 1, 2, -2, -1, 0, 1, 2],
         'voltage_ratio': [0.95, 0.975, 1, 1.025, 1.05, 0.98, 0.99, 1, 1.01, 1.02],
         'angle_deg': [0, 0, 0, 0, 0, -2, -1, 0, 1, 2], 'vk_percent': [5.5, 5.8, 6, 6.2, 6.5, 5.5, 5.8, 6, 6.2, 6.5],
         'vkr_percent': [1.4, 1.42, 1.44, 1.46, 1.48, 1.4, 1.42, 1.44, 1.46, 1.48], 'vk_hv_percent': np.nan, 'vkr_hv_percent': np.nan,
         'vk_mv_percent': np.nan, 'vkr_mv_percent': np.nan, 'vk_lv_percent': np.nan, 'vkr_lv_percent': np.nan})
    net["trafo_characteristic_table"] = pd.concat([net["trafo_characteristic_table"], new_rows], ignore_index=True)
    net.trafo['id_characteristic_table'].at[0] = 2
    net.trafo['id_characteristic_table'].at[1] = 1
    net.trafo['tap_dependency_table'].at[0] = True
    net.trafo['tap_dependency_table'].at[1] = True
    net.trafo['tap_pos'].at[0] = -2
    net.trafo['tap_pos'].at[1] = 2

    runpp(net)

    tol = 0.001

    assert net.converged == True
    assert np.isclose(net.res_bus.loc[4, 'va_degree'], -1 * (150 - 2 + 0.03717), 0, tol, False)
    assert np.isclose(net.res_bus.loc[5, 'va_degree'], -1 * (150 - 0 + 0.03810), 0, tol, False)
    assert np.isclose(net.res_bus.loc[4, 'vm_pu'], 1.03144595, tol, False)
    assert np.isclose(net.res_bus.loc[5, 'vm_pu'], 0.96268165, tol, False)

def test_shunt_step_dependency_warning():
    net = simple_test_net_shunt_control()
    net["shunt_characteristic_table"] = pd.DataFrame(
        {'id_characteristic': [0, 0, 0, 0, 0], 'step': [1, 2, 3, 4, 5], 'q_mvar': [-25, -50, -75, -100, -125],
         'p_mw': [0, 0, 0, 0, 0]})
    create_shunt(net, bus=0, q_mvar=-10, p_mw=20, step=1, max_step=5)
    net.shunt.step_dependency_table.at[0] = True
    net.shunt.step.at[0] = 1

    with pytest.raises(UserWarning):
        runpp(net)


@pytest.mark.skipif(not lightsim2grid_available, reason="lightsim2grid is not installed")
def test_lightsim2grid():
    # test several nets
    for net in result_test_network_generator():
        try:
            net_ref = copy.deepcopy(net)
            runpp(net_ref, lightsim2grid=False)
            runpp_with_consistency_checks(net, lightsim2grid=True)
            assert_res_equal(net, net_ref)
        except AssertionError:
            raise UserWarning("Consistency Error after adding %s" % net.last_added_case)
        except LoadflowNotConverged:
            raise UserWarning("Power flow did not converge after adding %s" % net.last_added_case)
        except NotImplementedError as err:
            assert len(net.ext_grid) > 1
            assert "multiple ext_grids are found" in str(err)


@pytest.mark.skipif(not lightsim2grid_available, reason="lightsim2grid is not installed")
def test_lightsim2grid_case118():
    net = case118()
    net_ref = copy.deepcopy(net)
    runpp(net_ref, lightsim2grid=False)
    runpp_with_consistency_checks(net, lightsim2grid=True)
    assert_res_equal(net, net_ref)


@pytest.mark.skipif(not lightsim2grid_available, reason="lightsim2grid is not installed")
def test_lightsim2grid_zip():
    # voltage dependent loads are not implemented in lightsim2grid
    with pytest.raises(NotImplementedError, match="voltage-dependent loads"):
        test_zip_loads_consistency(lightsim2grid=True)


@pytest.mark.skipif(not lightsim2grid_available, reason="lightsim2grid is not installed")
def test_lightsim2grid_qlims():
    test_minimal_net(lightsim2grid=True, enforce_q_lims=True)


@pytest.mark.skipif(not lightsim2grid_available, reason="lightsim2grid is not installed")
def test_lightsim2grid_extgrid():
    # multiple ext grids not implemented
    with pytest.raises(NotImplementedError, match="multiple ext_grids"):
        test_ext_grid_and_gen_at_one_bus(lightsim2grid=True)


@pytest.mark.skipif(lightsim2grid_available, reason="only relevant if lightsim2grid is not installed")
def test_lightsim2grid_option_basic():
    net = simple_four_bus_system()
    runpp(net)
    assert not net._options["lightsim2grid"]


@pytest.mark.skipif(not lightsim2grid_available, reason="lightsim2grid is not installed")
def test_lightsim2grid_option():
    # basic usage
    net = simple_four_bus_system()
    runpp(net)
    assert net._options["lightsim2grid"]

    runpp(net, lightsim2grid=False)
    assert not net._options["lightsim2grid"]

    # missing algorithm
    runpp(net, algorithm="gs")
    assert not net._options["lightsim2grid"]

    with pytest.raises(NotImplementedError, match=r"algorithm"):
        runpp(net, algorithm="gs", lightsim2grid=True)

    # voltage-dependent loads
    net.load["const_z_p_percent"] = 100.
    net.load["const_z_q_percent"] = 100.
    runpp(net, voltage_depend_loads=True)
    assert not net._options["lightsim2grid"]

    with pytest.raises(NotImplementedError, match=r"voltage-dependent loads"):
        runpp(net, voltage_depend_loads=True, lightsim2grid=True)

    with pytest.raises(NotImplementedError, match=r"voltage-dependent loads"):
        runpp(net, lightsim2grid=True)
    net.load.const_z_p_percent = 0
    net.load.const_z_q_percent = 0

    # multiple slacks
    xg = create_ext_grid(net, 1, 1.)
    runpp(net)
    assert not net._options["lightsim2grid"]

    with pytest.raises(NotImplementedError, match=r"multiple ext_grids"):
        runpp(net, lightsim2grid=True)

    net.ext_grid.at[xg, 'in_service'] = False
    runpp(net)
    assert net._options["lightsim2grid"]

    create_gen(net, 1, 0, 1., slack=True)
    with pytest.raises(NotImplementedError, match=r"multiple ext_grids"):
        runpp(net, lightsim2grid=True)

    runpp(net, distributed_slack=True)
    assert net._options["lightsim2grid"]


def test_at_isolated_bus():
    net = pandapowerNet(name="test_at_isolated_bus")
    create_buses(net, 4, 110)
    create_ext_grid(net, 0)

    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)

    create_gen(net, 3, 0, vm_pu=0, in_service=False)

    runpp(net)
    assert np.isclose(net._options["init_vm_pu"], 1.0)


def test_shunt_with_missing_vn_kv():
    net = pandapowerNet(name="test_shunt_with_missing_vn_kv")
    create_buses(net, 2, 110)
    create_ext_grid(net, 0)
    create_line_from_parameters(net, 0, 1, 30, 0.0487, 0.13823, 160, 0.664)

    create_shunt(net, 1, 10)
    net.shunt.vn_kv=np.nan

    runpp(net)


def _test_net_for_q_capability_curve():
    net = pandapowerNet(name="_test_net_for_q_capability_curve")

    bus1 = create_bus(net, name="bus1", vn_kv=20., type="b", min_vm_pu=0.96, max_vm_pu=1.02)
    bus2 = create_bus(net, name="bus2", vn_kv=110., type="b", min_vm_pu=0.96, max_vm_pu=1.02)
    bus3 = create_bus(net, name="bus3", vn_kv=110., type="b", min_vm_pu=0.96, max_vm_pu=1.02)

    create_ext_grid(net, bus3, name="External Grid", vm_pu=1.0, va_degree=0.0,max_p_mw=100000, min_p_mw=0,
                    min_q_mvar=-300, max_q_mvar=300, s_sc_max_mva=10000, s_sc_min_mva=8000, rx_max=0.1, rx_min=0.1)
    # create lines
    create_line_from_parameters(net, bus2, bus3, length_km=10,df=1,max_loading_percent=100,vn_kv=110,max_i_ka=0.74,type="ol",
                   r_ohm_per_km=0.0949, x_ohm_per_km =0.38, c_nf_per_km=0.0092, name="Line")
    # create load
    create_load(net, bus3, p_mw=198, q_mvar=500, name="Load", vm_pu=1.0 )

    # create transformer
    create_transformer_from_parameters(
        net, bus2, bus1, name="110kV/20kV transformer", parallel=1, max_loading_percent=100, sn_mva=210, vn_hv_kv=110,
        vn_lv_kv=20, vk_percent=12.5, vkr_percent=0.01904762, vk0_percent=10, vkr0_percent=0, shift_degree=330,
        vector_group="YNd11", i0_percent= 0.26, pfe_kw=0,si0_hv_partial=0.5
    )

    create_gen(net, bus1, p_mw=100, sn_mva=255.0, scaling=1.0, type="Hydro",
                 cos_phi=0.8, pg_percent=0.0, vn_kv=19.0, vm_pu=1.0) #,min_q_mvar=-255, max_q_mvar=255,  min_p_mw=-331.01001, max_p_mw=331.01001)
    return net


def test_q_capability_curve():
    net = _test_net_for_q_capability_curve()
    runpp(net)

    net.gen.loc[0,"max_q_mvar"] = 50.0
    net.gen.loc[0, "min_q_mvar"] = -3
    runpp(net, enforce_q_lims=True)
    assert net.res_gen.q_mvar.loc[0] == -3
    assert net.res_gen.p_mw.loc[0] == 100

    # create q characteristics table
    net["q_capability_curve_table"] = pd.DataFrame(
        {'id_q_capability_curve': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         'p_mw': [-331.01001, -298.0, -198.0, -66.2000, -0.1, 0, 0.1, 66.200, 100, 198.00, 298.00, 331.01001],
         'q_min_mvar': [-0.0100, -134.0099, -265.01001, -323.01001, -323.01001, -323.01001, -323.01001, -323.01001,
                        0, -265.01001, -134.00999, -0.01000],
         'q_max_mvar': [0.01000, 134.00999,  228.00999, 257.01001, 261.01001, 261.01001, 261.01001, 257.01001, 30, 40,
                        134.0099, 0.01]})

    net.gen.id_q_capability_characteristic.at[0] = 0
    net.gen['curve_style'] = "straightLineYValues"

    # Add q_capability_characteristic for one gen based on q_capability_curve_table
    create_q_capability_characteristics_object(net)
    runpp(net, enforce_q_lims=True)
    assert net.res_gen.q_mvar.loc[0] == 0
    assert net.res_gen.p_mw.loc[0] == 100

def test_q_capability_curve_for_sgen():
    net = _test_net_for_q_capability_curve()
    drop_elements(net, 'gen', 0)
    create_sgen(net, 0, p_mw=198, sn_mva=255.0, scaling=1.0, type="Hydro", cos_phi=0.8, pg_percent=0.0, vn_kv=19.0,
                vm_pu=1.0, min_q_mvar=-255, max_q_mvar=255, controllable=True, min_p_mw=-0.03, max_p_mw=0)
    net.ext_grid["controllable"] = True
    create_poly_cost(net, 0, "sgen", cp1_eur_per_mw=0.1)
    create_poly_cost(net, 0, "ext_grid", cp1_eur_per_mw=-0.1)
    net.trafo.loc[0, "parallel"] = 3
    net.line.parallel = 5
    runopp(net, init='pf', calculate_voltage_angles=False)
    assert max(net.res_bus.vm_pu) < 1.02
    assert min(net.res_bus.vm_pu) > 0.96
    assert net.res_sgen.q_mvar.loc[0] < 255
    assert net.res_sgen.q_mvar.loc[0] > -255
    logger.info("------------general limit------------------")
    logger.info("res_sgen:\n%s" % net.res_sgen)
    logger.info("res_bus.vm_pu: \n%s" % net.res_bus)

    logger.info("------------given maximum limit------------------\n")
    net.sgen.loc[0,"max_q_mvar"] = 261.01001
    net.sgen.loc[0, "min_q_mvar"] = -323.01001
    runopp(net, init='pf', calculate_voltage_angles=False)
    logger.info("res_sgen:\n%s" % net.res_sgen)
    logger.info("res_bus.vm_pu: \n%s" % net.res_bus)
    assert max(net.res_bus.vm_pu) < 1.02
    assert min(net.res_bus.vm_pu) > 0.96
    assert net.res_sgen.q_mvar.loc[0] < 261.01001
    assert net.res_sgen.q_mvar.loc[0] > -323.01001

    logger.info("------------curve limit------------------\n")
    # create q characteristics table
    net["q_capability_curve_table"] = pd.DataFrame(
        {'id_q_capability_curve': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         'p_mw': [-331.01001, -298.0, -198.0, -66.2000, -0.1, 0, 0.1, 66.200, 198.00, 298.00, 331.01001],
         'q_min_mvar': [-0.0100, -134.0099, -265.01001, -323.01001, -323.01001, -323.01001, -323.01001, -323.01001,
                        -265.01001, -134.00999, -0.01000],
         'q_max_mvar': [0.01000, 134.00999,  228.00999, 257.01001, 261.01001, 261.01001, 261.01001, 257.01001, 218.0099945068,
                        134.0099, 0.01]})

    net.sgen.id_q_capability_characteristic.at[0] = 0
    net.sgen['curve_style'] = "straightLineYValues"
    create_q_capability_characteristics_object(net)

    runopp(net, init='pf', calculate_voltage_angles=False)
    logger.info("res_sgen:\n%s" % net.res_sgen)
    logger.info("res_bus.vm_pu: \n%s" % net.res_bus)
    assert max(net.res_bus.vm_pu) < 1.02
    assert min(net.res_bus.vm_pu) > 0.96
    assert net.res_sgen.q_mvar.loc[0] < 218.0099945068
    assert net.res_sgen.q_mvar.loc[0] > -265.01001

if __name__ == "__main__":
    pytest.main([__file__, "-xs"])

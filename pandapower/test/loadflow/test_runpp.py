# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

import pytest
import pandas as pd
import numpy as np

import pandapower as pp
from pandapower.test.toolbox import add_grid_connection, create_test_line, assert_net_equal
from pandapower.test.loadflow.result_test_network_generator import result_test_network_generator
from pandapower.test.consistency_checks import runpp_with_consistency_checks
from pandapower.test.loadflow.result_test_network_generator import add_test_oos_bus_with_is_element
from pandapower.auxiliary import _check_connectivity, _add_ppc_options
from pandapower.pd2ppc import _pd2ppc
from pandapower.powerflow import _select_is_elements
from pandapower.networks import create_cigre_network_mv, four_loads_with_branches_out, example_simple
from pandapower.powerflow import LoadflowNotConverged


def test_runpp_init():
    net = pp.create_empty_network()
    b1, b2, l1 = add_grid_connection(net)
    b3 = pp.create_bus(net, vn_kv=0.4)
    tidx = pp.create_transformer(net, hv_bus=b2, lv_bus=b3, std_type="0.25 MVA 20/0.4 kV")
    net.trafo.shift_degree.at[tidx] = 70
    pp.runpp(net)
    va = net.res_bus.va_degree.at[4]
    pp.runpp(net, calculate_voltage_angles=True, init="dc")
    assert np.allclose(va - net.trafo.shift_degree.at[tidx], net.res_bus.va_degree.at[4])
    pp.runpp(net, calculate_voltage_angles=True, init="results")
    assert np.allclose(va - net.trafo.shift_degree.at[tidx], net.res_bus.va_degree.at[4])


def test_runpp_init_auxiliary_buses():
    net = pp.create_empty_network()
    b1, b2, l1 = add_grid_connection(net, vn_kv=110.)
    b3 = pp.create_bus(net, vn_kv=20.)
    b4 = pp.create_bus(net, vn_kv=10.)
    tidx = pp.create_transformer3w(net, b2, b3, b4, std_type='63/25/38 MVA 110/20/10 kV')
    pp.create_load(net, b3, 5e3)
    pp.create_load(net, b4, 5e3)
    pp.create_xward(net, b4, 1000, 1000, 1000, 1000, 0.1, 0.1, 1.0)
    net.trafo3w.shift_lv_degree.at[tidx] = 120
    net.trafo3w.shift_mv_degree.at[tidx] = 80
    pp.runpp(net)
    va = net.res_bus.va_degree.at[b2]
    pp.runpp(net, calculate_voltage_angles=True, init="dc")
    assert np.allclose(va - net.trafo3w.shift_mv_degree.at[tidx], net.res_bus.va_degree.at[b3],
                       atol=2)
    assert np.allclose(va - net.trafo3w.shift_lv_degree.at[tidx], net.res_bus.va_degree.at[b4],
                       atol=2)
    pp.runpp(net, calculate_voltage_angles=True, init="results")
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


def test_bus_bus_switches():
    net = pp.create_empty_network()
    add_grid_connection(net)
    for _u in range(4):
        pp.create_bus(net, vn_kv=.4)
    pp.create_load(net, 5, p_kw=10)
    pp.create_switch(net, 3, 6, et="b")
    pp.create_switch(net, 4, 5, et="b")
    pp.create_switch(net, 6, 5, et="b")
    pp.create_switch(net, 0, 7, et="b")
    create_test_line(net, 4, 7)
    pp.create_load(net, 4, p_kw=10)
    pp.runpp(net)
    assert net.res_bus.vm_pu.at[3] == net.res_bus.vm_pu.at[4] == net.res_bus.vm_pu.at[5] == \
           net.res_bus.vm_pu.at[6]
    assert net.res_bus.vm_pu.at[0] == net.res_bus.vm_pu.at[7]

    net.bus.in_service.at[5] = False
    pp.runpp(net)
    assert net.res_bus.vm_pu.at[3] == net.res_bus.vm_pu.at[6]
    assert net.res_bus.vm_pu.at[0] == net.res_bus.vm_pu.at[7]
    assert pd.isnull(net.res_bus.vm_pu.at[5])
    assert net.res_bus.vm_pu.at[6] != net.res_bus.vm_pu.at[4]


def test_two_open_switches():
    net = pp.create_empty_network()
    b1, b2, l1 = add_grid_connection(net)
    b3 = pp.create_bus(net, vn_kv=20.)
    l2 = create_test_line(net, b2, b3)
    create_test_line(net, b3, b1)
    pp.create_switch(net, b2, l2, et="l", closed=False)
    pp.create_switch(net, b3, l2, et="l", closed=False)
    pp.runpp(net)
    assert np.isnan(net.res_line.i_ka.at[l2])


def test_oos_bus():
    net = pp.create_empty_network()
    add_test_oos_bus_with_is_element(net)
    assert runpp_with_consistency_checks(net)

    #    test for pq-node result
    pp.create_shunt(net, 6, q_kvar=-800)
    assert runpp_with_consistency_checks(net)

    #   1test for pv-node result
    pp.create_gen(net, 4, p_kw=-500)
    assert runpp_with_consistency_checks(net)


def get_isolated(net):
    net._options = {}
    _add_ppc_options(net, calculate_voltage_angles=False,
                     trafo_model="t", check_connectivity=False,
                     mode="pf", copy_constraints_to_ppc=False,
                     r_switch=0.0, init="flat", enforce_q_lims=False, recycle=None)

    ppc, ppci = _pd2ppc(net)
    return _check_connectivity(ppc)


def test_connectivity_check_island_without_pv_bus():
    # Network with islands without pv bus -> all buses in island should be set out of service
    net = create_cigre_network_mv(with_der=False)
    iso_buses, iso_p, iso_q = get_isolated(net)
    assert len(iso_buses) == 0
    assert np.isclose(iso_p, 0)
    assert np.isclose(iso_q, 0)

    isolated_bus1 = pp.create_bus(net, vn_kv=20., name="isolated Bus1")
    isolated_bus2 = pp.create_bus(net, vn_kv=20., name="isolated Bus2")
    pp.create_line(net, isolated_bus2, isolated_bus1, length_km=1,
                   std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV",
                   name="IsolatedLine")
    iso_buses, iso_p, iso_q = get_isolated(net)
    assert len(iso_buses) == 2
    assert np.isclose(iso_p, 0)
    assert np.isclose(iso_q, 0)

    pp.create_load(net, isolated_bus1, p_kw=200., q_kvar=20)
    pp.create_sgen(net, isolated_bus2, p_kw=-150., q_kvar=-10)

    # with pytest.warns(UserWarning):
    iso_buses, iso_p, iso_q = get_isolated(net)
    assert len(iso_buses) == 2
    assert np.isclose(iso_p, 350)
    assert np.isclose(iso_q, 30)
    # with pytest.warns(UserWarning):
    runpp_with_consistency_checks(net, check_connectivity=True)


def test_connectivity_check_island_with_one_pv_bus():
    # Network with islands with one PV bus -> PV bus should be converted to the reference bus
    net = create_cigre_network_mv(with_der=False)
    iso_buses, iso_p, iso_q = get_isolated(net)
    assert len(iso_buses) == 0
    assert np.isclose(iso_p, 0)
    assert np.isclose(iso_q, 0)

    isolated_bus1 = pp.create_bus(net, vn_kv=20., name="isolated Bus1")
    isolated_bus2 = pp.create_bus(net, vn_kv=20., name="isolated Bus2")
    isolated_gen = pp.create_bus(net, vn_kv=20., name="isolated Gen")
    isolated_pv_bus = pp.create_gen(net, isolated_gen, p_kw=350, vm_pu=1.0, name="isolated PV bus")
    pp.create_line(net, isolated_bus2, isolated_bus1, length_km=1,
                   std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV",
                   name="IsolatedLine")
    pp.create_line(net, isolated_gen, isolated_bus1, length_km=1,
                   std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV",
                   name="IsolatedLineToGen")
    # with pytest.warns(UserWarning):
    iso_buses, iso_p, iso_q = get_isolated(net)

    # assert len(iso_buses) == 0
    # assert np.isclose(iso_p, 0)
    # assert np.isclose(iso_q, 0)
    #
    # pp.create_load(net, isolated_bus1, p_kw=200., q_kvar=20)
    # pp.create_sgen(net, isolated_bus2, p_kw=-150., q_kvar=-10)
    #
    # iso_buses, iso_p, iso_q = get_isolated(net)
    # assert len(iso_buses) == 0
    # assert np.isclose(iso_p, 0)
    # assert np.isclose(iso_q, 0)

    # with pytest.warns(UserWarning):
    runpp_with_consistency_checks(net, check_connectivity=True)


def test_connectivity_check_island_with_multiple_pv_buses():
    # Network with islands an multiple PV buses in the island ->
    # Error should be thrown since it would be random to choose just some PV bus as the reference bus
    net = create_cigre_network_mv(with_der=False)
    iso_buses, iso_p, iso_q = get_isolated(net)
    assert len(iso_buses) == 0
    assert np.isclose(iso_p, 0)
    assert np.isclose(iso_q, 0)

    isolated_bus1 = pp.create_bus(net, vn_kv=20., name="isolated Bus1")
    isolated_bus2 = pp.create_bus(net, vn_kv=20., name="isolated Bus2")
    isolated_pv_bus1 = pp.create_bus(net, vn_kv=20., name="isolated PV bus1")
    isolated_pv_bus2 = pp.create_bus(net, vn_kv=20., name="isolated PV bus2")
    pp.create_gen(net, isolated_pv_bus1, p_kw=300, vm_pu=1.0, name="isolated PV bus1")
    pp.create_gen(net, isolated_pv_bus2, p_kw=50, vm_pu=1.0, name="isolated PV bus2")

    pp.create_line(net, isolated_pv_bus1, isolated_bus1, length_km=1,
                   std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV",
                   name="IsolatedLineToGen1")
    pp.create_line(net, isolated_pv_bus2, isolated_bus2, length_km=1,
                   std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV",
                   name="IsolatedLineToGen2")
    pp.create_line(net, isolated_bus2, isolated_bus1, length_km=1,
                   std_type="N2XS(FL)2Y 1x300 RM/35 64/110 kV",
                   name="IsolatedLine")
    # ToDo with pytest.warns(UserWarning):
    iso_buses, iso_p, iso_q = get_isolated(net)


def test_makeYbus():
    # tests if makeYbus fails for nets where every bus is connected to each other
    net = pp.create_empty_network()
    b1, b2, l1 = add_grid_connection(net)

    # number of buses to create
    n_bus = 20
    bus_list = []
    # generate buses and connect them
    for _ in range(n_bus):
        bus_list.append(pp.create_bus(net, vn_kv=20.))

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


def test_test_sn_kva():
    test_net_gen1 = result_test_network_generator(sn_kva=1e3)
    test_net_gen2 = result_test_network_generator(sn_kva=2e3)
    for net1, net2 in zip(test_net_gen1, test_net_gen2):
        pp.runpp(net1)
        pp.runpp(net2)
        try:
            assert_net_equal(net1, net2)
        except:
            raise UserWarning("Result difference due to sn_kva after adding %s" % net1.last_added_case)


def test_pf_algorithms():
    alg_to_test = ['bfsw', 'fdbx', 'fdxb', 'gs']
    for alg in alg_to_test:
        net = create_cigre_network_mv(with_der=False)

        pp.runpp(net, algorithm='nr')
        vm_nr = net.res_bus.vm_pu
        va_nr = net.res_bus.va_degree

        pp.runpp(net, algorithm=alg)
        vm_alg = net.res_bus.vm_pu
        va_alg = net.res_bus.va_degree

        assert np.allclose(vm_nr, vm_alg)
        assert np.allclose(va_nr, va_alg)

        # testing with a network which contains DERs
        net = create_cigre_network_mv()

        pp.runpp(net)
        vm_nr = net.res_bus.vm_pu
        va_nr = net.res_bus.va_degree

        pp.runpp(net, algorithm=alg)
        vm_alg = net.res_bus.vm_pu
        va_alg = net.res_bus.va_degree

        assert np.allclose(vm_nr, vm_alg)
        assert np.allclose(va_nr, va_alg)

        # testing a weakly meshed network and consideration of phase-shifting transformer using calculate_voltage_angles
        net = four_loads_with_branches_out()
        # adding a line in order to create a loop
        pp.create_line(net, from_bus=8, to_bus=9, length_km=0.05, name='line9', std_type='NAYY 4x120 SE')

        pp.runpp(net, calculate_voltage_angles=True)
        vm_nr = net.res_bus.vm_pu
        va_nr = net.res_bus.va_degree

        pp.runpp(net, algorithm=alg, calculate_voltage_angles=True)
        vm_alg = net.res_bus.vm_pu
        va_alg = net.res_bus.va_degree

        assert np.allclose(vm_nr, vm_alg)
        assert np.allclose(va_nr, va_alg)

        # testing a network with PV buses
        net = example_simple()

        pp.runpp(net)
        vm_nr = net.res_bus.vm_pu
        va_nr = net.res_bus.va_degree

        pp.runpp(net, algorithm='bfsw')
        vm_alg = net.res_bus.vm_pu
        va_alg = net.res_bus.va_degree

        assert np.allclose(vm_nr, vm_alg)
        assert np.allclose(va_nr, va_alg)


if __name__ == "__main__":
    pytest.main(["test_runpp.py", "-xs"])

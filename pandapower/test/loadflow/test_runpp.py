# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import copy

import numpy as np
import pandas as pd
import pytest

import pandapower as pp
from pandapower.auxiliary import _check_connectivity, _add_ppc_options
from pandapower.networks import create_cigre_network_mv, four_loads_with_branches_out, example_simple
from pandapower.pd2ppc import _pd2ppc
from pandapower.powerflow import LoadflowNotConverged
from pandapower.toolbox import nets_equal
from pandapower.test.consistency_checks import runpp_with_consistency_checks
from pandapower.test.loadflow.result_test_network_generator import add_test_oos_bus_with_is_element, \
    result_test_network_generator
from pandapower.test.toolbox import add_grid_connection, create_test_line, assert_net_equal


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


@pytest.fixture
def bus_bus_net():
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
    return net


def test_bus_bus_switches(bus_bus_net):
    net = bus_bus_net
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


def test_bus_bus_switches_throws_exception_for_two_gens(bus_bus_net):
    "buses should not be fused if two gens are connected"
    net = bus_bus_net
    net.bus.in_service.at[5] = False
    pp.create_gen(net, 6, 10)
    pp.create_gen(net, 4, 10)
    pp.runpp(net)
    net.bus.in_service.at[5] = True
    with pytest.raises(UserWarning):
        pp.runpp(net)


@pytest.fixture
def r_switch_net():
    net = pp.create_empty_network()
    for i in range(3):
        pp.create_bus(net, vn_kv=.4)
        pp.create_load(net, i, p_kw=100)
    pp.create_ext_grid(net, 0, vm_pu=1.0)
    pp.create_line_from_parameters(net, 0, 1, 0.1, r_ohm_per_km=0.1, x_ohm_per_km=0,
                                   c_nf_per_km=0, max_i_ka=.2)
    pp.create_switch(net, 0, 2, et="b")
    return net


def test_r_switch(r_switch_net):
    net = r_switch_net
    pp.runpp(net, r_switch=0.01, numba=False)
    assert net.res_bus.vm_pu.at[1] == net.res_bus.vm_pu.at[2]


def test_r_switch_numba(r_switch_net):
    net = r_switch_net
    pp.runpp(net, r_switch=0.01, numba=True)
    assert net.res_bus.vm_pu.at[1] == net.res_bus.vm_pu.at[2]


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


def test_bsfw_algorithm():
    net = example_simple()

    pp.runpp(net)
    vm_nr = net.res_bus.vm_pu
    va_nr = net.res_bus.va_degree

    pp.runpp(net, algorithm='bfsw')
    vm_alg = net.res_bus.vm_pu
    va_alg = net.res_bus.va_degree

    assert np.allclose(vm_nr, vm_alg)
    assert np.allclose(va_nr, va_alg)


def test_pypower_algorithms_iter():
    alg_to_test = ['fdbx', 'fdxb', 'gs']
    for alg in alg_to_test:
        for net in result_test_network_generator(skip_test_impedance=True):
            try:
                runpp_with_consistency_checks(net, enforce_q_lims=True, algorithm=alg)
                runpp_with_consistency_checks(net, enforce_q_lims=False, algorithm=alg)
            except (AssertionError):
                raise UserWarning("Consistency Error after adding %s" % net.last_added_case)
            except(LoadflowNotConverged):
                raise UserWarning("Power flow did not converge after adding %s" % net.last_added_case)

def test_recycle():
    # Note: Only calls recycle functions and tests if load and gen are updated.
    # Todo: To fully test the functionality, it must be checked if the recycle methods are being called
    # or alternatively if the "non-recycle" functions are not being called.
    net = pp.create_empty_network()
    b1, b2, ln = add_grid_connection(net)
    pl = 1200
    ql = 1100
    ps = -500
    u_set = 1.0

    b3 = pp.create_bus(net, vn_kv=.4)
    pp.create_line_from_parameters(net, b2, b3, 12.2, r_ohm_per_km=0.08, x_ohm_per_km=0.12,
                                   c_nf_per_km=300, max_i_ka=.2, df=.8)
    pp.create_load(net, b3, p_kw=pl, q_kvar=ql)
    pp.create_gen(net, b2, p_kw=ps, vm_pu=u_set)

    runpp_with_consistency_checks(net, recycle=dict(_is_elements=True, ppc=True, Ybus=True))

    # copy.deepcopy(net)

    # update values
    pl = 600
    ql = 550
    ps = -250
    u_set = 0.98

    net["load"].p_kw.iloc[0] = pl
    net["load"].q_kvar.iloc[0] = ql
    net["gen"].p_kw.iloc[0] = ps
    net["gen"].vm_pu.iloc[0] = u_set

    runpp_with_consistency_checks(net, recycle=dict(_is_elements=True, ppc=True, Ybus=True))

    assert np.allclose(net.res_load.p_kw.iloc[0], pl)
    assert np.allclose(net.res_load.q_kvar.iloc[0], ql)
    assert np.allclose(net.res_gen.p_kw.iloc[0], ps)
    assert np.allclose(net.res_gen.vm_pu.iloc[0], u_set)


import os


def test_zip_loads_gridcal():
    ## Tests newton power flow considering zip loads against GridCal's pf result

    ## Results used for benchmarking are obtained using GridCal with the following code:
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
    # options = PowerFlowOptions(SolverType.NR, verbose=False, robust=False)
    # power_flow = PowerFlow(grid, options)
    # power_flow.run()
    #
    # print('\n\n', grid.name)
    # print('\t|V|:', abs(grid.power_flow_results.voltage))
    # print('\tVang:', np.rad2deg(np.angle(grid.power_flow_results.voltage)))

    vm_pu_gridcal = np.array([1., 0.9566486349, 0.9555640318, 0.9340468428, 0.9540542172])
    va_degree_gridcal = np.array([0., -2.3717973886, -2.345654238, -3.6303651197, -2.6713716569])

    Ybus_gridcal = np.array(
        [[10.9589041096 - 25.9973972603j, -3.4246575342 + 7.5342465753j, -3.4246575342 + 7.5342465753j,
          0.0000000000 + 0.j, -4.1095890411 + 10.9589041096j],
         [-3.4246575342 + 7.5342465753j, 11.8320802147 - 26.1409476063j, -4.1237113402 + 9.2783505155j,
          0.0000000000 + 0.j, -4.1237113402 + 9.2783505155j],
         [-3.4246575342 + 7.5342465753j, -4.1237113402 + 9.2783505155j, 10.4751981427 - 23.1190605054j,
          -2.9268292683 + 6.3414634146j, 0.0000000000 + 0.j],
         [0.0000000000 + 0.j, 0.0000000000 + 0.j, -2.9268292683 + 6.3414634146j, 7.0505406085 - 15.5948139301j,
          -4.1237113402 + 9.2783505155j],
         [-4.1095890411 + 10.9589041096j, -4.1237113402 + 9.2783505155j, 0.0000000000 + 0.j,
          -4.1237113402 + 9.2783505155j, 12.3570117215 - 29.4856051405j]])

    losses_gridcal = 4.69773448916 - 2.710430515j

    abs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                            'networks', 'power_system_test_case_pickles', 'case5_demo_gridcal.p')
    net = pp.from_pickle(abs_path)

    pp.runpp(net, voltage_depend_loads=True,
             recycle=dict(_is_elements=False, ppc=False, Ybus=True, bfsw=False))

    # Test Ybus matrix
    Ybus_pp = net["_ppc"]['internal']['Ybus'].todense()
    bus_ord = net["_pd2ppc_lookups"]["bus"]
    Ybus_pp = Ybus_pp[bus_ord, :][:, bus_ord]

    assert np.allclose(Ybus_pp, Ybus_gridcal)

    # Test Results
    assert np.allclose(net.res_bus.vm_pu, vm_pu_gridcal)
    assert np.allclose(net.res_bus.va_degree, va_degree_gridcal)

    # Test losses
    losses_pp = net.res_bus.p_kw.sum() + 1.j * net.res_bus.q_kvar.sum()
    assert np.isclose(losses_gridcal, - losses_pp / 1.e3)

    # Test bfsw algorithm
    pp.runpp(net, voltage_depend_loads=True, algorithm='bfsw')
    assert np.allclose(net.res_bus.vm_pu, vm_pu_gridcal)
    assert np.allclose(net.res_bus.va_degree, va_degree_gridcal)


def test_zip_loads_consistency():
    net = four_loads_with_branches_out()
    net.load['const_i_percent'] = 40
    net.load['const_z_percent'] = 40
    assert runpp_with_consistency_checks(net)


def test_zip_loads_pf_algorithms():
    net = four_loads_with_branches_out()
    net.load['const_i_percent'] = 40
    net.load['const_z_percent'] = 40

    alg_to_test = ['bfsw']
    for alg in alg_to_test:
        pp.runpp(net, algorithm='nr')
        vm_nr = net.res_bus.vm_pu
        va_nr = net.res_bus.va_degree

        pp.runpp(net, algorithm=alg)
        vm_alg = net.res_bus.vm_pu
        va_alg = net.res_bus.va_degree

        assert np.allclose(vm_nr, vm_alg)
        assert np.allclose(va_nr, va_alg)


def test_pvpq_lookup():
    net = pp.create_empty_network()

    b1 = pp.create_bus(net, vn_kv=0.4, index=4)
    b2 = pp.create_bus(net, vn_kv=0.4, index=2)
    b3 = pp.create_bus(net, vn_kv=0.4, index=3)

    g2 = pp.create_gen(net, b1, p_kw=-10, vm_pu=0.4)
    l3 = pp.create_load(net, b2, p_kw=10)
    e1 = pp.create_ext_grid(net, b3)

    pp.create_line(net, from_bus=b1, to_bus=b2, length_km=0.5, std_type="NAYY 4x120 SE")
    pp.create_line(net, from_bus=b1, to_bus=b3, length_km=0.5, std_type="NAYY 4x120 SE")
    net_numba = copy.deepcopy(net)
    pp.runpp(net_numba, numba=True)
    pp.runpp(net, numba=False)

    assert nets_equal(net, net_numba)

if __name__ == "__main__":
    pytest.main(["test_runpp.py", "-xs"])

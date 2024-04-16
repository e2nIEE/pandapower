# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pytest
import pandapower as pp
from pandapower import networks
from pandapower.control import ContinuousTapControl
from pandapower.pypower.idx_bus import PD, GS, VM
from pandapower.pypower.idx_brch import PF
from pandapower.test.helper_functions import assert_res_equal
try:
    import numba

    numba_installed = True
except ImportError:
    numba_installed = False

try:
    import pandaplan.core.pplog as logging
except ImportError:
    import logging

logger = logging.getLogger(__name__)


def small_example_grid():
    net = pp.create_empty_network()
    pp.create_buses(net, 3, 20)

    pp.create_gen(net, 0, p_mw=100, vm_pu=1, slack=True, slack_weight=1)
    # pp.create_ext_grid(net, 0)

    pp.create_load(net, 1, p_mw=100, q_mvar=100)

    pp.create_line_from_parameters(net, 0, 1, length_km=3, r_ohm_per_km=0.01, x_ohm_per_km=0.1, c_nf_per_km=0,
                                   max_i_ka=1)
    pp.create_line_from_parameters(net, 1, 2, length_km=2, r_ohm_per_km=0.01, x_ohm_per_km=0.1, c_nf_per_km=0,
                                   max_i_ka=1)
    pp.create_line_from_parameters(net, 2, 0, length_km=1, r_ohm_per_km=0.01, x_ohm_per_km=0.1, c_nf_per_km=0,
                                   max_i_ka=1)
    return net


def _get_xward_result(net):
    # here I tried getting the xward power that is relevant for the balance
    p_results = np.array([])
    internal_results = np.array([])
    ppc = net._ppc

    ft = net._pd2ppc_lookups.get('branch', dict()).get('xward', [])
    if len(ft) > 0:
        f, t = ft
        p_impedance = ppc['branch'][f:t, PF].real
    else:
        p_impedance = np.array([])

    for b, x_id in zip(net.xward.query("in_service").bus.values, net.xward.query("in_service").index.values):
        p_bus = ppc['bus'][net._pd2ppc_lookups["bus"][b], PD]
        p_shunt = ppc['bus'][net._pd2ppc_lookups["bus"][b], VM] ** 2 * net["xward"].at[x_id, "pz_mw"]
        internal_results = np.append(internal_results, p_shunt)
        connected = pp.get_connected_elements_dict(net, [b], respect_in_service=True, connected_buses=False,
                                                   connected_branch_elements=False,
                                                   connected_other_elements=False)
        for e, idx in connected.items():
            # first, count the total slack weights per bus, and obtain the variable part of the active power
            total_weight = 0
            if e in ['xward', 'ext_grid']:
                continue
            p_bus -= net[e].loc[idx, "p_mw"].values.sum()

        p_results = np.append(p_results, p_bus)

    internal_results += p_impedance

    return p_results, internal_results


def _get_losses(net):
    pl_mw = 0
    for elm in ['line', 'trafo', 'trafo3w', 'impedance']:
        pl_mw += net['res_' + elm].pl_mw.sum()
    return pl_mw


def _get_injection_consumption(net):
    _, xward_internal = _get_xward_result(net)
    total_pl_mw = _get_losses(net)
    # xward is in the consumption reference system
    # active power consumption by the internal elements of xward is not adjusted by the distributed slack calculation
    # that is why we add the active power of the internal elements of the xward here
    consumed_p_mw = total_pl_mw + \
                    net.load.query("in_service").p_mw.sum() - \
                    net.sgen.query("in_service").p_mw.sum() + \
                    xward_internal.sum()
    injected_p_mw = net.gen.query("in_service").p_mw.sum()
    # we return the xward power separately because it is also already considered in the inputs and results
    return injected_p_mw, consumed_p_mw, net.xward.query("in_service").ps_mw.sum()


def _get_slack_weights(net):
    slack_weights = np.r_[net.gen.query("in_service").slack_weight,
                          net.ext_grid.query("in_service").slack_weight,
                          net.xward.query("in_service").slack_weight]
    return slack_weights / sum(slack_weights)


def _get_inputs_results(net):
    # distributed slack calculation only adjusts the pq part of the xward active power consumption
    # that is why we only consider the active power consumption by the PQ load of the xward here
    xward_pq_res, _ = _get_xward_result(net)
    # xward is in the consumption reference system, but here the results are all assumed in the generation reference system
    inputs = np.r_[net.gen.query("in_service").p_mw,
                   np.zeros(len(net.ext_grid.query("in_service"))),
                   -net.xward.query("in_service").ps_mw]
    results = np.r_[net.res_gen[net.gen.in_service].p_mw,
                    net.res_ext_grid[net.ext_grid.in_service].p_mw,
                    -xward_pq_res]
    return inputs, results


def assert_results_correct(net, tol=1e-8):
    # first, collect slack_weights, injection and consumption, inputs and reults from net
    injected_p_mw, consumed_p_mw, consumed_xward_p_mw = _get_injection_consumption(net)
    input_p_mw, result_p_mw = _get_inputs_results(net)
    slack_weights = _get_slack_weights(net)

    # assert power balance is correct
    assert abs(result_p_mw.sum() - consumed_p_mw) < tol, "power balance is wrong"
    # assert results are according to the distributed slack formula
    assert np.allclose(input_p_mw - (injected_p_mw - consumed_p_mw - consumed_xward_p_mw) * slack_weights, result_p_mw,
                       atol=tol, rtol=0), "distributed slack weights formula has a wrong result"


def check_xward_results(net, tol=1e-9):
    xward_pq_res, xward_internal = _get_xward_result(net)
    assert np.allclose(xward_pq_res + xward_internal, net.res_xward.p_mw, atol=tol, rtol=0)


def run_and_assert_numba(net, **kwargs):
    if numba_installed:
        net_temp = net.deepcopy()
        pp.runpp(net_temp, distributed_slack=True, numba=False, **kwargs)
        pp.runpp(net, distributed_slack=True, **kwargs)
        assert_res_equal(net, net_temp)
    else:
        pp.runpp(net, distributed_slack=True, numba=False, **kwargs)


def test_get_xward_result():
    # here we test the helper function that calculates the internal and PQ load results separately
    # it separates the results of other node ellments at the same bus, but only works for 1 xward at a bus
    net = small_example_grid()
    pp.create_xward(net, 2, 100, 0, 0, 0, 0.02, 0.2, 1)
    pp.create_load(net, 2, 50, 0, 0, 0, 0.02, 0.2, 1)
    pp.runpp(net)
    check_xward_results(net)


def test_numba():
    net = small_example_grid()
    # if no slack_weight is given for ext_grid, 1 is assumed, because normally
    # ext_grids are responsible to take the slack power
    pp.create_gen(net, 2, 200, 1., slack_weight=2)

    run_and_assert_numba(net)
    assert_results_correct(net)


def test_small_example():
    net = small_example_grid()
    # if no slack_weight is given for ext_grid, 1 is assumed, because normally
    # ext_grids are responsible to take the slack power
    net.gen["slack_weight"] = 1

    net2 = net.deepcopy()

    pp.runpp(net, distributed_slack=True, numba=False)

    pp.runpp(net2, distributed_slack=False)

    assert_res_equal(net, net2)
    assert_results_correct(net)


def test_two_gens():
    # a three-bus example with 3 lines in a ring, 2 gens and 1 load
    net = small_example_grid()
    pp.create_gen(net, 2, 200, 1., slack_weight=2)

    run_and_assert_numba(net)
    assert_results_correct(net)

    # check bus voltages
    assert np.allclose(net.res_bus.vm_pu, [1, 0.96542683532, 1], rtol=0, atol=1e-9)
    assert np.allclose(net.res_bus.va_degree, [0, -1.5537971468, 0.08132969181], rtol=0, atol=1e-6)
    # check gen p and q
    assert np.allclose(net.res_gen.p_mw, [33.548233528, 67.096467056], rtol=0, atol=1e-6)
    assert np.allclose(net.res_gen.q_mvar, [43.22006789, 63.226937869], rtol=0, atol=1e-6)
    # check line currents
    assert np.allclose(net.res_line.i_ka, [1.6717271156, 2.5572842138, 0.16309292919], rtol=0, atol=1e-9)


def test_three_gens():
    net = small_example_grid()
    pp.create_gen(net, 1, 200, 1., slack_weight=2)
    pp.create_gen(net, 2, 200, 1., slack_weight=2)

    run_and_assert_numba(net)
    assert_results_correct(net)


def test_gen_xward():
    # here testing for numba only
    net = small_example_grid()
    # note: xward is in the consumption reference system, so positive ps_mw stands for consumption
    pp.create_xward(net, 2, 200, 0, 0, 0, 0.02, 0.2, 1, slack_weight=2)
    run_and_assert_numba(net)
    # xward behavior is a bit different due to the shunt component and the impedance component of the xward
    # so we check the results by hand for the case when shunt values are != 0
    assert_results_correct(net)
    check_xward_results(net)


def test_xward_pz_mw():
    # here testing for numba only
    # for now, not implemented and should raise an error
    net = small_example_grid()
    # pp.create_xward(net, 2, 0, 0, 0, 0, 0.02, 0.2, 1, slack_weight=2)
    pp.create_xward(net, 2, 200, 20, 10, 1, 0.02, 0.2, 1, slack_weight=2)
    run_and_assert_numba(net)
    # xward behavior is a bit different due to the shunt component of the xward
    # so we check the results by hand for the case when shunt values are != 0
    assert_results_correct(net)
    check_xward_results(net)


def test_xward_manually():
    net_1 = small_example_grid()
    # pp.create_xward(net, 2, 0, 0, 0, 0, 0.02, 0.2, 1, slack_weight=2)
    pp.create_xward(net_1, 2, 200, 20, 10, 1, 0.02, 0.2, 1, slack_weight=2)
    run_and_assert_numba(net_1)
    slack_power = (net_1.res_gen.p_mw.at[0] - net_1.gen.p_mw.at[0]) * 3  # factor 3 since gen has
    # slack_weight==1 and xward has slack_weight==2

    # xward behavior is a bit different due to the shunt component of the xward
    # so we check the results by hand for the case when shunt values are != 0
    net = small_example_grid()
    pp.create_bus(net, 20)
    pp.create_load(net, 2, 200, 20)
    pp.create_shunt(net, 2, 1, 10)
    pp.create_gen(net, 3, 0, 1, slack_weight=2)
    pp.create_line_from_parameters(net, 2, 3, 1, 0.02, 0.2, 0, 1)

    net.load.at[1, 'p_mw'] = net_1._ppc['bus'][net_1.xward.bus.at[0], PD]
    assert np.isclose(200 - net.load.at[1, 'p_mw'], slack_power * 2 / 3, rtol=0, atol=1e-6)
    pp.runpp(net)

    assert np.isclose(net_1.res_gen.at[0, 'p_mw'], net.res_gen.at[0, 'p_mw'], rtol=0, atol=1e-6)
    assert np.isclose(net_1.res_gen.at[0, 'q_mvar'], net.res_gen.at[0, 'q_mvar'], rtol=0, atol=1e-6)
    assert np.isclose(net_1.res_bus.at[2, 'p_mw'], net.res_bus.at[2, 'p_mw'] + net.res_line.at[3, 'p_from_mw'], rtol=0,
                      atol=1e-6)
    assert np.allclose(net_1.res_bus.vm_pu, net.res_bus.loc[0:2, 'vm_pu'], rtol=0, atol=1e-6)
    assert np.allclose(net_1.res_bus.va_degree, net.res_bus.loc[0:2, 'va_degree'], rtol=0, atol=1e-6)


def test_ext_grid():
    net = small_example_grid()
    net.gen.in_service = False
    pp.create_ext_grid(net, 0, slack_weight=1)
    pp.create_ext_grid(net, 2, slack_weight=2)

    run_and_assert_numba(net)
    assert_results_correct(net)


def test_gen_ext_grid():
    net = small_example_grid()
    pp.create_ext_grid(net, 2, slack_weight=2)

    run_and_assert_numba(net)
    assert_results_correct(net)


def test_pvgen_ext_grid():
    # now test the behavior if gen is not slack
    net = small_example_grid()
    pp.create_ext_grid(net, 2, slack_weight=2)
    net.gen.slack = False

    pp.runpp(net, distributed_slack=True, numba=numba_installed)

    run_and_assert_numba(net)
    assert_results_correct(net)


def test_same_bus():
    net = small_example_grid()
    pp.create_ext_grid(net, 0, slack_weight=2)

    run_and_assert_numba(net)
    assert_results_correct(net)


def test_gen_oos():
    net = small_example_grid()
    pp.create_gen(net, 2, 200, 1., slack_weight=2, in_service=False)

    run_and_assert_numba(net)
    assert_results_correct(net)


def test_ext_grid_oos():
    net = small_example_grid()
    pp.create_ext_grid(net, 0, slack_weight=2, in_service=False)

    run_and_assert_numba(net)
    assert_results_correct(net)


def test_xward_oos():
    net = small_example_grid()
    pp.create_xward(net, 2, 200, 20, 10, 1, 0.02, 0.2, 1, slack_weight=2, in_service=False)

    run_and_assert_numba(net)
    assert_results_correct(net)


def test_only_xward():
    net = pp.create_empty_network()
    pp.create_bus(net, 110)
    pp.create_ext_grid(net, 0, vm_pu=1.05, slack_weight=2)
    pp.create_xward(net, 0, 200, 20, 10, 1, 0.02, 0.2, 1, slack_weight=2)
    with pytest.raises(NotImplementedError):
        pp.runpp(net, distributed_slack=True)


def test_xward_gen_same_bus():
    net = small_example_grid()
    pp.create_gen(net, 2, 200, 1., slack_weight=2)
    pp.create_xward(net, 2, 200, 20, 10, 1, 0.02, 0.2, 1, slack_weight=4)
    with pytest.raises(NotImplementedError):
        pp.runpp(net, distributed_slack=True)


def test_separate_zones():
    net = small_example_grid()
    b1, b2 = pp.create_buses(net, 2, 110)
    pp.create_line_from_parameters(net, b1, b2, length_km=1, r_ohm_per_km=0.01, x_ohm_per_km=0.1, c_nf_per_km=0,
                                   max_i_ka=1)
    pp.create_ext_grid(net, b1)
    pp.create_load(net, b2, 100)

    # distributed slack not implemented for separate zones
    with pytest.raises(NotImplementedError):
        pp.runpp(net, distributed_slack=True, numba=False)


def case9_simplified():
    net = pp.create_empty_network()
    pp.create_buses(net, 9, vn_kv=345.)
    lines = [[0, 3], [3, 4], [4, 5], [2, 5], [5, 6], [6, 7], [7, 1], [7, 8], [8, 3]]

    for i, (fb, tb) in enumerate(lines):
        pp.create_line_from_parameters(net, fb, tb, 1, 20, 100, 0, 1)

    pp.create_gen(net, 0, 0, slack=True, slack_weight=1)
    pp.create_gen(net, 1, 163, slack_weight=1)
    pp.create_gen(net, 2, 85, slack_weight=1)

    pp.create_load(net, 4, 90, 30)
    pp.create_load(net, 6, 100, 35)
    pp.create_load(net, 8, 125, 50)
    return net


def test_case9():
    """
    basic test with ext_grid + gen, scaling != 1, slack_weight sum = 1
    """
    tol_mw = 1e-6
    net = networks.case9()
    # net = case9_simplified()

    # set slack_weight (distributed slack participation factor)
    net.ext_grid['slack_weight'] = 1 / 3
    net.gen['slack_weight'] = 1 / 3
    # todo: is it clearer to consider scaling or to ignore it? right now is ignored
    # net.gen["scaling"] = [0.8, 0.7]
    net.gen["scaling"] = [1, 1]

    # # set ext_grid dispatched active power
    # net.ext_grid['p_disp_mw'] = 30

    run_and_assert_numba(net)

    # active power difference of dispatched and result
    ext_grid_diff_p = 0 - net.res_ext_grid.p_mw
    gen_diff = net.gen.p_mw * net.gen.scaling - net.res_gen.p_mw

    # resulting active slack power
    res_p_slack = ext_grid_diff_p.sum() + gen_diff.sum()

    # calculate target active power difference
    p_target_ext_grid = res_p_slack * net.ext_grid.slack_weight
    p_target_gen = res_p_slack * net.gen.slack_weight

    # check the power balances
    assert np.allclose(ext_grid_diff_p, p_target_ext_grid, atol=tol_mw)
    assert np.allclose(gen_diff, p_target_gen, atol=tol_mw)

    # check balance of power
    injected_p_mw, consumed_p_mw, xward_p_mw = _get_injection_consumption(net)
    assert abs(net.res_ext_grid.p_mw.sum() + net.res_gen.p_mw.sum() - consumed_p_mw - xward_p_mw) < 1e-6

    # check the distribution formula of the slack power difference
    assert_results_correct(net, tol=1e-6)


def test_case2848rte():
    # check how it works with a large grid
    net = networks.case2848rte()
    net.ext_grid['slack_weight'] = 0
    sl_gen = net.gen.loc[net.gen.p_mw > 1000].index
    net.gen.loc[sl_gen, 'slack_weight'] = net.gen.loc[sl_gen, 'p_mw']
    pp.runpp(net, distributed_slack=True, numba=numba_installed)
    assert_results_correct(net)


def test_multivoltage_example_with_controller():
    do_output = False

    net = networks.example_multivoltage()

    gen_p_disp = 10
    net.gen.at[0, "p_mw"] = gen_p_disp
    net.gen.at[0, "slack_weight"] = 1
    net.ext_grid.at[0, "slack_weight"] = 1
    net.trafo.at[1, "tap_max"] = 5
    net.trafo.at[1, "tap_min"] = -5
    ContinuousTapControl(net, tid=1, vm_set_pu=1.05)

    gen_disp = sum([net[elm].p_mw.sum() for elm in ["gen", "sgen"]])
    load_disp = sum([net[elm].p_mw.sum() for elm in ["load", "shunt"]]) + \
                net.xward[["ps_mw", "pz_mw"]].sum().sum()
    expected_losses = 2  # MW
    expected_slack_power = load_disp + expected_losses - gen_disp  # MW
    tol = 0.5  # MW

    net2 = net.deepcopy()
    # test distributed_slack
    run_and_assert_numba(net)

    # take results for gen and xward from net to net2 and compare results
    net2.gen.p_mw = net.res_gen.p_mw
    net2.xward.ps_mw = net._ppc['bus'][net._pd2ppc_lookups['bus'][net.xward.bus], PD] - net.load.loc[
        net.load.bus.isin([34, 32]), 'p_mw'].values
    pp.runpp(net2)

    assert_res_equal(net, net2)
    check_xward_results(net)
    assert_results_correct(net)

    if do_output:
        logger.info("grid losses: %.6f" % -net.res_bus.p_mw.sum())
        logger.info("slack generator p results: %.6f    %.6f" % (net.res_ext_grid.p_mw.at[0],
                                                                 net.res_gen.p_mw.at[0]))
        net.res_bus.vm_pu.plot()

    assert np.isclose(net.res_ext_grid.p_mw.at[0], net.res_gen.p_mw.at[0] - gen_p_disp)
    assert expected_slack_power / 2 - tol < net.res_ext_grid.p_mw.at[0] < expected_slack_power / 2 + tol
    losses_without_controller = -net.res_bus.p_mw.sum()
    slack_power_without_controller = net.res_ext_grid.p_mw.at[0] + net.res_gen.p_mw.at[0] - \
                                     gen_p_disp

    # test distributed_slack with controller
    run_and_assert_numba(net, run_control=True)
    check_xward_results(net)
    assert_results_correct(net)

    losses_with_controller = -net.res_bus.p_mw.sum()
    expected_slack_power = slack_power_without_controller - losses_without_controller + \
                           losses_with_controller
    slack_power_without_controller = net.res_ext_grid.p_mw.at[0] + net.res_gen.p_mw.at[0] - \
                                     gen_p_disp

    assert np.isclose(expected_slack_power, slack_power_without_controller, atol=1e-5)
    assert np.isclose(net.res_ext_grid.p_mw.at[0], expected_slack_power / 2, atol=1e-5)
    assert np.isclose(net.res_gen.p_mw.at[0], expected_slack_power / 2 + gen_p_disp, atol=1e-5)

    if do_output:
        logger.info("grid losses: %.6f" % -net.res_bus.p_mw.sum())
        logger.info("slack generator p results: %.6f    %.6f" % (net.res_ext_grid.p_mw.at[0],
                                                                 net.res_gen.p_mw.at[0]))
        net.res_bus.vm_pu.plot()


def test_dist_slack_user_pf_options():
    net = small_example_grid()
    # if no slack_weight is given for ext_grid, 1 is assumed, because normally
    # ext_grids are responsible to take the slack power
    net.gen["slack_weight"] = 1

    net2 = net.deepcopy()

    pp.runpp(net, distributed_slack=True)

    pp.set_user_pf_options(net2, distributed_slack=True)
    pp.runpp(net2)

    assert_res_equal(net, net2)
    assert_results_correct(net)
    assert_results_correct(net2)

    with pytest.raises(NotImplementedError):
        pp.runpp(net, distributed_slack=True, algorithm="bfsw")

    with pytest.raises(NotImplementedError):
        pp.runpp(net2, algorithm="bfsw")

    pp.set_user_pf_options(net2, algorithm="bfsw")
    with pytest.raises(NotImplementedError):
        pp.runpp(net2)


def test_dist_slack_with_enforce_q_lims():
    net = pp.networks.case9()
    net.ext_grid['slack_weight'] = 1 / 3
    net.gen['slack_weight'] = 1 / 3

    net.gen.at[0, 'max_q_mvar'] = 10
    net.gen.at[1, 'max_q_mvar'] = 0.5
    pp.runpp(net, distributed_slack=True, enforce_q_lims=True, numba=False)
    assert net._options["distributed_slack"] and net._options["enforce_q_lims"]
    assert np.allclose(net.res_gen.q_mvar, net.gen.max_q_mvar, rtol=0, atol=1e-6)

    assert_results_correct(net, tol=1e-6)


def test_dist_slack_with_enforce_q_lims_duplicate_gens():
    net = pp.networks.case9()
    pp.create_gen(net, net.ext_grid.bus.at[0], 1, slack=False, max_q_mvar=0.07)
    pp.create_gen(net, net.ext_grid.bus.at[0], 1, slack=False, max_q_mvar=0.01)
    pp.create_gen(net, net.ext_grid.bus.at[0], 2, slack=False, max_q_mvar=0.2)
    pp.create_gen(net, net.ext_grid.bus.at[0], 2, slack=False, max_q_mvar=0.1)
    pp.create_gen(net, net.ext_grid.bus.at[0], 0, slack=False, max_q_mvar=0.05)
    pp.create_gen(net, net.ext_grid.bus.at[0], 0, slack=False, max_q_mvar=0.25)
    net.gen['slack_weight'] = 1 / 8
    net.ext_grid['slack_weight'] = 1 / 8

    net.gen.at[0, 'max_q_mvar'] = 10
    net.gen.at[1, 'max_q_mvar'] = 0.5
    pp.runpp(net, distributed_slack=True, enforce_q_lims=True)
    assert net._options["distributed_slack"] and net._options["enforce_q_lims"]
    assert np.allclose(net.res_gen.q_mvar, net.gen.max_q_mvar, rtol=0, atol=1e-6)

    assert_results_correct(net, tol=1e-6)


# todo: implement distributed slack for when the grid has several disconnected zones


if __name__ == "__main__":
    pytest.main([__file__])

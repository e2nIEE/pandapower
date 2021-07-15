# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pandapower as pp
from pandapower import networks
import pytest
from pandapower.test.toolbox import assert_res_equal

try:
    import numba
    numba_installed = True
except ImportError:
    numba_installed = False


def small_example_grid():
    net = pp.create_empty_network()
    pp.create_buses(net, 3, 20)

    pp.create_gen(net, 0, p_mw=100, vm_pu=1, slack=True, slack_weight=1)
    # pp.create_ext_grid(net, 0)

    pp.create_load(net, 1, p_mw=100, q_mvar=100)

    pp.create_line_from_parameters(net, 0, 1, length_km=3, r_ohm_per_km=0.01, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
    pp.create_line_from_parameters(net, 1, 2, length_km=2, r_ohm_per_km=0.01, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
    pp.create_line_from_parameters(net, 2, 0, length_km=1, r_ohm_per_km=0.01, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
    return net


def _get_injection_consumption(net):
    consumed_p_mw = net.load.query("in_service").p_mw.sum() + net.res_line.pl_mw.sum()
    injected_p_mw = net.gen.query("in_service").p_mw.sum() + net.xward.query("in_service").ps_mw.sum()
    return injected_p_mw, consumed_p_mw


def _get_slack_weights(net):
    slack_weights = np.r_[net.gen.query("in_service").slack_weight,
                          net.ext_grid.query("in_service").slack_weight,
                          net.xward.query("in_service").slack_weight]
    return slack_weights / sum(slack_weights)


def _get_inputs_results(net):
    inputs = np.r_[net.gen.query("in_service").p_mw,
                   np.zeros(len(net.ext_grid.query("in_service"))),
                   net.xward.query("in_service").ps_mw]
    results = np.r_[net.res_gen[net.gen.in_service].p_mw,
                    net.res_ext_grid[net.ext_grid.in_service].p_mw,
                    net.res_xward[net.xward.in_service].p_mw]
    return inputs, results


def assert_results_correct(net):
    # first, collect slack_weights, injection and consumption, inputs and reults from net
    injected_p_mw, consumed_p_mw = _get_injection_consumption(net)
    input_p_mw, result_p_mw = _get_inputs_results(net)
    slack_weights = _get_slack_weights(net)

    # assert power balance is correct
    assert abs(result_p_mw.sum() - consumed_p_mw) < 1e-6
    # assert results are according to the distributed slack formula
    assert np.allclose(input_p_mw - (injected_p_mw - consumed_p_mw) * slack_weights, result_p_mw, atol=1e-6, rtol=0)


@pytest.mark.skipif(not numba_installed, reason="skip the test if numba not installed")
@pytest.mark.xfail(reason="is not implemented with numba")
def test_numba():
    net = small_example_grid()
    # if no slack_weight is given for ext_grid, 1 is assumed, because normally
    # ext_grids are responsible to take the slack power
    net.gen["slack_weight"] = 1
    net2 = net.deepcopy()
    pp.runpp(net, distributed_slack=True)

    assert_res_equal(net, net2)
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

    pp.runpp(net, distributed_slack=True, numba=False)
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

    pp.runpp(net, distributed_slack=True, numba=False, tolerance_mva=1e-6)
    assert_results_correct(net)


@pytest.mark.xfail(reason="xward not implemented as slack")
def test_gen_xward():
    net = small_example_grid()
    pp.create_xward(net, 2, 200, 0, 0, 0, 0, 6, 1, slack_weight=2)

    pp.runpp(net, distributed_slack=True, numba=False)
    assert_results_correct(net)


def test_ext_grid():
    net = small_example_grid()
    net.gen.in_service = False
    pp.create_ext_grid(net, 0, slack_weight=1)
    pp.create_ext_grid(net, 2, slack_weight=2)

    pp.runpp(net, distributed_slack=True, numba=False)
    assert_results_correct(net)


def test_gen_ext_grid():
    net = small_example_grid()
    pp.create_ext_grid(net, 2, slack_weight=2)

    pp.runpp(net, distributed_slack=True, numba=False)
    assert_results_correct(net)


def test_pvgen_ext_grid():
    # now test the behavior if gen is not slack
    net = small_example_grid()
    pp.create_ext_grid(net, 2, slack_weight=2)
    net.gen.slack = False

    pp.runpp(net, distributed_slack=True, numba=False)
    assert_results_correct(net)


def test_same_bus():
    net = small_example_grid()
    pp.create_ext_grid(net, 0, slack_weight=2)

    pp.runpp(net, distributed_slack=True, numba=False)
    assert_results_correct(net)


def test_separate_zones():
    net = small_example_grid()
    b1, b2 = pp.create_buses(net, 2, 110)
    pp.create_line_from_parameters(net, b1, b2, length_km=1, r_ohm_per_km=0.01, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
    pp.create_ext_grid(net, b1)
    pp.create_load(net, b2, 100)

    # distributed slack not implemented for separate zones
    with pytest.raises(NotImplementedError):
        pp.runpp(net, distributed_slack=True, numba=False)


def case9_simplified():
    net = pp.create_empty_network()
    pp.create_buses(net, 9, vn_kv=345.)
    lines = [[0, 3],[3, 4],[4, 5],[2, 5],[5, 6],[6, 7],[7, 1],[7, 8],[8, 3]]

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

    pp.runpp(net, distributed_slack=True, numba=False)

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
    injected_p_mw, consumed_p_mw = _get_injection_consumption(net)
    assert abs(net.res_ext_grid.p_mw.sum() + net.res_gen.p_mw.sum() - consumed_p_mw) < 1e-6

    # check the distribution formula of the slack power difference
    assert_results_correct(net)


# todo: implement distributed slack to work with numba
# todo: implement xward elements as slacks (similarly to gen with slack=True)
# todo add test for only xward when xward as slack is implemented
# todo: implement distributed slack for when the grid has several disconnected zones


if __name__ == "__main__":
    pytest.main([__file__])

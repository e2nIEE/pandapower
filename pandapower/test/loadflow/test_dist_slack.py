# -*- coding: utf-8 -*-

# Copyright (c) 2016-2021 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pandapower as pp
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

    pp.create_gen(net, 0, p_mw=100, vm_pu=1, slack=True, contribution_factor=1)
    # pp.create_ext_grid(net, 0)

    pp.create_load(net, 1, p_mw=100, q_mvar=100)

    pp.create_line_from_parameters(net, 0, 1, length_km=1, r_ohm_per_km=0.01, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
    pp.create_line_from_parameters(net, 1, 2, length_km=1, r_ohm_per_km=0.01, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
    pp.create_line_from_parameters(net, 2, 0, length_km=1, r_ohm_per_km=0.01, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
    return net


def get_injection_consumption(net):
    consumed_p_mw = net.load.query("in_service").p_mw.sum() + net.res_line.pl_mw.sum()
    injected_p_mw = net.gen.query("in_service").p_mw.sum() + net.xward.query("in_service").ps_mw.sum()
    return injected_p_mw, consumed_p_mw


@pytest.mark.skipif(not numba_installed, reason="skip the test if numba not installed")
@pytest.mark.xfail(reason="is not implemented with numba")
def test_numba():
    net = small_example_grid()
    # if no dspf is given for ext_grid, 1 is assumed, because normally
    # ext_grids are responsible to take the slack power
    net.gen["contribution_factor"] = 1
    net2 = net.deepcopy()
    pp.runpp(net, distributed_slack=True)

    assert_res_equal(net, net2)


def test_small_example():
    net = small_example_grid()
    # if no dspf is given for ext_grid, 1 is assumed, because normally
    # ext_grids are responsible to take the slack power
    net.gen["contribution_factor"] = 1

    net2 = net.deepcopy()

    pp.runpp(net, distributed_slack=True, numba=False)

    pp.runpp(net2, distributed_slack=False)

    assert_res_equal(net, net2)


def test_two_gens():
    net = small_example_grid()
    pp.create_gen(net, 2, 200, 1., slack=True, contribution_factor=2)

    pp.runpp(net, distributed_slack=True, numba=False)

    assert abs(net.res_gen.at[0, 'p_mw'] / net.res_gen.p_mw.sum() - 1 / 3) < 1e-6
    assert abs(net.res_gen.at[1, 'p_mw'] / net.res_gen.p_mw.sum() - 2 / 3) < 1e-6

    injected_p_mw, consumed_p_mw = get_injection_consumption(net)
    con_fac = net.gen.contribution_factor / np.sum(net.gen.contribution_factor)

    assert abs(net.res_gen.p_mw.sum() - consumed_p_mw) < 1e-6
    assert np.allclose(net.gen.p_mw - (injected_p_mw - consumed_p_mw) * con_fac, net.res_gen.p_mw, atol=1e-6, rtol=0)


@pytest.mark.xfail(reason="xward not implemented as slack")
def test_gen_xward():
    net = small_example_grid()
    pp.create_xward(net, 2, 200, 0, 0, 0, 0, 6, 1, contribution_factor=2)

    pp.runpp(net, distributed_slack=True, numba=False)

    assert abs(net.res_gen.at[0, 'p_mw'] / (net.res_gen.p_mw.sum() + net.res_xward.p_mw.sum()) - 1 / 3) < 1e-6
    assert abs(net.res_xward.at[0, 'p_mw'] / (net.res_gen.p_mw.sum() + net.res_xward.p_mw.sum()) - 2 / 3) < 1e-6

    consumed_p_mw = net.load.p_mw.sum() + net.res_line.pl_mw.sum()
    gen_p_mw = net.gen.p_mw.sum() + net.xward.ps_mw.sum()
    con_fac = np.array([1, 2]) / 3

    assert abs(net.res_gen.p_mw.sum() - consumed_p_mw) < 1e-6
    # todo fix energy check
    # assert np.allclose(net.gen.p_mw - (gen_p_mw - consumed_p_mw) * con_fac, net.res_gen.p_mw, atol=1e-6, rtol=0)


def test_ext_grid():
    net = small_example_grid()
    net.gen.in_service = False
    pp.create_ext_grid(net, 0, contribution_factor=1)
    pp.create_ext_grid(net, 2, contribution_factor=2)
    con_fac = net.ext_grid.contribution_factor / np.sum(net.ext_grid.contribution_factor)

    pp.runpp(net, distributed_slack=True, numba=False)

    injected_p_mw, consumed_p_mw = get_injection_consumption(net)
    assert abs(net.res_ext_grid.p_mw.sum() - consumed_p_mw) < 1e-6
    assert np.allclose((consumed_p_mw - injected_p_mw) * con_fac, net.res_ext_grid.p_mw, atol=1e-6, rtol=0)


def test_gen_ext_grid():
    net = small_example_grid()
    pp.create_ext_grid(net, 2, contribution_factor=2)
    con_fac = np.array([1, 2]) / 3

    pp.runpp(net, distributed_slack=True, numba=False)

    injected_p_mw, consumed_p_mw = get_injection_consumption(net)
    assert abs(net.res_ext_grid.p_mw.sum() + net.res_gen.p_mw.sum() - consumed_p_mw) < 1e-6
    input_p_mw = np.array([100, 0])  # 100 MW for gen and 0 for ext_grid
    res_p_mw = np.r_[net.res_gen.p_mw, net.res_ext_grid.p_mw]  # resulting injection after distributed slack
    assert np.allclose(input_p_mw - (injected_p_mw - consumed_p_mw) * con_fac, res_p_mw, atol=1e-6, rtol=0)


def test_pvgen_ext_grid():
    # now test the behavior if gen is not slack
    net = small_example_grid()
    pp.create_ext_grid(net, 2, contribution_factor=2)
    net.gen.slack = False
    con_fac = np.array([1, 2]) / 3

    # gen with slack=False does not work as distributed slack, we throw error
    with pytest.raises(UserWarning):
        pp.runpp(net, distributed_slack=True, numba=False)

    # injected_p_mw, consumed_p_mw = get_injection_consumption(net)
    # assert abs(net.res_ext_grid.p_mw.sum() + net.res_gen.p_mw.sum() - consumed_p_mw) < 1e-6
    # input_p_mw = np.array([100, 0])  # 100 MW for gen and 0 for ext_grid
    # res_p_mw = np.r_[net.res_gen.p_mw, net.res_ext_grid.p_mw]  # resulting injection after distributed slack
    # assert np.allclose(input_p_mw - (injected_p_mw - consumed_p_mw) * con_fac, res_p_mw, atol=1e-6, rtol=0)


def test_same_bus():
    net = small_example_grid()
    pp.create_ext_grid(net, 0, contribution_factor=2)
    con_fac = np.array([1, 2]) / 3

    pp.runpp(net, distributed_slack=True, numba=False)

    injected_p_mw, consumed_p_mw = get_injection_consumption(net)
    assert abs(net.res_ext_grid.p_mw.sum() + net.res_gen.p_mw.sum() - consumed_p_mw) < 1e-6
    input_p_mw = np.array([100, 0])  # 100 MW for gen and 0 for ext_grid
    res_p_mw = np.r_[net.res_gen.p_mw, net.res_ext_grid.p_mw]  # resulting injection after distributed slack
    assert np.allclose(input_p_mw - (injected_p_mw - consumed_p_mw) * con_fac, res_p_mw, atol=1e-6, rtol=0)


def test_separate_zones():
    net = small_example_grid()
    b1, b2 = pp.create_buses(net, 2, 110)
    pp.create_line_from_parameters(net, b1, b2, length_km=1, r_ohm_per_km=0.01, x_ohm_per_km=0.1, c_nf_per_km=0, max_i_ka=1)
    pp.create_gen(net, b1, 50, 1, slack=True, contribution_factor=1)
    pp.create_load(net, b2, 100)

    # distributed slack not implemented for separate zones
    with pytest.raises(NotImplementedError):
        pp.runpp(net, distributed_slack=True, numba=False)


# todo: implement distributed slack to work with numba
# todo: implement xward elements as slacks (similarly to gen with slack=True)
# todo add test for only xward when xward as slack is implemented
# todo: implement distributed slack for when the grid has several disconnected zones


if __name__ == "__main__":
    pytest.main([__file__])

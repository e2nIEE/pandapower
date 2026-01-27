# -*- coding: utf-8 -*-

# Copyright (c) 2016-2026 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest

from pandapower.networks.power_system_test_cases import case4gs, case5, case6ww, case9, case14, case24_ieee_rts, \
    case30, case_ieee30, case33bw, case39, case57, case89pegase, case118, case145, case_illinois200, case300, \
    case1354pegase, case1888rte, case2848rte, case2869pegase, case3120sp, case6470rte, case6495rte, case6515rte, \
    case9241pegase, GBnetwork, GBreducednetwork, iceland  # missing test for case11_iwamoto
from pandapower.run import runpp
import numpy as np


def _compare_arrays(arr1, arr2, tolerance=0.1):
    if len(arr1) != len(arr2):
        raise ValueError("Arrays must have the same shape.")

    # Calculate the absolute difference
    diff = np.abs(arr1 - arr2)

    # Find positions where the difference exceeds the tolerance
    indices = np.nonzero(diff > tolerance)[0]

    # Return the positions of differences that are too large
    # indices contains the row indices for 1D arrays
    return len(indices) == 0, indices


def compare_arrays(arr1, arr2, atol=0.1):
    is_equal, where = _compare_arrays(arr1, arr2, atol)
    if not is_equal:
        name = ''
        if 'name' in dir(arr1):
            name = arr1.name
        raise ValueError(f"In {name}, the following elements are not equal: {where}")


def _ppc_element_test(net, n_bus=None, n_branch=None, n_gen=None, gencost=False):
    runpp(net, trafo_model="pi")
    if n_bus:
        assert len(net.bus) == n_bus
    if n_branch:
        assert len(net.line) + len(net.trafo) == n_branch
    if n_gen:
        assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == n_gen
    if gencost:
        if isinstance(gencost, bool):
            assert net.poly_cost.shape[0] == n_gen
        else:
            assert net.poly_cost.shape[0] == gencost
    assert net.converged


def test_case4gs():
    net = case4gs()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 4, 4, 2, False)


def test_case5():
    net = case5()
    runpp(net)
    assert net.converged

    # the following values are calculated using powerfactory,
    # the network was recreated by hand and all the values were transferred.
    # Tolerances were set so that pp does not fail.
    compare_arrays(net.res_bus.vm_pu, [1.00, 0.99, 1.00, 1.00, 1.00], atol=0.02)
    compare_arrays(net.res_bus.va_degree, [3.275, -0.758, -0.492, 0.000, 4.113], atol=0.002)

    compare_arrays(net.res_line.p_from_mw, [249.81, 186.40, -226.21, -51.96, -28.59, -238.26], atol=0.2)
    compare_arrays(net.res_line.p_to_mw, [-248.04, -185.28, 226.54, 52.08, 28.62, 239.97], atol=0.2)
    compare_arrays(net.res_line.i_ka, [0.630, 0.469, 0.571, 0.268, 0.072, 0.604], atol=0.1)
    compare_arrays(net.res_line.q_from_mvar, [21.6, -14.6, 22.7, -94.0, 2.6, 32.16], atol=1.1)
    compare_arrays(net.res_line.q_to_mvar, [-4.6, 24.6, -22.5, 93.4, -3.1, -15.7], atol=1.1)

    compare_arrays(net.res_gen.p_mw, [40.0, 323.5, 466.5], atol=0.1)
    compare_arrays(net.res_gen.q_mvar, [29.7, 194.7, -38.2], atol=1.1)

    _ppc_element_test(net, 5, 6, 5, False)


def test_case6ww():
    net = case6ww()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 6, 11, 3, True)


def test_case9():
    net = case9()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 9, 9, 3, True)


def test_case14():
    net = case14()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 14, 20, 5, True)


def test_case24_ieee_rts():
    net = case24_ieee_rts()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 24)


def test_case30():
    net = case30()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 30, 41, 6, True)


def test_case_ieee30():
    net = case_ieee30()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 30, 41, 6, False)


def test_case33bw():
    net = case33bw()
    runpp(net, trafo_mode="pi")
    assert net.converged
    _ppc_element_test(net, 33, 37, 1, True)


def test_case39():
    net = case39()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 39, 46, 10, True)


def test_case57():
    net = case57()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 57, 80, 7, True)


def test_case89pegase():
    net = case89pegase()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 89, 210, 12 + 6, 12)


def test_case118():
    net = case118()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 118, 186, 54, True)


def test_case145():
    net = case145()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 145, 453, 50 + 9, 50)


def test_case_illinois200():
    net = case_illinois200()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 200, 245, 49, False)


def test_case300():
    net = case300()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 300, 411, 69 + 8, 69)


def test_case1354pegase():
    net = case1354pegase()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 1354, 1991, 260 + 52, 260)


def test_case1888rte():
    net = case1888rte()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 1888, 2531, 354, 297)


def test_case1888rte_changed_slack():
    ref_bus_idx = 1233
    net = case1888rte(ref_bus_idx=ref_bus_idx)
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 1888, 2531, 354, 297)
    assert net.ext_grid.bus.at[0] == ref_bus_idx

    ref_bus_idx = [1233, 1854]
    net = case1888rte(ref_bus_idx=ref_bus_idx)
    runpp(net, trafo_model='pi')
    assert list(net.ext_grid.bus.sort_values()) == ref_bus_idx
    assert net.converged


def test_case2848rte():
    net = case2848rte()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 2848, 3776, 670, 547)


def test_case2869pegase():
    net = case2869pegase()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 2869, 4582, 690, 510)


def test_case3120sp():
    net = case3120sp()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 3120, 3693, 505, True)


def test_case6470rte():
    net = case6470rte()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 6470, 9005, 1578, 1330)


def test_case6495rte():
    net = case6495rte()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 6495, 9019, 1650, 1372)


def test_case6515rte():
    net = case6515rte()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 6515, 9037, 1651, 1388)


def test_case9241pegase():
    net = case9241pegase()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 9241, 16049, 1879, 1445)


def test_GBreducednetwork():
    net = GBreducednetwork()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 29, 99, 66, True)


def test_GBnetwork():
    net = GBnetwork()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 2224, 3207, 394 + 37, 394)


def test_iceland():
    net = iceland()
    runpp(net)
    assert net.converged
    _ppc_element_test(net, 189, 206, 35, True)


if __name__ == '__main__':
    pytest.main([__file__, "-xs"])

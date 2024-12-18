# -*- coding: utf-8 -*-

# Copyright (c) 2016-2024 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest

from pandapower.networks.power_system_test_cases import case4gs, case5, case6ww, case9, case14, case24_ieee_rts, \
    case30, case_ieee30, case33bw, case39, case57, case89pegase, case118, case145, case_illinois200, case300, \
    case1354pegase, case1888rte, case2848rte, case2869pegase, case3120sp, case6470rte, case6495rte, case6515rte, \
    case9241pegase, GBnetwork, GBreducednetwork, iceland  # missing test for case11_iwamoto
from pandapower.run import runpp


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

# -*- coding: utf-8 -*-

# Copyright (c) 2016-2018 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pytest

import pandapower as pp
import pandapower.networks as pn


def test_case4gs():
    net = pn.case4gs()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 4
    assert len(net.line) + len(net.trafo) == 4
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 2
    assert net.converged


def test_case6ww():
    net = pn.case6ww()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 6
    assert len(net.line) + len(net.trafo) == 11
    n_gen = 3
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == n_gen
    assert len(net.polynomial_cost) == n_gen
    assert net.converged


def test_case9():
    net = pn.case9()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 9
    assert len(net.line) + len(net.trafo) == 9
    n_gen = 3
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == n_gen
    assert len(net.polynomial_cost) == n_gen
    assert net.converged


def test_case14():
    net = pn.case14()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 14
    assert len(net.line) + len(net.trafo) == 20
    n_gen = 5
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == n_gen
    assert len(net.polynomial_cost) == n_gen
    assert net.converged


def test_case24_ieee_rts():
    net = pn.case24_ieee_rts()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 24
    assert net.converged


def test_case30():
    net = pn.case30()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 30
    assert len(net.line) + len(net.trafo) == 41
    n_gen = 6
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == n_gen
    assert len(net.polynomial_cost) == n_gen
    assert net.converged


def test_case33bw():
    net = pn.case33bw()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 33
    assert len(net.line) + len(net.trafo) == 37
    n_gen = 1
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == n_gen
    assert len(net.polynomial_cost) == n_gen
    assert net.converged


def test_case39():
    net = pn.case39()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 39
    assert len(net.line) + len(net.trafo) == 46
    n_gen = 10
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == n_gen
    assert len(net.polynomial_cost) == n_gen
    assert net.converged


def test_case57():
    net = pn.case57()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 57
    assert len(net.line) + len(net.trafo) == 80
    n_gen = 7
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == n_gen
    assert len(net.polynomial_cost) == n_gen
    assert net.converged


def test_case89pegase():
    net = pn.case89pegase()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 89
    assert len(net.line) + len(net.trafo) == 210
    n_gen = 12
    assert len(net.ext_grid) + len(net.gen) == n_gen
    assert len(net.polynomial_cost) == n_gen
    assert net.converged


def test_case118():
    net = pn.case118()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 118
    assert len(net.line) + len(net.trafo) == 186
    n_gen = 54
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == n_gen
    assert len(net.polynomial_cost) == n_gen
    assert net.converged


def test_case145():
    net = pn.case145()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 145
    assert len(net.line) + len(net.trafo) == 453
    n_gen = 50
    assert len(net.ext_grid) + len(net.gen) == n_gen
    assert len(net.polynomial_cost) == n_gen
    assert net.converged


def test_case300():
    net = pn.case300()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 300
    assert len(net.line) + len(net.trafo) == 411
    n_gen = 69
    assert len(net.ext_grid) + len(net.gen) == n_gen
    assert len(net.polynomial_cost) == n_gen
    assert net.converged


def test_case1354pegase():
    net = pn.case1354pegase()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 1354
    assert len(net.line) + len(net.trafo) == 1991
    n_gen = 260
    assert len(net.ext_grid) + len(net.gen) == n_gen
    assert len(net.polynomial_cost) == n_gen
    assert net.converged


def test_case1888rte():
    net = pn.case1888rte()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 1888
    assert len(net.line) + len(net.trafo) == 2531
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 354
    assert len(net.polynomial_cost) == 297
    assert net.converged


def test_case1888rte_changed_slack():
    ref_bus_idx = 1233
    net = pn.case1888rte(ref_bus_idx=ref_bus_idx)
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 1888
    assert len(net.line) + len(net.trafo) == 2531
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 354
    assert len(net.polynomial_cost) == 297
    assert net.ext_grid.bus.at[0] == ref_bus_idx
    assert net.converged

    ref_bus_idx = [1233, 1854]
    net = pn.case1888rte(ref_bus_idx=ref_bus_idx)
    pp.runpp(net, trafo_model='pi')
    assert list(net.ext_grid.bus) == ref_bus_idx
    assert net.converged


def test_case2848rte():
    net = pn.case2848rte()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 2848
    assert len(net.line) + len(net.trafo) == 3776
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 670
    assert len(net.polynomial_cost) == 547
    assert net.converged


def test_case2869pegase():
    net = pn.case2869pegase()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 2869
    assert len(net.line) + len(net.trafo) == 4582
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 690
    assert len(net.polynomial_cost) == 510
    assert net.converged


def test_case3120sp():
    net = pn.case3120sp()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 3120
    assert len(net.line) + len(net.trafo) == 3693
    n_gen = 505
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == n_gen
    assert len(net.polynomial_cost) == n_gen
    assert net.converged


def test_case6470rte():
    net = pn.case6470rte()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 6470
    assert len(net.line) + len(net.trafo) == 9005
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 1578
    assert len(net.polynomial_cost) == 1330
    assert net.converged


def test_case6495rte():
    net = pn.case6495rte()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 6495
    assert len(net.line) + len(net.trafo) == 9019
    n_gen = 1650
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == n_gen
#    assert len(net.polynomial_cost) == n_gen
    assert net.converged


def test_case6515rte():
    net = pn.case6515rte()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 6515
    assert len(net.line) + len(net.trafo) == 9037
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 1651
    assert len(net.polynomial_cost) == 1388
    assert net.converged


def test_case9241pegase():
    net = pn.case9241pegase()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 9241
    assert len(net.line) + len(net.trafo) == 16049
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 1879
    assert len(net.polynomial_cost) == 1445
    assert net.converged


def test_GBreducednetwork():
    net = pn.GBreducednetwork()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 29
    assert len(net.line) + len(net.trafo) == 99
    n_gen = 66
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == n_gen
    assert len(net.polynomial_cost) == n_gen
    assert net.converged


def test_GBnetwork():
    net = pn.GBnetwork()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 2224
    assert len(net.line) + len(net.trafo) == 3207
    n_gen = 394
    assert len(net.ext_grid) + len(net.gen) == n_gen
    assert len(net.polynomial_cost) == n_gen
    assert net.converged


def test_iceland():
    net = pn.iceland()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 189
    assert len(net.line) + len(net.trafo) == 206
    n_gen = 35
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == n_gen
    assert len(net.polynomial_cost) == n_gen
    assert net.converged


if __name__ == '__main__':
    pytest.main(["test_power_system_test_cases.py", "-xs"])

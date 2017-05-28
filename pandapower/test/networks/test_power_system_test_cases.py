# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

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
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 3
    assert net.converged


def test_case9():
    net = pn.case9()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 9
    assert len(net.line) + len(net.trafo) == 9
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 3
    assert net.converged


def test_case14():
    net = pn.case14()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 14
    assert len(net.line) + len(net.trafo) == 20
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 5
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
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 6
    assert net.converged


def test_case33bw():
    net = pn.case33bw()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 33
    assert len(net.line) + len(net.trafo) == 37
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 1
    assert net.converged


def test_case39():
    net = pn.case39()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 39
    assert len(net.line) + len(net.trafo) == 46
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 10
    assert net.converged


def test_case57():
    net = pn.case57()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 57
    assert len(net.line) + len(net.trafo) == 80
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 7
    assert net.converged


def test_case89pegase():
    net = pn.case89pegase()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 89
    assert len(net.line) + len(net.trafo) == 210
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 18
    assert net.converged


def test_case118():
    net = pn.case118()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 118
    assert len(net.line) + len(net.trafo) == 186
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 54
    assert net.converged


def test_case145():
    net = pn.case145()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 145
    assert len(net.line) + len(net.trafo) == 453
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 59
    assert net.converged


def test_case300():
    net = pn.case300()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 300
    assert len(net.line) + len(net.trafo) == 411
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 77
    assert net.converged


def test_case1354pegase():
    net = pn.case1354pegase()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 1354
    assert len(net.line) + len(net.trafo) == 1991
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 312
    assert net.converged


def test_case1888rte():
    net = pn.case1888rte()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 1888
    assert len(net.line) + len(net.trafo) == 2531
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 354
    assert net.converged


def test_case2848rte():
    net = pn.case2848rte()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 2848
    assert len(net.line) + len(net.trafo) == 3776
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 670
    assert net.converged


def test_case2869pegase():
    net = pn.case2869pegase()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 2869
    assert len(net.line) + len(net.trafo) == 4582
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 690
    assert net.converged


def test_case3120sp():
    net = pn.case3120sp()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 3120
    assert len(net.line) + len(net.trafo) == 3693
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 505
    assert net.converged


def test_case6470rte():
    net = pn.case6470rte()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 6470
    assert len(net.line) + len(net.trafo) == 9005
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 1578
    assert net.converged


def test_case6495rte():
    net = pn.case6495rte()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 6495
    assert len(net.line) + len(net.trafo) == 9019
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 1650
    assert net.converged


def test_case6515rte():
    net = pn.case6515rte()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 6515
    assert len(net.line) + len(net.trafo) == 9037
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 1651
    assert net.converged


def test_case9241pegase():
    net = pn.case9241pegase()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 9241
    assert len(net.line) + len(net.trafo) == 16049
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 1879
    assert net.converged


def test_GBreducednetwork():
    net = pn.GBreducednetwork()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 29
    assert len(net.line) + len(net.trafo) == 99
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 66
    assert net.converged


def test_GBnetwork():
    net = pn.GBnetwork()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 2224
    assert len(net.line) + len(net.trafo) == 3207
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 431
    assert net.converged


def test_iceland():
    net = pn.iceland()
    assert net.converged
    pp.runpp(net, trafo_model='pi')
    assert len(net.bus) == 189
    assert len(net.line) + len(net.trafo) == 206
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 35
    assert net.converged


if __name__ == '__main__':
#    net = pn.case30Q()
    pytest.main(["test_power_system_test_cases.py", "-xs"])

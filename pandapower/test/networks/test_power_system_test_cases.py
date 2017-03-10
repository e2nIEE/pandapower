# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import pytest
import pandapower as pp
import pandapower.networks as pn


def test_case4gs():
    net = pn.case4gs()
    assert net.converged
    pp.runpp(net)
    assert len(net.bus) == 4
    assert len(net.line) + len(net.trafo) == 4
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 2
    assert net.converged


def test_case6ww():
    net = pn.case6ww()
    assert net.converged
    pp.runpp(net)
    assert len(net.bus) == 6
    assert len(net.line) + len(net.trafo) == 11
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 3
    assert net.converged


def test_case9():
    net = pn.case9()
    assert net.converged
    pp.runpp(net)
    assert len(net.bus) == 9
    assert len(net.line) + len(net.trafo) == 9
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 3
    assert net.converged


def test_case14():
    net = pn.case14()
    assert net.converged
    pp.runpp(net)
    assert len(net.bus) == 14
    assert len(net.line) + len(net.trafo) == 20
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 5
    assert net.converged


def test_case24_ieee_rts():
    net = pn.case24_ieee_rts()
    assert net.converged
    pp.runpp(net)
    assert len(net.bus) == 24
    assert net.converged


def test_case30():
    net = pn.case30()
    assert net.converged
    pp.runpp(net)
    assert len(net.bus) == 30
    assert len(net.line) + len(net.trafo) == 41
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 6
    assert net.converged


def test_case33bw():
    net = pn.case33bw()
    pp.runpp(net)
    assert len(net.bus) == 33
    assert len(net.line) + len(net.trafo) == 37
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 1
    assert net.converged


def test_case39():
    net = pn.case39()
    assert net.converged
    pp.runpp(net)
    assert len(net.bus) == 39
    assert len(net.line) + len(net.trafo) == 46
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 10
    assert net.converged


def test_case57():
    net = pn.case57()
    assert net.converged
    pp.runpp(net)
    assert len(net.bus) == 57
    assert len(net.line) + len(net.trafo) == 80
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 7
    assert net.converged


def test_case118():
    net = pn.case118()
    assert net.converged
    pp.runpp(net)
    assert len(net.bus) == 118
    assert len(net.line) + len(net.trafo) == 186
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 54
    assert net.converged


def test_case300():
    net = pn.case300()
    assert net.converged
    pp.runpp(net)
    assert len(net.bus) == 300
    assert len(net.line) + len(net.trafo) == 411
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 77
    assert net.converged


def test_case1354pegase():
    net = pn.case1354pegase()
    pp.runpp(net)
    assert len(net.bus) == 1354
    assert len(net.line) + len(net.trafo) == 1991
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 312
    assert net.converged


def test_case2869pegase():
    net = pn.case2869pegase()
    pp.runpp(net)
    assert len(net.bus) == 2869
    assert len(net.line) + len(net.trafo) == 4582
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 690
    assert net.converged


def test_case9241pegase():
    net = pn.case9241pegase()
    pp.runpp(net)
    assert len(net.bus) == 9241
    assert len(net.line) + len(net.trafo) == 16049
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 1879
    assert net.converged


def test_GBreducednetwork():
    net = pn.GBreducednetwork()
    pp.runpp(net)
    assert len(net.bus) == 29
    assert len(net.line) + len(net.trafo) == 99
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 66
    assert net.converged


def test_GBnetwork():
    net = pn.GBnetwork()
    pp.runpp(net)
    assert len(net.bus) == 2224
    assert len(net.line) + len(net.trafo) == 3207
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 431
    assert net.converged


def test_iceland():
    net = pn.iceland()
    pp.runpp(net)
    assert len(net.bus) == 189
    assert len(net.line) + len(net.trafo) == 206
    assert len(net.ext_grid) + len(net.gen) + len(net.sgen) == 35
    assert net.converged


if __name__ == '__main__':
#    net = pn.case30Q()
    pytest.main(["test_power_system_test_cases.py", "-xs"])

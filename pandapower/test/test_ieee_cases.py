# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a 
# BSD-style license that can be found in the LICENSE file.

import pytest
import pandapower as pp
import pandapower.networks as pn


def test_case4gs():
    net = pn.case4gs()
    assert net.converged is True
    pp.runpp(net)
    assert len(net.bus) == 4
    assert len(net.line) + len(net.trafo) == 4
    assert len(net.gen) + len(net.ext_grid) == 2
    assert net.converged is True


def test_case6ww():
    net = pn.case6ww()
    assert net.converged is True
    pp.runpp(net)
    assert len(net.bus) == 6
    assert len(net.line) + len(net.trafo) == 11
    assert len(net.gen) + len(net.ext_grid) == 3
    assert net.converged is True


def test_case9():
    net = pn.case9()
    assert net.converged is True
    pp.runpp(net)
    assert len(net.bus) == 9
    assert len(net.line) + len(net.trafo) == 9
    assert len(net.gen) + len(net.ext_grid) == 3
    assert net.converged is True


def test_case9Q():
    net = pn.case9Q()
    assert net.converged is True
    pp.runpp(net)
    assert len(net.bus) == 9
    assert len(net.line) + len(net.trafo) == 9
    assert len(net.gen) + len(net.ext_grid) == 3
    assert net.converged is True


def test_case30():
    net = pn.case30()
    assert net.converged is True
    pp.runpp(net)
    assert len(net.bus) == 30
    assert len(net.line) + len(net.trafo) == 41
    assert len(net.gen) + len(net.ext_grid) == 6
    assert net.converged is True


def test_case30pwl():
    net = pn.case30pwl()
    assert net.converged is True
    pp.runpp(net)
    assert len(net.bus) == 30
    assert len(net.line) + len(net.trafo) == 41
    assert len(net.gen) + len(net.ext_grid) == 6
    assert net.converged is True


def test_case30Q():
    net = pn.case30Q()
    assert net.converged is True
    pp.runpp(net)
    assert len(net.bus) == 30
    assert len(net.line) + len(net.trafo) == 41
    assert len(net.gen) + len(net.ext_grid) == 6
    assert net.converged is True

if __name__ == '__main__':
    pytest.main(["test_ieee_cases.py","-xs"])

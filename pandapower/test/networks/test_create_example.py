# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import pandas as pd
import pytest

import pandapower as pp
import pandapower.networks as pn


def test_create_simple():
    net = pn.example_simple()
    pp.runpp(net)

    assert len(net.bus) >= 1
    assert len(net.line) >= 1
    assert len(net.gen) >= 1
    assert len(net.sgen) >= 1
    assert len(net.shunt) >= 1
    assert len(net.trafo) >= 1
    assert len(net.load) >= 1
    assert len(net.ext_grid) >= 1
    assert len(net.switch[net.switch.et == 'l']) >= 1
    assert len(net.switch[net.switch.et == 'b']) >= 1
    assert net.converged


def test_create_realistic():
    net = pn.example_multivoltage()
    pp.runpp(net)

    all_vn_kv = pd.Series([380, 110, 20, 10, 0.4])
    assert net.bus.vn_kv.isin(all_vn_kv).all()
    assert len(net.bus) >= 1
    assert len(net.line) >= 1
    assert len(net.gen) >= 1
    assert len(net.sgen) >= 1
    assert len(net.shunt) >= 1
    assert len(net.trafo) >= 1
    assert len(net.trafo3w) >= 1
    assert len(net.load) >= 1
    assert len(net.ext_grid) >= 1
    assert len(net.switch[net.switch.et == 'l']) >= 1
    assert len(net.switch[net.switch.et == 'b']) >= 1
    assert len(net.switch[net.switch.et == 't']) >= 1
    assert len(net.switch[net.switch.type == 'CB']) >= 1
    assert len(net.switch[net.switch.type == 'DS']) >= 1
    assert len(net.switch[net.switch.type == 'LBS']) >= 1
    assert len(net.switch[net.switch.closed]) >= 1
    assert len(net.switch[~net.switch.closed]) >= 1
    assert len(net.impedance) >= 1
    assert len(net.xward) >= 1
    assert net.converged

if __name__ == '__main__':
    pytest.main(['-x', "test_create_example.py"])

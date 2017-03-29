# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import pytest

import pandapower.networks as pn


def test_kb_extrem_landnetz_freileitung():
    pd_net = pn.kb_extrem_landnetz_freileitung(p_load_in_kw=2., q_load_in_kvar=1.)
    assert abs(pd_net.line.length_km.sum() - 0.312) < 0.00000001
    assert abs(pd_net.load.p_kw.sum() - 52.) < 0.00000001
    assert abs(pd_net.load.q_kvar.sum() - 26.) < 0.00000001
    assert len(pd_net.bus.index) == 28
    assert len(pd_net.line.index) == 26
    assert len(pd_net.trafo.index) == 1


def test_kb_extrem_landnetz_kabel():
    pd_net = pn.kb_extrem_landnetz_kabel(p_load_in_kw=2., q_load_in_kvar=1.)
    assert abs(pd_net.line.length_km.sum() - 1.339) < 0.00000001
    assert abs(pd_net.load.p_kw.sum() - 52.) < 0.00000001
    assert abs(pd_net.load.q_kvar.sum() - 26.) < 0.00000001
    assert len(pd_net.bus.index) == 54
    assert len(pd_net.line.index) == 52
    assert len(pd_net.trafo.index) == 1


def test_kb_extrem_landnetz_freileitung_trafo():
    pd_net = pn.kb_extrem_landnetz_freileitung_trafo(p_load_in_kw=2., q_load_in_kvar=1.)
    assert abs(pd_net.line.length_km.sum() - 0.348) < 0.00000001
    assert abs(pd_net.load.p_kw.sum() - 54.) < 0.00000001
    assert abs(pd_net.load.q_kvar.sum() - 27.) < 0.00000001
    assert len(pd_net.bus.index) == 29
    assert len(pd_net.line.index) == 27
    assert len(pd_net.trafo.index) == 1


def test_kb_extrem_landnetz_kabel_trafo():
    pd_net = pn.kb_extrem_landnetz_kabel_trafo(p_load_in_kw=2., q_load_in_kvar=1.)
    assert abs(pd_net.line.length_km.sum() - 1.435) < 0.00000001
    assert abs(pd_net.load.p_kw.sum() - 54.) < 0.00000001
    assert abs(pd_net.load.q_kvar.sum() - 27.) < 0.00000001
    assert len(pd_net.bus.index) == 56
    assert len(pd_net.line.index) == 54
    assert len(pd_net.trafo.index) == 1


def test_kb_extrem_dorfnetz():
    pd_net = pn.kb_extrem_dorfnetz(p_load_in_kw=2., q_load_in_kvar=1.)
    assert abs(pd_net.line.length_km.sum() - 3.088) < 0.00000001
    assert abs(pd_net.load.p_kw.sum() - 116.) < 0.00000001
    assert abs(pd_net.load.q_kvar.sum() - 58.) < 0.00000001
    assert len(pd_net.bus.index) == 118
    assert len(pd_net.line.index) == 116
    assert len(pd_net.trafo.index) == 1


def test_kb_extrem_dorfnetz_trafo():
    pd_net = pn.kb_extrem_dorfnetz_trafo(p_load_in_kw=2., q_load_in_kvar=1.)
    assert abs(pd_net.line.length_km.sum() - 6.094) < 0.00000001
    assert abs(pd_net.load.p_kw.sum() - 234.) < 0.00000001
    assert abs(pd_net.load.q_kvar.sum() - 117.) < 0.00000001
    assert len(pd_net.bus.index) == 236
    assert len(pd_net.line.index) == 234
    assert len(pd_net.trafo.index) == 1


def test_kb_extrem_vorstadtnetz_1():
    pd_net = pn.kb_extrem_vorstadtnetz_1(p_load_in_kw=2., q_load_in_kvar=1.)
    assert abs(pd_net.line.length_km.sum() - 3.296) < 0.00000001
    assert abs(pd_net.load.p_kw.sum() - 290.) < 0.00000001
    assert abs(pd_net.load.q_kvar.sum() - 145.) < 0.00000001
    assert len(pd_net.bus.index) == 292
    assert len(pd_net.line.index) == 290
    assert len(pd_net.trafo.index) == 1


def test_kb_extrem_vorstadtnetz_2():
    pd_net = pn.kb_extrem_vorstadtnetz_2(p_load_in_kw=2., q_load_in_kvar=1.)
    assert abs(pd_net.line.length_km.sum() - 4.019) < 0.00000001
    assert abs(pd_net.load.p_kw.sum() - 290.) < 0.00000001
    assert abs(pd_net.load.q_kvar.sum() - 145.) < 0.00000001
    assert len(pd_net.bus.index) == 292
    assert len(pd_net.line.index) == 290
    assert len(pd_net.trafo.index) == 1


def test_kb_extrem_vorstadtnetz_trafo_1():
    pd_net = pn.kb_extrem_vorstadtnetz_trafo_1(p_load_in_kw=2., q_load_in_kvar=1.)
    assert abs(pd_net.line.length_km.sum() - 5.256) < 0.00000001
    assert abs(pd_net.load.p_kw.sum() - 382.) < 0.00000001
    assert abs(pd_net.load.q_kvar.sum() - 191.) < 0.00000001
    assert len(pd_net.bus.index) == 384
    assert len(pd_net.line.index) == 382
    assert len(pd_net.trafo.index) == 1


def test_kb_extrem_vorstadtnetz_trafo_2():
    pd_net = pn.kb_extrem_vorstadtnetz_trafo_2(p_load_in_kw=2., q_load_in_kvar=1.)
    assert abs(pd_net.line.length_km.sum() - 5.329) < 0.00000001
    assert abs(pd_net.load.p_kw.sum() - 384.) < 0.00000001
    assert abs(pd_net.load.q_kvar.sum() - 192.) < 0.00000001
    assert len(pd_net.bus.index) == 386
    assert len(pd_net.line.index) == 384
    assert len(pd_net.trafo.index) == 1

if __name__ == '__main__':
    pytest.main(['-x', "test_kerber_extreme_networks.py"])
#    test_kb_extrem_landnetz_kabel()

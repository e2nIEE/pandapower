# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.

import pandapower as pp
import pandapower.shortcircuit as sc
import numpy as np
import pytest

@pytest.fixture
def one_line_one_static_generator():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    
    pp.create_ext_grid(net, b1, s_sc_max_mva=10., s_sc_min_mva=8., rx_min=0.1, rx_max=0.1)
    l1 = pp.create_line_from_parameters(net, b1, b2, c_nf_per_km=190, max_i_ka=0.829,
                                        r_ohm_per_km=0.0306, x_ohm_per_km=0.1256637, length_km=1.)
    net.line.loc[l1, "endtemp_degree"] = 250
    pp.create_sgen(net, b2, p_kw=0, sn_kva=500.)
    pp.create_sgen(net, b2, p_kw=0, sn_kva=500.)
    return net

def test_max_sgen_3ph(one_line_one_static_generator):
    net = one_line_one_static_generator
    sc.calc_sc(net, fault="3ph", case="max", ith=True, ip=True)
    assert np.isclose(net.res_bus_sc.ikss_ka.at[1], 0.069801849155)
    assert np.isclose(net.res_bus_sc.ip_ka.at[1], 0.1723538861)
    assert np.isclose(net.res_bus_sc.ith_ka.at[1], 0.070982785191)
    
def test_min_sgen_3ph(one_line_one_static_generator):
    net = one_line_one_static_generator
    sc.calc_sc(net, fault="3ph", case="min", ith=True, ip=True)
    assert np.isclose(net.res_bus_sc.ikss_ka.at[1], 0.05773139511)
    assert np.isclose(net.res_bus_sc.ip_ka.at[1], 0.14254751833)
    assert np.isclose(net.res_bus_sc.ith_ka.at[1], 0.058708001755)
    

def test_max_sgen_2ph(one_line_one_static_generator):
    net = one_line_one_static_generator
    sc.calc_sc(net, fault="2ph", case="max", ith=True, ip=True)
    assert np.isclose(net.res_bus_sc.ikss_ka.at[1], 0.045450174275)
    assert np.isclose(net.res_bus_sc.ip_ka.at[1], 0.11222455043)
    assert np.isclose(net.res_bus_sc.ith_ka.at[1], 0.046219118778)
    
def test_min_sgen_2ph(one_line_one_static_generator):
    net = one_line_one_static_generator
    sc.calc_sc(net, fault="2ph", case="min", ith=True, ip=True)
    assert np.isclose(net.res_bus_sc.ikss_ka.at[1], 0.03636049113)
    assert np.isclose(net.res_bus_sc.ip_ka.at[1], 0.08977859705)
    assert np.isclose(net.res_bus_sc.ith_ka.at[1], 0.036975579276)    

def test_neglect_sgens(one_line_one_static_generator):
    net = one_line_one_static_generator
    sc.calc_sc(net, fault="3ph", case="max", consider_sgens=False)
    assert np.isclose(net.res_bus_sc.ikss_ka.at[1], 0.052481340705)  
    
if __name__ == '__main__':
    pytest.main(['test_sgen.py'])    

# -*- coding: utf-8 -*-

# Copyright (c) 2016 by University of Kassel and Fraunhofer Institute for Wind Energy and Energy
# System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed by a
# BSD-style license that can be found in the LICENSE file.


import pandapower as pp
import pandapower.shortcircuit as sc
import numpy as np
import pytest

@pytest.fixture
def ring_network():
    net = pp.create_empty_network()
    b0 = pp.create_bus(net, 220)
    b1 = pp.create_bus(net, 110)
    b2 = pp.create_bus(net, 110)
    b3 = pp.create_bus(net, 110)
    pp.create_ext_grid(net, b0, s_sc_max_mva=100., s_sc_min_mva=80., rx_min=0.1, rx_max=0.1)
    pp.create_transformer(net, b0, b1, "100 MVA 220/110 kV")
    pp.create_line(net, b2, b1, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV" , length_km=20.)
    l2 = pp.create_line(net, b2, b3, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV" , length_km=15.)
    pp.create_line(net, b3, b1, std_type="N2XS(FL)2Y 1x185 RM/35 64/110 kV" , length_km=10.)
    pp.create_switch(net, b3, l2, closed=False, et="l")
    return net


def test_open_ring(ring_network):
    net = ring_network
    sc.calc_sc(net)
    assert np.allclose(net.res_trafo_sc.ikss_lv_ka.values, [0.4745379023])
    assert np.allclose(net.res_line_sc.ikss_ka.values, [0.46413866727, 0, 0.4692880263])

def test_closed_ring(ring_network):    
    net = ring_network    
    net.switch.closed = True
    sc.calc_sc(net, ip=True, ith=True)
    
    assert np.allclose(net.res_trafo_sc.ikss_lv_ka.values, [0.4745379023])
    assert np.allclose(net.res_line_sc.ikss_ka.values, [0.26039497, 0.20831598, 0.36590236])
    
if __name__ == '__main__':  
    pytest.main(['-xs'])


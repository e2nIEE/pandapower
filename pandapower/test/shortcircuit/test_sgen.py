# -*- coding: utf-8 -*-

# Copyright (c) 2016-2017 by University of Kassel and Fraunhofer Institute for Wind Energy and
# Energy System Technology (IWES), Kassel. All rights reserved. Use of this source code is governed
# by a BSD-style license that can be found in the LICENSE file.

import numpy as np
import pytest

import pandapower as pp
import pandapower.shortcircuit as sc

@pytest.fixture
def wind_park_example():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=110., index=1)
    b2 = pp.create_bus(net, vn_kv=110., index=2)
    b3 = pp.create_bus(net, vn_kv=110., index=3)
    b4 = pp.create_bus(net, vn_kv=110., index=4)
    pp.create_ext_grid(net, b1, s_sc_max_mva=20*110*np.sqrt(3), rx_max=0.1)
    
    pp.create_line_from_parameters(net, from_bus=b1, to_bus=b2, length_km=100, r_ohm_per_km=0.120, x_ohm_per_km=0.393, c_nf_per_km=0, max_i_ka=10)
    pp.create_line_from_parameters(net, from_bus=b1, to_bus=b3, length_km=50, r_ohm_per_km=0.120, x_ohm_per_km=0.393, c_nf_per_km=0, max_i_ka=10)
    pp.create_line_from_parameters(net, from_bus=b2, to_bus=b3, length_km=50, r_ohm_per_km=0.120, x_ohm_per_km=0.393, c_nf_per_km=0, max_i_ka=10)
    pp.create_line_from_parameters(net, from_bus=b3, to_bus=b4, length_km=25, r_ohm_per_km=0.120, x_ohm_per_km=0.393, c_nf_per_km=0, max_i_ka=10)
    
    pp.create_sgen(net, b2, p_kw=100e3, sn_kva=100e3)
    pp.create_sgen(net, b3, p_kw=50e3, sn_kva=50e3)
    pp.create_sgen(net, b4, p_kw=50e3, sn_kva=50e3)
    net.sgen["k"] = 1.2
    return net
        
def test_wind_park(wind_park_example):
    net = wind_park_example
    sc.calc_sc(net, ip=True)
    np.isclose(net.res_bus_sc.ikss_ka.at[2], 3.9034, rtol=1e-4)
    np.isclose(net.res_bus_sc.ip_ka.at[2], 7.3746, rtol=1e-4)
    
if __name__ == '__main__':
    pytest.main([])



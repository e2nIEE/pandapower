# -*- coding: utf-8 -*-

# Copyright (c) 2016-2020 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import numpy as np
import pytest

import pandapower as pp
import pandapower.shortcircuit as sc

@pytest.fixture
def motor_net():
    net = pp.create_empty_network()
    b1 = pp.create_bus(net, vn_kv=0.4)
    b2 = pp.create_bus(net, vn_kv=0.4)
    pp.create_ext_grid(net, b1, s_sc_max_mva=10., rx_max=0.1)
    pp.create_line_from_parameters(net, from_bus=b1, to_bus=b2, length_km=1.,
                                   r_ohm_per_km=0.32, c_nf_per_km=0,
                                   x_ohm_per_km=0.07, max_i_ka=1)
    pp.create_motor(net, b2, pn_mech_mw=0.5, lrc_pu=7., vn_kv=0.45, rx=0.4,
                    efficiency_n_percent=95, cos_phi_n=0.9)
    return net

def test_motor(motor_net):
    net = motor_net
    
    net.motor.in_service = False
    sc.calc_sc(net)
    assert np.isclose(net.res_bus_sc.ikss_ka.at[0], 14.433757337, rtol=1e-4)
    assert np.isclose(net.res_bus_sc.ikss_ka.at[1], 0.7618582511, rtol=1e-4)


    net.motor.in_service = True    
    sc.calc_sc(net)

    assert np.isclose(net.res_bus_sc.ikss_ka.at[0], 14.743809197, rtol=1e-4)
    assert np.isclose(net.res_bus_sc.ikss_ka.at[1], 5.626994278, rtol=1e-4)

if __name__ == '__main__':
    pytest.main([__file__])

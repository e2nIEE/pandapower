# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 17:21:05 2017

@author: thurner
"""

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
    pp.create_sgen(net, b2, p_kw=0, sn_kva=1000.)
    return net

def test_max_sgen(one_line_one_static_generator):
    net = one_line_one_static_generator
    sc.runsc(net, case="max")
    np.isclose(net.res_bus_sc.ikss_max_ka.at[1], 0.069801849155)
    
def test_min_sgen(one_line_one_static_generator):
    net = one_line_one_static_generator
    sc.runsc(net, case="min")
    np.isclose(net.res_bus_sc.ikss_min_ka.at[1], 0.05773139511)
    
if __name__ == '__main__':
#    net = one_line_one_static_generator()
#    sc.runsc(net)
    pytest.main(['test_sgen.py'])    
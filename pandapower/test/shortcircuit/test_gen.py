# -*- coding: utf-8 -*-

# Copyright (c) 2016-2019 by University of Kassel and Fraunhofer Institute for Energy Economics
# and Energy System Technology (IEE), Kassel. All rights reserved.


import pandas as pd
import pytest

import pandapower as pp
import pandapower.shortcircuit as sc
import numpy as np

@pytest.fixture
def one_line_one_generator():
    net = pp.create_empty_network(sn_mva=2)
    b1 = pp.create_bus(net, vn_kv=10.)
    b2 = pp.create_bus(net, vn_kv=10.)
    b3 = pp.create_bus(net, vn_kv=10.)
    pp.create_bus(net, vn_kv=0.4, in_service=False)
    pp.create_gen(net, b1, vn_kv=10.5, xdss_pu=0.2, rdss_pu=0.001, cos_phi=0.8, p_mw=0.1, sn_mva=2.5)
    pp.create_gen(net, b1, vn_kv=10.5, xdss_pu=0.2, rdss_pu=0.001, cos_phi=0.8, p_mw=0.1, sn_mva=2.5)
    l = pp.create_line_from_parameters(net, b2, b1, length_km=1.0, max_i_ka=0.29,
                                       r_ohm_per_km=0.1548, x_ohm_per_km=0.0816814, c_nf_per_km=165)
    net.line.loc[l, "endtemp_degree"] = 165
    pp.create_switch(net, b3, b1, et="b")
    return net

def test_max_gen(one_line_one_generator):
    net = one_line_one_generator
    sc.calc_sc(net, case="max")
    assert abs(net.res_bus_sc.ikss_ka.at[0] - 1.5395815) < 1e-7
    assert abs(net.res_bus_sc.ikss_ka.at[2] - 1.5395815) < 1e-7
    assert abs(net.res_bus_sc.ikss_ka.at[1] - 1.5083952) < 1e-7
    assert pd.isnull(net.res_bus_sc.ikss_ka.at[3])

def test_min_gen(one_line_one_generator):
    net = one_line_one_generator
    sc.calc_sc(net, case="min")
    assert abs(net.res_bus_sc.ikss_ka.at[0] - 1.3996195) < 1e-7
    assert abs(net.res_bus_sc.ikss_ka.at[2] - 1.3996195) < 1e-7
    assert abs(net.res_bus_sc.ikss_ka.at[1] - 1.3697407) < 1e-7
    assert pd.isnull(net.res_bus_sc.ikss_ka.at[3])

def test_max_gen_fault_impedance(one_line_one_generator):
    net = one_line_one_generator
    sc.calc_sc(net, case="max", r_fault_ohm=2, x_fault_ohm=10)
    assert abs(net.res_bus_sc.ikss_ka.at[0] - 0.4450868) < 1e-7
    assert abs(net.res_bus_sc.ikss_ka.at[1] - 0.4418823) < 1e-7
    assert abs(net.res_bus_sc.ikss_ka.at[2] - 0.4450868) < 1e-7
    assert pd.isnull(net.res_bus_sc.ikss_ka.at[3])


def test_rdss_estimations():
    net = pp.create_empty_network(sn_mva=1)
    b1 = pp.create_bus(net, vn_kv=0.4)
    g1 = pp.create_gen(net, b1, vn_kv=0.4, xdss_pu=0.1, cos_phi=0.8, p_mw=0.1, sn_mva=0.1)
    b2 = pp.create_bus(net, vn_kv=20.)
    g2 = pp.create_gen(net, b2, vn_kv=21., xdss_pu=0.2, cos_phi=0.85, p_mw=0.1, sn_mva=2.5)
    b3 = pp.create_bus(net, vn_kv=20.)
    g3 = pp.create_gen(net, b3, vn_kv=30., xdss_pu=0.25, cos_phi=0.9, p_mw=0.1, sn_mva=150)
    
    sc.calc_sc(net, case="max")
    assert np.isclose(net.res_bus_sc.ikss_ka.at[b1], 1.5130509845)
    net.gen.rdss_pu.at[g1] = net.gen.xdss_pu.at[g1] * 0.15
    sc.calc_sc(net, case="max")
    assert np.isclose(net.res_bus_sc.ikss_ka.at[b1], 1.5130509845)
    
    sc.calc_sc(net, case="max")
    assert np.isclose(net.res_bus_sc.ikss_ka.at[b2], 0.37894052506)
    net.gen.rdss_pu.at[g2] = net.gen.xdss_pu.at[g2] * 0.07
    sc.calc_sc(net, case="max")
    assert np.isclose(net.res_bus_sc.ikss_ka.at[b2], 0.37894052506)
    
    sc.calc_sc(net, case="max")
    assert np.isclose(net.res_bus_sc.ikss_ka.at[b3], 12.789334853)
    net.gen.rdss_pu.at[g3] = net.gen.xdss_pu.at[g3] * 0.05
    sc.calc_sc(net, case="max")
    assert np.isclose(net.res_bus_sc.ikss_ka.at[b3], 12.789334853)

if __name__ == '__main__':
    pytest.main(['test_gen.py'])
